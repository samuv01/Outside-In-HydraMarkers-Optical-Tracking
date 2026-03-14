# NPZ runner that uses the shared pipeline for several STAs

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Any, Sequence, List
import csv
import sys
import types

import cv2
import matplotlib.pyplot as plt
import numpy as np



def _load_pipeline_functions(script_path: Path) -> Tuple[Any, Any, Any | None]:
    """
    Load the pipeline helpers from ReadMarker_SV.py without executing its
    script-only section.
    """
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {script_path!s}")

    source = script_path.read_text(encoding="utf-8")
    sentinel = "# Load the grayscale board image"

    try:
        cutoff = source.index(sentinel)
    except ValueError as exc:
        raise RuntimeError(
            "Could not locate the sentinel comment that separates the pipeline "
            "definitions from the script entry point in ReadMarker_SV.py. "
            "Please ensure the file still contains the expected comment."
        ) from exc

    module_name = "ReadMarker_SV_embedded"
    module = types.ModuleType(module_name)
    module.__file__ = str(script_path)
    module.__dict__["__name__"] = module_name
    module.__dict__["__package__"] = None
    sys.modules[module_name] = module
    exec(source[:cutoff], module.__dict__)  # noqa: S102 - controlled exec of trusted code

    try:
        read_marker = module.__dict__["read_marker"]
        draw_overlay = module.__dict__["_draw_overlay"]
    except KeyError as exc:
        raise RuntimeError(
            "Failed to extract required functions from ReadMarker_SV.py. "
            "Please verify the script still defines read_marker and _draw_overlay."
        ) from exc

    read_marker_multi = module.__dict__.get("read_marker_multi")

    return read_marker, draw_overlay, read_marker_multi


FrameEntry = Dict[str, Any]


@dataclass
class Config:
    npz_path: Path = Path(r"path_to_frames.npz")
    data_key: str | None = None
    image_key: str = "img"
    sta_path: Path = Path(r"marker_sta.npy")
    output_dir: Path = Path("npz_marker_results")
    max_frames: int | None = None
    show: bool = False
    pipeline_script: Path = Path("ReadMarker_SV_multi.py")
    prefix: str = "frame"
    save_overlays: bool = False  # Change to false to disable saving overlays
    save_raw_images: bool = False  # Save grayscale input frames alongside overlays
    save_detections: bool = True  # Save per-frame detections to NPZ for SfM
    save_dot_tables: bool = False  # Save per-frame dot tables from the debug output
    report_timings: bool = True  # Print per-frame timing breakdowns
    save_timing_stats: bool = True  # Persist timing arrays and summary statistics
    r: int = 5
    sigma: float = 3.0


def _normalize_entry(entry: Any) -> FrameEntry:
    """
    Convert an NPZ entry into a standard dictionary.
    Supports dicts, structured arrays, and simple namespaces.
    """
    if isinstance(entry, dict):
        return dict(entry)

    if hasattr(entry, "item"):
        value = entry.item()
        if isinstance(value, dict):
            return dict(value)

    if hasattr(entry, "_asdict"):
        return dict(entry._asdict())

    if isinstance(entry, np.void) and entry.dtype.names:
        return {name: entry[name] for name in entry.dtype.names}

    raise TypeError(
        "Unsupported NPZ frame entry type. Expected dict-like objects with an "
        "'img' key."
    )


def _iter_npz_frames(npz_path: Path, data_key: str | None) -> Iterator[FrameEntry]:
    with np.load(npz_path, allow_pickle=True) as npz:
        keys = list(npz.files)
        if not keys:
            raise ValueError(f"No arrays stored in {npz_path!s}")

        target_key = data_key
        if target_key is None:
            if len(keys) == 1:
                target_key = keys[0]
            else:
                raise ValueError(
                    f"Multiple arrays in {npz_path!s}: {keys}. Set Config.data_key to choose one."
                )

        if target_key not in npz:
            raise KeyError(f"Array {target_key!r} not found in {npz_path!s}. Available keys: {keys}")

        dataset = npz[target_key]

    if isinstance(dataset, np.ndarray) and dataset.dtype == object:
        for entry in dataset:
            yield _normalize_entry(entry)
    elif isinstance(dataset, np.ndarray) and dataset.ndim == 3:
        for frame in dataset:
            yield {"img": frame}
    else:
        raise TypeError(
            "Unsupported NPZ dataset shape or dtype. Expected an object array of dict-like "
            "frames or a stack of grayscale images."
        )


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Images must be single-channel grayscale or BGR.")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype in (np.float32, np.float64):
        finite_max = float(np.nanmax(image)) if np.isfinite(image).any() else 0.0
        if finite_max <= 1.0:
            scaled = np.clip(image, 0.0, 1.0) * 255.0
        else:
            scaled = np.clip(image, 0.0, 255.0)
        return scaled.astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _build_output_name(frame: FrameEntry, index: int) -> str:
    if "framenumber" in frame:
        return f"{int(frame['framenumber']):06d}"
    if "timestamp" in frame:
        timestamp = float(frame["timestamp"])
        return f"ts_{timestamp:012.3f}".replace(".", "_")
    return f"{index:06d}"


def run_pipeline_multi(config: Config, sta_inputs: Sequence[Any]) -> int:
    """
    Process an NPZ sequence while identifying multiple markers using the shared
    corner detection pipeline from ReadMarker_SV_multi.py.
    Each entry in sta_inputs can be either a Path-like object pointing to a marker
    STA file, or a (sta_path, output_dir) tuple to override the output location.
    """
    if not sta_inputs:
        raise ValueError("sta_inputs must contain at least one marker STA path.")

    npz_path = config.npz_path
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path!s}")

    _, draw_overlay, read_marker_multi_fn = _load_pipeline_functions(config.pipeline_script)
    if read_marker_multi_fn is None:
        raise RuntimeError(
            "Pipeline script does not expose read_marker_multi. "
            "Ensure Config.pipeline_script points to ReadMarker_SV_multi.py."
        )

    marker_states: List[dict[str, Any]] = []
    total_expected_points = 0
    base_output_root = config.output_dir

    for idx, entry in enumerate(sta_inputs, start=1):
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            sta_path = Path(entry[0]).expanduser()
            output_dir = Path(entry[1]).expanduser()
        else:
            sta_path = Path(entry).expanduser()
            marker_folder_name = sta_path.parent.name or sta_path.stem or f"marker_{idx}"
            output_dir = base_output_root / marker_folder_name

        if not sta_path.is_file():
            raise FileNotFoundError(f"STA pattern file not found: {sta_path!s}")

        sta_array = np.load(sta_path)
        marker_name = sta_path.parent.name or sta_path.stem or f"marker_{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_pts = (sta_array.shape[0] + 1) * (sta_array.shape[1] + 1)
        total_expected_points += expected_pts

        id_base = idx * 1000 + 1

        marker_states.append(
            {
                "index": idx,
                "name": marker_name,
                "sta": sta_array,
                "sta_path": sta_path,
                "output_dir": output_dir,
                "id_base": id_base,
                "summaries": [],
                "detections": [],
                "dot_tables": [],
                "timing_accum": defaultdict(list),
            }
        )

    if not marker_states:
        print("No valid marker definitions supplied.")
        return 1

    r = config.r
    sigma = config.sigma
    expect_n = max(100, int(total_expected_points * 10))
    sta_arrays = [state["sta"] for state in marker_states]
    marker_labels = [state["name"] for state in marker_states]
    marker_id_bases = [state["id_base"] for state in marker_states]

    frames_processed = 0
    for index, frame in enumerate(_iter_npz_frames(npz_path, config.data_key)):
        if config.max_frames is not None and index >= config.max_frames:
            break

        if config.image_key not in frame:
            raise KeyError(
                f"Frame {index} does not contain the {config.image_key!r} key. "
                "Update the configuration to specify the correct field."
            )

        image = np.asarray(frame[config.image_key])
        image = _ensure_grayscale(image)

        return_debug = (
            config.save_overlays
            or config.show
            or config.save_dot_tables
            or config.report_timings
            or config.save_timing_stats
        )

        if return_debug:
            multi_output, debug_bundle = read_marker_multi_fn(
                image=image,
                sta_list=sta_arrays,
                sta_labels=marker_labels,
                r=r,
                expect_n=expect_n,
                sigma=sigma,
                contrast_threshold=10,
                id_base=1,
                id_bases=marker_id_bases,
                column_major_ids=True,
                return_debug=True,
            )
            markers_debug = debug_bundle.get("markers") if isinstance(debug_bundle, dict) else None
            if not isinstance(markers_debug, list):
                markers_debug = [None] * len(marker_states)
        else:
            multi_output = read_marker_multi_fn(
                image=image,
                sta_list=sta_arrays,
                sta_labels=marker_labels,
                r=r,
                expect_n=expect_n,
                sigma=sigma,
                contrast_threshold=10,
                id_base=1,
                id_bases=marker_id_bases,
                column_major_ids=True,
                return_debug=False,
            )
            markers_debug = [None] * len(marker_states)

        frames_processed += 1
        name = _build_output_name(frame, index)
        frame_idx = index
        timestamp = float("nan")
        if isinstance(frame, dict):
            if "index" in frame:
                frame_idx = int(frame["index"])
            elif "frame" in frame:
                frame_idx = int(frame["frame"])
            timestamp = float(frame.get("timestamp", float("nan")))

        if return_debug:
            multi_results = multi_output  # list of (pt_list, edges, debug)
        else:
            multi_results = multi_output  # list of (pt_list, edges)

        for marker_state, result, marker_debug in zip(marker_states, multi_results, markers_debug):
            if return_debug:
                pt_list, edges, overlay_debug = result
            else:
                pt_list, edges = result
                overlay_debug = None

            needs_overlay = config.save_overlays or config.show
            overlay = (
                draw_overlay(image, pt_list, edges, dot_debug=overlay_debug) if needs_overlay else None
            )

            marker_output_dir = marker_state["output_dir"]
            overlay_path = None
            raw_path = None
            if config.save_overlays and overlay is not None:
                overlay_path = marker_output_dir / f"{config.prefix}_{name}.png"
                cv2.imwrite(str(overlay_path), overlay)
            if config.save_raw_images:
                raw_image_uint8 = _to_uint8(image)
                raw_path = marker_output_dir / f"{config.prefix}_{name}_raw.png"
                cv2.imwrite(str(raw_path), raw_image_uint8)

            timings: Dict[str, float] | None = None
            effective_debug = overlay_debug if overlay_debug is not None else marker_debug
            if isinstance(effective_debug, dict):
                raw_timings = effective_debug.get("timings")
                if isinstance(raw_timings, dict):
                    timings = {}
                    for key, value in raw_timings.items():
                        try:
                            value_float = float(value)
                        except (TypeError, ValueError):
                            continue
                        key_str = str(key)
                        timings[key_str] = value_float
                        if config.save_timing_stats:
                            marker_state["timing_accum"][key_str].append(value_float)

            valid_mask = ~np.isnan(pt_list[:, 2]) if pt_list.size else np.array([], dtype=bool)
            identified = int(np.count_nonzero(valid_mask))
            detections = (
                np.column_stack((pt_list[valid_mask, 2], pt_list[valid_mask, 1], pt_list[valid_mask, 0])).astype(np.float64)
                if identified > 0
                else np.empty((0, 3), dtype=np.float64)
            )

            marker_state["summaries"].append(
                {
                    "index": index,
                    "name": name,
                    "detected_points": int(pt_list.shape[0]),
                    "identified_points": identified,
                    "overlay_path": str(overlay_path) if overlay_path is not None else None,
                    "raw_path": str(raw_path) if raw_path is not None else None,
                    "timestamp": timestamp,
                    "timings": timings,
                }
            )

            if config.save_detections:
                marker_state["detections"].append(
                    {
                        "index": frame_idx,
                        "name": name,
                        "timestamp": timestamp,
                        "detections": detections,
                    }
                )

            if config.save_dot_tables and isinstance(effective_debug, dict):
                dot_tables = effective_debug.get("dot_tables")
                dot_tables_payload = None
                if dot_tables is not None:
                    dot_tables_payload = np.asarray(dot_tables, dtype=object)
                marker_state["dot_tables"].append(
                    {
                        "index": frame_idx,
                        "name": name,
                        "timestamp": timestamp,
                        "dot_tables": dot_tables_payload,
                    }
                )

            overlay_status = str(overlay_path) if overlay_path is not None else "overlay disabled"
            raw_status = str(raw_path) if raw_path is not None else "raw disabled"
            print(
                f"[{index}] marker={marker_state['name']} frame={name} "
                f"detected={pt_list.shape[0]} identified={identified} "
                f"-> overlay: {overlay_status}, raw: {raw_status}"
            )
            if isinstance(timings, dict) and timings:
                timing_parts = []
                for stage, value in sorted(timings.items()):
                    timing_parts.append(f"{stage}={value * 1000.0:.2f} ms")
                print("    timings: " + ", ".join(timing_parts))

            if config.show and overlay is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                plt.title(f"{marker_state['name']} - {name} - detected {identified}")
                plt.axis("off")
                plt.show()
                plt.close()

    if frames_processed == 0:
        print("No frames processed. Check the NPZ content and filters.")
        return 1

    exit_code = 0
    for marker_state in marker_states:
        summaries = marker_state["summaries"]
        output_dir = marker_state["output_dir"]
        if not summaries:
            print(f"No detections recorded for marker {marker_state['name']}.")
            exit_code = max(exit_code, 1)
            continue

        summary_path = output_dir / "detections_summary.npy"
        np.save(summary_path, summaries, allow_pickle=True)
        print(f"[{marker_state['name']}] Saved detection summary to {summary_path!s}")

        if config.save_detections and marker_state["detections"]:
            detections_path = output_dir / "detections_data.npz"
            np.savez_compressed(
                detections_path,
                frames=np.asarray(marker_state["detections"], dtype=object),
            )
            print(f"[{marker_state['name']}] Saved detection data to {detections_path!s}")

        if config.save_dot_tables and marker_state["dot_tables"]:
            dot_tables_path = output_dir / "dot_tables_data.npz"
            np.savez_compressed(
                dot_tables_path,
                frames=np.asarray(marker_state["dot_tables"], dtype=object),
            )
            print(f"[{marker_state['name']}] Saved dot tables to {dot_tables_path!s}")

        if config.save_timing_stats and marker_state["timing_accum"]:
            timing_arrays = {
                stage: np.asarray(values, dtype=np.float64)
                for stage, values in marker_state["timing_accum"].items()
                if values
            }
            if timing_arrays:
                timings_path = output_dir / "timings_data.npz"
                np.savez_compressed(timings_path, **timing_arrays)
                print(f"[{marker_state['name']}] Saved raw timing data to {timings_path!s}")

                stats_path = output_dir / "timings_summary.csv"
                with stats_path.open("w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["stage", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "count"])
                    for stage in sorted(timing_arrays.keys()):
                        arr_ms = timing_arrays[stage] * 1000.0
                        writer.writerow(
                            [
                                stage,
                                float(arr_ms.mean()),
                                float(np.median(arr_ms)),
                                float(arr_ms.std(ddof=0)),
                                float(arr_ms.min()),
                                float(arr_ms.max()),
                                int(arr_ms.size),
                            ]
                        )
                print(f"[{marker_state['name']}] Saved timing summary to {stats_path!s}")

    return exit_code


def run_pipeline(config: Config) -> int:
    npz_path = config.npz_path
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path!s}")

    sta_path = config.sta_path
    if not sta_path.is_file():
        raise FileNotFoundError(f"STA pattern file not found: {sta_path!s}")

    read_marker, draw_overlay, _ = _load_pipeline_functions(config.pipeline_script)

    sta = np.load(sta_path)
    r = 5
    sigma = 3.0
    expect_n = int(10*(sta.shape[0] + 1) * (sta.shape[1] + 1)) # number of expected corners * 10
    # careful because this works with only 1 marker

    config.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    dot_table_records: list[dict[str, Any]] = []
    timing_accum: Dict[str, list[float]] = defaultdict(list)
    for index, frame in enumerate(_iter_npz_frames(npz_path, config.data_key)):
    
        if config.max_frames is not None and index >= config.max_frames:
            break

        if config.image_key not in frame:
            raise KeyError(
                f"Frame {index} does not contain the {config.image_key!r} key. "
                "Update the configuration to specify the correct field."
            )

        image = np.asarray(frame[config.image_key])
        image = _ensure_grayscale(image)

        return_debug = config.save_overlays or config.show or config.save_dot_tables or config.report_timings
        marker_output = read_marker(image,sta,r=r,expect_n=expect_n,sigma=sigma,return_debug=return_debug)
        if return_debug:
            pt_list, edges, overlay_debug = marker_output
        else:
            pt_list, edges = marker_output
            overlay_debug = None

        needs_overlay = config.save_overlays or config.show
        overlay = draw_overlay(image, pt_list, edges, dot_debug=overlay_debug) if needs_overlay else None
        name = _build_output_name(frame, index)
        overlay_path = None
        raw_path = None
        if config.save_overlays and overlay is not None:
            overlay_path = config.output_dir / f"{config.prefix}_{name}.png"
            cv2.imwrite(str(overlay_path), overlay)
        if config.save_raw_images:
            raw_image_uint8 = _to_uint8(image)
            raw_path = config.output_dir / f"{config.prefix}_{name}_raw.png"
            cv2.imwrite(str(raw_path), raw_image_uint8)

        timings: Dict[str, float] | None = None
        if overlay_debug is not None:
            raw_timings = overlay_debug.get("timings")
            if isinstance(raw_timings, dict):
                timings = {}
                for key, value in raw_timings.items():
                    try:
                        value_float = float(value)
                    except (TypeError, ValueError):
                        continue
                    key_str = str(key)
                    timings[key_str] = value_float
                    if config.save_timing_stats:
                        timing_accum[key_str].append(value_float)

        frame_idx = index
        timestamp = float("nan")
        if isinstance(frame, dict):
            if "index" in frame:
                frame_idx = int(frame["index"])
            elif "frame" in frame:
                frame_idx = int(frame["frame"])
            timestamp = float(frame.get("timestamp", float("nan")))

        valid_mask = ~np.isnan(pt_list[:, 2]) if pt_list.size else np.array([], dtype=bool)
        identified = int(np.count_nonzero(valid_mask))
        detections = (
            np.column_stack((pt_list[valid_mask, 2], pt_list[valid_mask, 1], pt_list[valid_mask, 0])).astype(np.float64)
            if identified > 0
            else np.empty((0, 3), dtype=np.float64)
        )

        summaries.append(
            {
                "index": index,
                "name": name,
                "detected_points": int(pt_list.shape[0]),
                "identified_points": identified,
                "overlay_path": str(overlay_path) if overlay_path is not None else None,
                "raw_path": str(raw_path) if raw_path is not None else None,
                "timestamp": float(frame.get("timestamp", float("nan"))),
                "timings": timings,
            }
        )

        if config.save_detections:
            detection_records.append(
                {
                    "index": frame_idx,
                    "name": name,
                    "timestamp": timestamp,
                    "detections": detections,                
                }
            )

        if config.save_dot_tables:
            dot_tables_payload = None
            if overlay_debug is not None:
                dot_tables = overlay_debug.get("dot_tables")
                if dot_tables is not None:
                    dot_tables_payload = np.asarray(dot_tables, dtype=object)

            dot_table_records.append(
                {
                    "index": frame_idx,
                    "name": name,
                    "timestamp": timestamp,
                    "dot_tables": dot_tables_payload,
                }
            )

        overlay_status = str(overlay_path) if overlay_path is not None else "overlay disabled"
        raw_status = str(raw_path) if raw_path is not None else "raw disabled"
        print(
            f"[{index}] frame={name} detected={pt_list.shape[0]} identified={identified} "
            f"-> overlay: {overlay_status}, raw: {raw_status}"
        )
        if isinstance(timings, dict) and timings:
            timing_parts = []
            for stage, value in sorted(timings.items()):
                timing_parts.append(f"{stage}={value * 1000.0:.2f} ms")
            print("    timings: " + ", ".join(timing_parts))

        if config.show and overlay is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{name} - detected {identified}")
            plt.axis("off")
            plt.show()
            plt.close()

    if not summaries:
        print("No frames processed. Check the NPZ content and filters.")
        return 1

    summary_path = config.output_dir / "detections_summary.npy"
    np.save(summary_path, summaries, allow_pickle=True)
    print(f"Saved detection summary to {summary_path!s}")

    if config.save_detections and detection_records:
        detections_path = config.output_dir / "detections_data.npz"
        np.savez_compressed(detections_path, frames=np.asarray(detection_records, dtype=object))
        print(f"Saved detection data to {detections_path!s}")

    if config.save_dot_tables and dot_table_records:
        dot_tables_path = config.output_dir / "dot_tables_data.npz"
        np.savez_compressed(dot_tables_path, frames=np.asarray(dot_table_records, dtype=object))
        print(f"Saved dot tables to {dot_tables_path!s}")

    if config.save_timing_stats and timing_accum:
        timing_arrays = {stage: np.asarray(values, dtype=np.float64) for stage, values in timing_accum.items() if values}
        if timing_arrays:
            timings_path = config.output_dir / "timings_data.npz"
            np.savez_compressed(timings_path, **timing_arrays)
            print(f"Saved raw timing data to {timings_path!s}")

            stats_path = config.output_dir / "timings_summary.csv"
            with stats_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["stage", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "count"])
                for stage in sorted(timing_arrays.keys()):
                    arr_ms = timing_arrays[stage] * 1000.0
                    writer.writerow(
                        [
                            stage,
                            float(arr_ms.mean()),
                            float(np.median(arr_ms)),
                            float(arr_ms.std(ddof=0)),
                            float(arr_ms.min()),
                            float(arr_ms.max()),
                            int(arr_ms.size),
                        ]
                    )
            print(f"Saved timing summary to {stats_path!s}")

    return 0


def main() -> int:
    # Update these paths before running the script.
    input_dir = Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\3x45_3x3tag")
    output_dir = input_dir / "tracking/npz_marker_results_deodorant_tracking_x_15"
    npz_path = input_dir / "tracking/tracking_rec_test_20251023_120002.npz"
    sta_path = input_dir / "marker_sta.npy"
    pipeline_path = Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\ReadMarker_SV_multi.py")

    config = Config(
        npz_path=npz_path,
        sta_path=sta_path,
        # data_key=None,
        # image_key="img",
        output_dir=output_dir,
        # max_frames=None,
        # show=False,
        pipeline_script=pipeline_path,
        # prefix="frame",
    )
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())

