# Use this when multiple markers appear in the same frames
# you get one set of outputs per marker in a single pass.

# CLI wrapper that scans marker folders and calls the multi-marker batch runner

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from ReadMarker_FromNPZ_multi import Config, run_pipeline_multi


# Optional defaults so the script can be run directly (e.g. via IDE run button)
# without specifying command-line arguments. Set these to the paths you use most
# often. Leave them as None to force providing the values via CLI arguments.
DEFAULT_MULTI_DIR: Path | None = Path("MULTI_TRACK")
DEFAULT_NPZ_PATH: Path | None = Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\tracking\tracking_rec_test_20251028_091809.npz")


def _iter_marker_dirs(base_dir: Path, marker_filename: str) -> Iterable[Path]:
    """
    Yield marker directories inside base_dir that contain the marker file.
    Sorted for deterministic processing order.
    """
    for entry in sorted(base_dir.iterdir()):
        marker_file = entry / marker_filename
        if entry.is_dir() and marker_file.is_file():
            yield entry


def _resolve_relative(base: Path, candidate: Path, fallback: Path | None = None) -> Path:
    if candidate.is_absolute():
        return candidate

    candidate_path = base / candidate
    if fallback is not None and not candidate_path.exists():
        fallback_path = fallback / candidate
        if fallback_path.exists():
            return fallback_path.resolve()

    return candidate_path.resolve()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    default_pipeline = Path(__file__).with_name("ReadMarker_SV_multi.py")

    parser = argparse.ArgumentParser(
        description=(
            "Run ReadMarker_FromNPZ on multiple marker folders that share the same NPZ frames."
        )
    )
    parser.add_argument(
        "multi_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Directory containing subfolders with marker assets (marker_sta.npy, etc.). "
            "If omitted, DEFAULT_MULTI_DIR is used when set."
        ),
    )
    parser.add_argument(
        "--npz",
        dest="npz_path",
        type=Path,
        required=False,
        default=None,
        help="Shared NPZ file with frames to process. If omitted, DEFAULT_NPZ_PATH is used when set.",
    )
    parser.add_argument(
        "--pipeline",
        dest="pipeline_script",
        type=Path,
        default=default_pipeline,
        help=f"Path to ReadMarker_SV pipeline script (default: {default_pipeline}).",
    )
    parser.add_argument(
        "--marker-file",
        default="marker_sta.npy",
        help="Marker definition filename expected inside each marker folder.",
    )
    parser.add_argument(
        "--output-prefix",
        default="npz_marker_results",
        help="Prefix for the per-marker output folder (suffix '_MARKER_N' is added automatically).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index used when numbering marker result folders.",
    )
    parser.add_argument(
        "--data-key",
        default=None,
        help="Optional override for Config.data_key.",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Optional override for Config.image_key.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames processed per marker.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Enable interactive overlay display during processing.",
    )
    parser.add_argument(
        "--save-overlays",
        type= bool,
        default=True,
        #action="store_true",
        help="Request overlay image saving (uses Config.save_overlays).",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Request raw grayscale frame saving (uses Config.save_raw_images).",
    )
    parser.add_argument(
        "--skip-detections",
        action="store_true",
        help="Disable writing detections_data.npz (sets Config.save_detections = False).",
    )
    parser.add_argument(
        "--save-dot-tables",
        action="store_true",
        help="Enable debug dot table saving (uses Config.save_dot_tables).",
    )
    parser.add_argument(
        "--skip-timing-stats",
        action="store_true",
        help="Disable timing summaries (sets Config.save_timing_stats = False).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    script_dir = Path(__file__).resolve().parent

    default_config = Config()

    multi_dir_candidate = args.multi_dir
    if multi_dir_candidate is None:
        if DEFAULT_MULTI_DIR is None:
            print(
                "[ERROR] multi_dir not provided. Pass it on the command line or set DEFAULT_MULTI_DIR in the script.",
                file=sys.stderr,
            )
            return 2
        multi_dir_candidate = DEFAULT_MULTI_DIR

    base_dir = _resolve_relative(script_dir, Path(multi_dir_candidate).expanduser())
    if not base_dir.is_dir():
        print(f"[ERROR] Multi-marker directory not found: {base_dir}", file=sys.stderr)
        return 2

    npz_candidate = args.npz_path
    if npz_candidate is None:
        if DEFAULT_NPZ_PATH is None:
            print(
                "[ERROR] --npz not provided. Pass it on the command line or set DEFAULT_NPZ_PATH in the script.",
                file=sys.stderr,
            )
            return 2
        npz_candidate = DEFAULT_NPZ_PATH

    npz_path = _resolve_relative(
        script_dir,
        Path(npz_candidate).expanduser(),
        fallback=base_dir,
    )
    if not npz_path.is_file():
        print(f"[ERROR] NPZ file not found: {npz_path}", file=sys.stderr)
        return 2

    pipeline_path = _resolve_relative(script_dir, args.pipeline_script.expanduser(), fallback=base_dir)
    if not pipeline_path.is_file():
        print(f"[ERROR] Pipeline script not found: {pipeline_path}", file=sys.stderr)
        return 2

    marker_dirs = list(_iter_marker_dirs(base_dir, args.marker_file))
    if not marker_dirs:
        print(f"[WARN] No marker folders with '{args.marker_file}' found in {base_dir}", file=sys.stderr)
        return 1

    marker_entries: List[dict[str, object]] = []
    for idx, marker_dir in enumerate(marker_dirs, start=args.start_index):
        sta_path = marker_dir / args.marker_file
        output_dir = marker_dir / "tracking" / f"{args.output_prefix}_MARKER_{idx}"
        marker_entries.append(
            {
                "name": marker_dir.name,
                "sta_path": sta_path,
                "output_dir": output_dir,
            }
        )

    output_root = base_dir / "tracking"
    output_root.mkdir(parents=True, exist_ok=True)

    config = Config(
        npz_path=npz_path,
        data_key=args.data_key,
        image_key=args.image_key if args.image_key is not None else default_config.image_key,
        sta_path=marker_entries[0]["sta_path"],
        output_dir=output_root / args.output_prefix,
        max_frames=args.max_frames,
        show=args.show,
        pipeline_script=pipeline_path,
        save_overlays=args.save_overlays,
        save_raw_images=args.save_raw,
        save_detections=not args.skip_detections,
        save_dot_tables=args.save_dot_tables,
        save_timing_stats=not args.skip_timing_stats,
    )

    marker_specs = [(entry["sta_path"], entry["output_dir"]) for entry in marker_entries]

    try:
        exit_code = run_pipeline_multi(config, marker_specs)
    except Exception as exc:
        print(f"[ERROR] Multi-marker pipeline failed with exception: {exc}", file=sys.stderr)
        return 1

    print("\n=== Multi-marker run summary ===")
    for entry in marker_entries:
        summary_file = entry["output_dir"] / "detections_summary.npy"
        status = "OK" if summary_file.exists() else "FAILED"
        print(f"{entry['name']}: {status} -> {entry['output_dir']}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
