from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import json
from NonUniform_Filtering import smooth_nonuniform_gorry
from typing import Tuple


from ReadMarker_FromNPZ import Config as NpzDetectConfig, run_pipeline as run_npz_detection
from sfm import (FilterConfig, bootstrap_with_fallback, bundle_adjustment_pyceres, compute_median_mean_reprojection_error, compute_reprojection_errors, align_state_to_object_frame, incremental_pose_estimation,load_calibration, load_frame_observations, preprocess_detections)
from track_from_map import (MapTracker,load_camera_intrinsics, load_detection_records,load_map, run_tracking_from_records)
from ReadMarker_FromNPZ_multi import Config as MultiConfig, run_pipeline_multi

# Hardcoded calibration transforms
# ^M T_P : probe transducer in MARKER frame
HARD_CODED_T_M_P = np.array([
    [1.0, 0.0, 0.0,  0.0],
    [0.0, 1.0, 0.0,  0.0],
    [0.0, 0.0, 1.0, 50.0],   
    [0.0, 0.0, 0.0,  1.0],
], dtype=float)

# ^P T_U : ultrasound image frame in PROBE frame
HARD_CODED_T_P_U = np.eye(4, dtype=float)
# or set to None
# HARD_CODED_T_P_U = None


@dataclass
class AutomationConfig:
    # Folder where marker_sta.npy lives; map will be stored in sfm_outputs/ here
    sta_and_map_dir: Path

    # Folder containing .npz recordings of the OBJECT MOTION
    motion_npz_dir: Path

    # NPZ recording to use for building the MAP, if the map is missing
    mapping_npz: Optional[Path]

    # Camera calibration
    camera_matrix_path: Path
    distortion_path: Path

    # Path to ReadMarker_SV.py
    pipeline_script: Path

    # Optional names / relative paths
    sta_filename: str = "marker_sta.npy"
    map_relpath: Path = Path("mapping\\sfm_outputs") / "marker_map_aligned.npz"

    # Optional glob to select recordings inside motion_npz_dir
    motion_npz_glob: str = "*.npz"

    # If you want to skip subfolders inside motion_npz_dir, list them here:
    # skip_motion_subdirs: Tuple[str, ...] = (
    #     "accuracy_increment_x",
    #     "accuracy_increment_y",
    #     "accuracy_increment_z",
    #     "mapping",
    # )

    # Marker IDs used to define object frame in SfM
    origin_marker_id: int = 17
    x_axis_marker_id: int = 59
    y_axis_marker_id: int = 21

    # Physical spacing between adjacent cells (mm) for map scaling
    cell_size_mm: float = 8.0

    r: int = 5
    sigma: float = 4.0

    id_base: int = 1

@dataclass
class MultiMarkerAutomationConfig:
    # Root that contains one subfolder per marker, each with its own marker_sta.npy
    # and its own sfm_outputs/marker_map_aligned.npz (built with sfm.py and sfm_runner.py).
    marker_root_dir: Path

    # Folder containing NPZ recordings where multiple markers appear together
    motion_npz_dir: Path

    # Camera calibration (shared by all markers)
    camera_matrix_path: Path
    distortion_path: Path

    # Path to ReadMarker_SV_multi.py
    pipeline_script: Path

    # File naming conventions
    marker_sta_filename: str = "marker_sta.npy"
    map_relpath: Path = Path("mapping\\sfm_outputs") / "marker_map_aligned.npz"

    # Glob for motion recordings
    motion_npz_glob: str = "*.npz"

    r: int = 5
    sigma: float = 3.0

    per_marker_config_filename: str = "marker_config.json"


@dataclass
class ProbeCalibrationConfig:
    # 4x4 npy/npz with ^M T_P (probe transducer in marker frame)
    marker_to_probe_path: Path | None = None

    # 4x4 npy/npz with ^P T_U (US image in probe frame)
    probe_to_us_path: Path | None = None


def _iter_marker_dirs_multi(cfg: MultiMarkerAutomationConfig) -> List[Path]:
    marker_dirs: List[Path] = []
    for entry in sorted(cfg.marker_root_dir.iterdir()):
        if not entry.is_dir():
            continue
        sta_path = entry / cfg.marker_sta_filename
        if sta_path.is_file():
            marker_dirs.append(entry)
    if not marker_dirs:
        raise FileNotFoundError(
            f"No marker folders with '{cfg.marker_sta_filename}' found in {cfg.marker_root_dir}"
        )
    return marker_dirs

def _ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    
def _load_T_M_P(path: Path | None) -> np.ndarray:
    """Load ^M T_P (marker->probe) from file or use hardcoded default."""
    if path is None:
        return HARD_CODED_T_M_P
    T = np.asarray(np.load(path), dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_M_P must be 4x4, got {T.shape}")
    return T


def _load_T_P_U(path: Path | None):
    """Load ^P T_U (probe->US) from file or use hardcoded default / None."""
    if path is None:
        # If you don't want a default US frame, return None instead.
        return HARD_CODED_T_P_U  # or return None
    T = np.asarray(np.load(path), dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_P_U must be 4x4, got {T.shape}")
    return T


# ----------------------------- #
# Ultrasound + interpolation helpers

def _load_npz_data_records(npz_path: Path, data_key: str = "data") -> np.ndarray | None:
    """Load the object-array of dict records stored under `data_key` in a recording NPZ.

    Returns None if the key is missing.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        if data_key not in z.files:
            return None
        return z[data_key]


def _split_records_by_device(data_np: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    """Split `data_np` (array of dict-like records) into clarius / dart / vega lists.

    Mirrors preprocessing_v1_AS.split_data().
    """
    clarius_list: list[dict] = []
    dart_list: list[dict] = []
    vega_list: list[dict] = []
    for datum in data_np:
        dev = None
        if isinstance(datum, dict):
            dev = datum.get("dev")
        if dev == "clarius":
            clarius_list.append(datum)
        elif dev == "dart":
            dart_list.append(datum)
        elif dev == "vega":
            vega_list.append(datum)
        else:
            # keep silent; many recordings store other record types
            pass
    return clarius_list, dart_list, vega_list


def _get_valid_us_times(
    clarius_data: list[dict],
    max_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (clarius_data_arr, image_times_s) cropped to times we have tracking for.

    Mirrors preprocessing_v1_AS.get_valid_US_times().
    """
    if not clarius_data:
        return np.asarray([], dtype=object), np.asarray([], dtype=float)

    ti_ns = np.empty((len(clarius_data),), dtype=float)
    for i, rec in enumerate(clarius_data):
        if not isinstance(rec, dict) or "ti" not in rec:
            raise KeyError("Clarius record missing 'ti' timestamp field.")
        ti_ns[i] = float(rec["ti"])

    image_times = ti_ns / 1e9
    image_times = image_times - image_times[0]  # zero to first US frame
    ok = np.where(image_times < float(max_time_s))[0]
    image_times = image_times[: len(ok)]
    clarius_arr = np.asarray(clarius_data, dtype=object)[ok]
    return clarius_arr, image_times


def _auto_window_size_from_times(times: np.ndarray, poly_order: int) -> int | None:
    """Pick a reasonable odd smoothing window based on estimated frame-rate.

    Uses the idea of preprocessing_v1_AS.get_window_size().
    Returns None if filtering should be skipped.
    """
    times = np.asarray(times, dtype=float)
    if times.size < (poly_order + 3):
        return None

    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None

    fps = float(np.mean(1.0 / dt))
    nearest = int(round(fps))
    if nearest < (poly_order + 3):
        nearest = poly_order + 3

    # make odd
    if nearest % 2 == 0:
        lower = nearest - 1
        higher = nearest + 1
        nearest = lower if abs(fps - lower) <= abs(fps - higher) else higher

    # cap by N
    if nearest > times.size:
        nearest = times.size if times.size % 2 == 1 else times.size - 1

    if nearest < (poly_order + 3):
        return None

    return nearest


def _filter_camera_poses_nonuniform(
    times: np.ndarray,
    trans_mm: np.ndarray,
    rots: R,
    window_size: int | None,
    poly_order: int,
) -> tuple[np.ndarray, R]:
    """Non-uniform Savitzky–Golay smoothing for translation and rotation.

    Implements the quaternion-component approach used in preprocessing_v1_AS.filter_camera_data().
    """
    times = np.asarray(times, dtype=float)
    trans_mm = np.asarray(trans_mm, dtype=float)

    if window_size is None:
        window_size = _auto_window_size_from_times(times, poly_order)

    if window_size is None or window_size <= poly_order or window_size > times.size:
        return trans_mm, rots

    if window_size % 2 != 1:
        window_size += 1
        if window_size > times.size:
            window_size = times.size if times.size % 2 == 1 else times.size - 1

    # Quaternions with continuity
    quats = rots.as_quat().copy()
    for i in range(1, len(quats)):
        if float(np.dot(quats[i], quats[i - 1])) < 0.0:
            quats[i] = -quats[i]

    # Filter translations
    x_f = smooth_nonuniform_gorry(times, trans_mm[:, 0], window_size, poly_order, 0)
    y_f = smooth_nonuniform_gorry(times, trans_mm[:, 1], window_size, poly_order, 0)
    z_f = smooth_nonuniform_gorry(times, trans_mm[:, 2], window_size, poly_order, 0)
    trans_f = np.stack([x_f, y_f, z_f], axis=1)

    # Filter quaternion components
    q_f = np.zeros_like(quats, dtype=float)
    for j in range(4):
        q_f[:, j] = smooth_nonuniform_gorry(times, quats[:, j], window_size, poly_order, 0)

    q_norm = np.linalg.norm(q_f, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q_f = q_f / q_norm

    rots_f = R.from_quat(q_f)
    return trans_f, rots_f


def _dedupe_strictly_increasing(times: np.ndarray, trans_mm: np.ndarray, rots: R) -> tuple[np.ndarray, np.ndarray, R]:
    """Remove duplicate / non-increasing timestamps (required by Slerp)."""
    times = np.asarray(times, dtype=float)
    keep = np.ones(times.shape[0], dtype=bool)
    keep[1:] = np.diff(times) > 0
    if np.all(keep):
        return times, trans_mm, rots
    return times[keep], trans_mm[keep], rots[keep]


def _interpolate_camera_to_times(
    times_src: np.ndarray,
    trans_src_mm: np.ndarray,
    rots_src: R,
    times_query: np.ndarray,
) -> tuple[np.ndarray, R]:
    """Interpolate SE(3) samples to `times_query` (linear trans + SLERP rot)."""
    times_src, trans_src_mm, rots_src = _dedupe_strictly_increasing(times_src, trans_src_mm, rots_src)

    times_query = np.asarray(times_query, dtype=float)
    trans_q = np.empty((times_query.size, 3), dtype=float)
    for j in range(3):
        trans_q[:, j] = np.interp(times_query, times_src, trans_src_mm[:, j])

    slerp = Slerp(times_src, rots_src)
    rots_q = slerp(times_query)
    return trans_q, rots_q




# 1) MAP STAGE: build the map if needed (detection + SfM)
def ensure_map(cfg: AutomationConfig) -> Path:
    """Return path to marker_map_aligned.npz, building it if necessary."""
    sta_path = cfg.sta_and_map_dir / cfg.sta_filename
    _ensure_file(sta_path, "STA matrix")

    map_path = cfg.sta_and_map_dir / cfg.map_relpath
    sfm_out_dir = map_path.parent

    if map_path.is_file():
        print(f"[MAP] Using existing map: {map_path}")
        return map_path

    # Map missing -> user must supply a mapping NPZ
    if cfg.mapping_npz is None:
        raise FileNotFoundError(f"[MAP] Map file not found at {map_path}.\n" f"A mapping recording (.npz) is required to build the map.")
    _ensure_file(cfg.mapping_npz, "Mapping NPZ recording")

    print(f"[MAP] No map found, building map from mapping recording: {cfg.mapping_npz}")

    # ---- 1A) Run detection on mapping NPZ ----
    mapping_output_dir = cfg.sta_and_map_dir / "mapping"
    mapping_output_dir.mkdir(parents=True, exist_ok=True)

    det_cfg = NpzDetectConfig(npz_path=cfg.mapping_npz, sta_path=sta_path, output_dir=mapping_output_dir, pipeline_script=cfg.pipeline_script, id_base=cfg.id_base,r=cfg.r,sigma=cfg.sigma)
    print(f"[MAP] Running detection on mapping NPZ...")
    run_npz_detection(det_cfg)
    
    detections_path = mapping_output_dir / "detections_data.npz"
    if not detections_path.is_file():
        det_cfg = NpzDetectConfig(npz_path=cfg.mapping_npz, sta_path=sta_path, output_dir=mapping_output_dir, pipeline_script=cfg.pipeline_script, id_base=cfg.id_base, r=cfg.r,sigma=cfg.sigma)
        print(f"[MAP] Running detection on mapping NPZ...")
        run_npz_detection(det_cfg)
    else:
        print(f"[MAP] Reusing existing mapping detections: {detections_path}")

    _ensure_file(detections_path, "Mapping detections_data.npz")

    # ---- 1B) Run SfM (equivalent to sfm_runner.main) ----
    print(f"[MAP] Running SfM to build 3D marker map...")
    calibration = load_calibration(cfg.camera_matrix_path, cfg.distortion_path)
    raw_frames = load_frame_observations(detections_path)
    print(f"[SfM] Loaded {len(raw_frames)} raw frames from mapping detections")

    filter_cfg = FilterConfig(min_shared_markers=40, min_bootstrap_inlier_ratio=0.8)
    filtered_frames, diagnostics = preprocess_detections(raw_frames, calibration, config=filter_cfg)
    print(f"[SfM] Preprocessing diagnostics: {diagnostics}")

    state, tri_result, bootstrap_diag = bootstrap_with_fallback(filtered_frames, calibration, config=filter_cfg)
    print(f"[SfM] Bootstrap triangulated {len(tri_result.marker_ids)} IDs " f"using frames {bootstrap_diag['selected_pair']}")

    sfm_out_dir.mkdir(parents=True, exist_ok=True)

    incremental_pose_estimation(state, checkpoint_path=sfm_out_dir / "checkpoint_incremental.pkl")

    median_before, mean_before = compute_median_mean_reprojection_error(state)
    print(f"[SfM] Median reprojection error (pre-BA): {median_before}, mean: {mean_before}")

    ba_summary = bundle_adjustment_pyceres(state, checkpoint_path=sfm_out_dir / "checkpoint_ba.pkl")
    print(f"[SfM] Bundle adjustment message: {str(getattr(ba_summary, 'message', '')).strip()}")

    median_after, mean_after = compute_median_mean_reprojection_error(state)
    print(f"[SfM] Median reprojection error (post-BA): {median_after}, mean: {mean_after}")

    # Align map to object frame and scale to physical units
    rotation, origin = align_state_to_object_frame(state, origin_marker_id=cfg.origin_marker_id, x_axis_marker_id=cfg.x_axis_marker_id, y_axis_marker_id=cfg.y_axis_marker_id)

    marker_ids = np.array(sorted(state.marker_positions.keys()), dtype=np.int32)
    marker_points = np.stack([state.marker_positions[mid] for mid in marker_ids], axis=0)

    pose_ids = np.array(sorted(state.poses.keys()), dtype=np.int32)
    pose_rotations = np.stack([state.poses[fid].rotation for fid in pose_ids], axis=0)
    pose_translations = np.stack([state.poses[fid].translation for fid in pose_ids], axis=0)

    # Scale reconstruction by nearest-neighbour spacing (same idea as sfm_runner.py) 
    if cfg.cell_size_mm is not None and marker_points.shape[0] >= 2:
        from scipy.spatial import cKDTree  # only needed here

        tree = cKDTree(marker_points)
        dists, _ = tree.query(marker_points, k=2)  # nearest neighbour
        nearest = dists[:, 1]
        finite = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        if finite.size:
            median_spacing = float(np.median(finite))
            scale = cfg.cell_size_mm / median_spacing
            marker_points *= scale
            pose_translations *= scale
            origin *= scale
            print(f"[SfM] Rescaled map: median spacing {median_spacing:.6f} -> " f"{cfg.cell_size_mm:.6f} mm (scale factor {scale:.6f})")
        else:
            print("[SfM] Skipping scaling: insufficient valid nearest-neighbour distances.")
    else:
        print("[SfM] Skipping scaling: need at least two markers and a valid cell_size_mm.")

    # Save aligned, scaled map
    np.savez(map_path, marker_ids=marker_ids, marker_points=marker_points, rotation=rotation, origin=origin, pose_ids=pose_ids, pose_rotations=pose_rotations, pose_translations=pose_translations)
    print(f"[MAP] Saved aligned marker map to {map_path}")

    return map_path

# 2) MOTION STAGE: detection + tracking for all motion NPZs
def _iter_motion_npz_files(cfg: AutomationConfig) -> List[Path]:
    files = sorted(cfg.motion_npz_dir.glob(cfg.motion_npz_glob))
    # Avoid re-processing detection/tracking outputs
    files = [f for f in files if f.name != "detections_data.npz"]
    return files

# If you have to run multiple subfolders with skip list as Tuple in the config, use this version:
# def _iter_motion_npz_files(cfg: AutomationConfig) -> List[Path]:
#     """
#     Find all motion NPZ files to process.

#     - Includes any NPZ directly under motion_npz_dir.
#     - For each immediate subfolder of motion_npz_dir:
#         * if its name is in cfg.skip_motion_subdirs -> skip it;
#         * otherwise include all NPZs matching cfg.motion_npz_glob inside it.
#     """
#     files: List[Path] = []

#     # 1) NPZs directly in motion_npz_dir
#     for f in sorted(cfg.motion_npz_dir.glob(cfg.motion_npz_glob)):
#         if f.is_file() and f.name != "detections_data.npz":
#             files.append(f)

#     # 2) One level of subfolders
#     for sub in sorted(cfg.motion_npz_dir.iterdir()):
#         if not sub.is_dir():
#             continue

#         if sub.name in cfg.skip_motion_subdirs:
#             print(f"[MOTION] Skipping subfolder (skip list): {sub.name}")
#             continue

#         for npz in sorted(sub.glob(cfg.motion_npz_glob)):
#             if npz.is_file():
#                 files.append(npz)

#     return files


def run_motion_detection_and_tracking(cfg: AutomationConfig, map_path: Path) -> None:
    """Run detection + tracking for all motion NPZ recordings."""
    sta_path = cfg.sta_and_map_dir / cfg.sta_filename
    _ensure_file(sta_path, "STA matrix")

    # Load map + camera intrinsics
    map_data = load_map(map_path)
    camera_matrix, distortion = load_camera_intrinsics(cfg.camera_matrix_path, cfg.distortion_path)

    motion_files = _iter_motion_npz_files(cfg)
    if not motion_files:
        raise FileNotFoundError(f"No motion NPZ files found in {cfg.motion_npz_dir} matching {cfg.motion_npz_glob}")

    print(f"[MOTION] Found {len(motion_files)} NPZ recordings to process.")

    for idx, npz_path in enumerate(motion_files, start=1):
        trial_dir = cfg.motion_npz_dir / f"trial_{idx:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        # Save link to the source recording so later stages can load ultrasound (clarius) stream
        meta = {
            "source_npz": str(npz_path.resolve()),
            "source_name": npz_path.name,
        }
        try:
            (trial_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[MOTION] WARNING: could not write trial_meta.json in {trial_dir}: {e}")
        print(f"\n[MOTION] === Processing {npz_path.name} -> {trial_dir.name} ===")

        # ---- 2A) Detection on this NPZ (ReadMarker_FromNPZ) ----
        det_cfg = NpzDetectConfig(npz_path=npz_path, sta_path=sta_path, output_dir=trial_dir, pipeline_script=cfg.pipeline_script, id_base=cfg.id_base, r=cfg.r, sigma=cfg.sigma)
        print(f"[MOTION] Running detection for {npz_path.name}...")
        run_npz_detection(det_cfg)

        detections_path = trial_dir / "detections_data.npz"
        if not detections_path.is_file():
            print(f"[MOTION] Skipping {npz_path.name}: detections_data.npz not found in {trial_dir}")
            continue

        # ---- 2B) Tracking for this trial (track_from_map) ----
        tracker = MapTracker(map_data, camera_matrix, distortion)
        detection_records = load_detection_records(detections_path)
        print(f"[MOTION] Loaded {len(detection_records)} detection records from {detections_path}")

        run_tracking_from_records(
            tracker,
            detection_records,
            image_lookup=None,        # set up load_npz_image_lookup(...) if you want playback
            display=False,            # True: OpenCV visualization
            axis_length=50.0,
            min_markers=6,
            report_every=10,
            quit_key="q",
            use_pose_prior=True,
            display_delay_ms=1,
            plot_output_dir=trial_dir,
            results_output_path=trial_dir / "tracking_results.npy",
        )
        print(f"[MOTION] Tracking done for {npz_path.name}. Results saved in {trial_dir}")

def run_multi_detection_and_tracking_for_npz(
    cfg: MultiMarkerAutomationConfig,
    npz_path: Path,
) -> None:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Motion NPZ not found: {npz_path}")

    # List of marker folders (each contains marker_sta.npy and a map, or enough info to build it)
    marker_dirs = _iter_marker_dirs_multi(cfg)

    # --- 0) Make sure each marker has a map (build it from its mapping NPZ if needed) --- #
    for marker_dir in marker_dirs:
        per_cfg_path = marker_dir / cfg.per_marker_config_filename
        if not per_cfg_path.is_file():
            raise FileNotFoundError(f"No per-marker config file '{cfg.per_marker_config_filename}' in {marker_dir}")

        per_cfg = json.loads(per_cfg_path.read_text())

        # Resolve mapping NPZ path (allow relative paths in the JSON)
        mapping_npz_path = Path(per_cfg["mapping_npz"])
        if not mapping_npz_path.is_absolute():
            mapping_npz_path = (marker_dir / mapping_npz_path).resolve()

        # Build a single-marker AutomationConfig for this marker
        single_cfg = AutomationConfig(
            sta_and_map_dir=marker_dir,
            motion_npz_dir=cfg.motion_npz_dir,  # not used for map building
            mapping_npz=mapping_npz_path,
            camera_matrix_path=cfg.camera_matrix_path,
            distortion_path=cfg.distortion_path,
            pipeline_script=cfg.pipeline_script,
            origin_marker_id=per_cfg["origin_marker_id"],
            x_axis_marker_id=per_cfg["x_axis_marker_id"],
            y_axis_marker_id=per_cfg["y_axis_marker_id"],
            cell_size_mm=per_cfg["cell_size_mm"],
            r=cfg.r,
            sigma=cfg.sigma,
            id_base=per_cfg["id_base"],  # <- per-tool ID base (1001, 2001, ...)
        )

        # This will:
        #  - reuse the map if map_path exists
        #  - otherwise run detection + SfM to build marker_map_aligned.npz
        ensure_map(single_cfg)

    # Common camera intrinsics (shared by all markers)
    camera_matrix, distortion = load_camera_intrinsics(cfg.camera_matrix_path, cfg.distortion_path)

    # --- 1) Multi-marker detection on this NPZ --- #
    # Output root specific to this NPZ
    npz_output_root = cfg.motion_npz_dir / f"{npz_path.stem}_multi"
    npz_output_root.mkdir(parents=True, exist_ok=True)

    # MultiConfig from ReadMarker_FromNPZ_multi:
    # We just need some sta_path; it is only used by run_pipeline (single-marker)
    # and for consistency, we pass the first marker's STA.
    first_marker_sta = marker_dirs[0] / cfg.marker_sta_filename

    multi_config = MultiConfig(
        npz_path=npz_path,
        data_key=None,
        image_key="img",
        sta_path=first_marker_sta,
        output_dir=npz_output_root,
        max_frames=None,
        show=False,
        pipeline_script=cfg.pipeline_script,
        save_overlays=False,
        save_raw_images=False,
        save_detections=True,
        save_dot_tables=False,
        save_timing_stats=True,
        r=cfg.r,
        sigma=cfg.sigma,
    )

    # Build sta_inputs: (sta_path, per-marker-output-dir)
    sta_inputs = []
    for marker_dir in marker_dirs:
        sta_path = marker_dir / cfg.marker_sta_filename
        marker_output_dir = npz_output_root / marker_dir.name
        sta_inputs.append((sta_path, marker_output_dir))

    print(f"[MULTI] Running multi-marker detection for {npz_path.name}...")
    run_pipeline_multi(multi_config, sta_inputs)

    # --- 2) For each marker: run tracking using its own map + detections --- #
    for marker_dir in marker_dirs:
        marker_name = marker_dir.name
        map_path = marker_dir / cfg.map_relpath
        if not map_path.is_file():
            print(f"[MULTI] WARNING: map not found for marker '{marker_name}': {map_path}")
            print("         → check marker_config.json and mapping NPZ for this marker.")
            continue

        detections_path = npz_output_root / marker_name / "detections_data.npz"
        if not detections_path.is_file():
            print(f"[MULTI] WARNING: no detections for marker '{marker_name}' in {npz_path.name}")
            continue

        print(f"[MULTI] Tracking marker '{marker_name}' in {npz_path.name}...")
        map_data = load_map(map_path)
        tracker = MapTracker(map_data, camera_matrix, distortion)

        detection_records = load_detection_records(detections_path)

        marker_output_dir = npz_output_root / marker_name
        marker_output_dir.mkdir(parents=True, exist_ok=True)

        results_output_path = marker_output_dir / "tracking_results.npy"

        run_tracking_from_records(
            tracker,
            detection_records,
            image_lookup=None,        # or use load_npz_image_lookup(...) if you want playback
            display=False,            # True -> OpenCV overlays
            axis_length=50.0,
            min_markers=6,
            report_every=10,
            quit_key="q",
            use_pose_prior=True,
            display_delay_ms=1,
            plot_output_dir=marker_output_dir,
            results_output_path=results_output_path,
        )

        print(f"[MULTI] Done: {marker_name} → {results_output_path}")


def compose_probe_poses_for_trial(
    trial_dir: Path,
    marker_to_probe_path: Path | None,
    probe_to_us_path: Path | None,
    *,
    source_npz_path: Path | None = None,
    npz_data_key: str = "data",
    apply_nonuniform_filter: bool = True,
    filter_window_size: int | None = None,
    filter_poly_order: int = 3,
) -> None:
    tracking_path = trial_dir / "tracking_results.npy"
    if not tracking_path.is_file():
        print(f"[POSE] No tracking_results.npy in {trial_dir}, skipping")
        return

    # Load tracking results (marker pose in camera frame at camera timestamps)
    payload = np.load(tracking_path, allow_pickle=True).item()
    timestamps = np.asarray(payload["timestamps"], dtype=float)
    R_CM_all = np.asarray(payload["rotation_matrices"], dtype=float)
    t_CM_all = np.asarray(payload["translations_mm"], dtype=float)
    N = R_CM_all.shape[0]

    # Build homogeneous ^C T_M for each frame (marker in camera frame)
    T_C_M_all = np.repeat(np.eye(4)[None, :, :], N, axis=0)
    T_C_M_all[:, :3, :3] = R_CM_all
    T_C_M_all[:, :3, 3] = t_CM_all

    # Load calibration transforms
    # ^M T_P : probe transducer in MARKER frame
    T_M_P = _load_T_M_P(marker_to_probe_path)
    if T_M_P.shape != (4, 4):
        raise ValueError(f"T_M_P must be 4x4, got {T_M_P.shape}")

    # ^P T_U : ultrasound image frame in PROBE frame (optional)
    T_P_U = _load_T_P_U(probe_to_us_path)

    # Load ultrasound (clarius) timestamps from the source NPZ
    # Get the source NPZ explicitly, or read it from trial_meta.json saved in run_motion_detection_and_tracking.
    if source_npz_path is None:
        meta_path = trial_dir / "trial_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                if "source_npz" in meta:
                    source_npz_path = Path(meta["source_npz"])
            except Exception as e:
                print(f"[POSE] WARNING: failed reading {meta_path}: {e}")

    clarius_arr = None
    us_times = None

    if source_npz_path is not None and source_npz_path.is_file():
        data_np = _load_npz_data_records(source_npz_path, data_key=npz_data_key)
        if data_np is None:
            print(f"[POSE] NOTE: '{npz_data_key}' key not found in {source_npz_path.name}; skipping US interpolation.")
        else:
            clarius_data, _, _ = _split_records_by_device(data_np)
            if not clarius_data:
                print(f"[POSE] NOTE: no 'clarius' records found in {source_npz_path.name}; skipping US interpolation.")
            else:
                clarius_arr, us_times = _get_valid_us_times(
                    clarius_data, max_time_s=float(np.max(timestamps))
                )

    # Filter camera->marker poses on the camera timebase
    cam_rots = R.from_matrix(R_CM_all)
    cam_trans = t_CM_all

    if apply_nonuniform_filter:
        cam_trans_filt, cam_rots_filt = _filter_camera_poses_nonuniform(timestamps, cam_trans, cam_rots, window_size=filter_window_size, poly_order=filter_poly_order)
    else:
        cam_trans_filt, cam_rots_filt = cam_trans, cam_rots

    # Build filtered ^C T_M at the SAME camera timestamps
    T_C_M_filt = np.repeat(np.eye(4)[None, :, :], N, axis=0)
    T_C_M_filt[:, :3, :3] = cam_rots_filt.as_matrix()
    T_C_M_filt[:, :3, 3] = cam_trans_filt


    # I have to check that everything below here is correct

    # Compose probe poses at camera timestamps
    # ^C T_P = ^C T_M · ^M T_P
    T_C_P_all = T_C_M_all @ T_M_P
    T_C_P_filt = T_C_M_filt @ T_M_P

    # Extract probe translations and rotations (raw)
    probe_trans_mm = T_C_P_all[:, :3, 3]
    probe_rots = R.from_matrix(T_C_P_all[:, :3, :3])
    probe_euler_rad = probe_rots.as_euler("xyz", degrees=False)
    probe_euler_deg = np.degrees(probe_euler_rad)
    probe_quat = probe_rots.as_quat()

    # Extract probe translations and rotations (filtered, camera timestamps)
    probe_trans_mm_filt = T_C_P_filt[:, :3, 3]
    probe_rots_filt = R.from_matrix(T_C_P_filt[:, :3, :3])
    probe_quat_filt = probe_rots_filt.as_quat()

    # Interpolate filtered marker/probe poses to ultrasound timestamps (if available)
    T_C_M_us = None
    T_C_P_us = None
    if us_times is not None and np.asarray(us_times).size > 0:
        trans_us, rots_us = _interpolate_camera_to_times(
            times_src=timestamps,
            trans_src_mm=cam_trans_filt,
            rots_src=cam_rots_filt,
            times_query=us_times,
        )

        T_C_M_us = np.repeat(np.eye(4)[None, :, :], us_times.size, axis=0)
        T_C_M_us[:, :3, :3] = rots_us.as_matrix()
        T_C_M_us[:, :3, 3] = trans_us

        # Compose probe poses at US times: ^C T_P_us = ^C T_M_us · ^M T_P
        T_C_P_us = T_C_M_us @ T_M_P

    # compute US frame poses (^C T_U) and rebase to first frame
    T_C_U_all = None
    T_U0_U_all = None
    T_C_U_us = None
    T_U0_U_us = None

    if T_P_U is not None:
        # Camera-timebase: ^C T_U = ^C T_M_filt · ^M T_P · ^P T_U
        T_C_U_all = T_C_M_filt @ T_M_P @ T_P_U
        T_C_U0 = T_C_U_all[0]
        T_U0_U_all = np.linalg.inv(T_C_U0)[None, :, :] @ T_C_U_all

        # US-timebase (if we have US interpolation):
        if T_C_M_us is not None:
            T_C_U_us = T_C_M_us @ T_M_P @ T_P_U
            T_C_U0_us = T_C_U_us[0]
            T_U0_U_us = np.linalg.inv(T_C_U0_us)[None, :, :] @ T_C_U_us

    out = {
        "timestamps": timestamps,
        "T_C_M": T_C_M_all,                 # marker in camera frame (raw)
        "T_C_P": T_C_P_all,                 # probe in camera frame (raw)
        "probe_trans_mm": probe_trans_mm,   # (N, 3)
        "probe_euler_rad": probe_euler_rad, # (N, 3)
        "probe_euler_deg": probe_euler_deg, # (N, 3)
        "probe_quat": probe_quat,           # (N, 4)
        "T_C_M_filt": T_C_M_filt,           # marker in camera frame (filtered)
        "T_C_P_filt": T_C_P_filt,           # probe in camera frame (filtered)
        "probe_trans_mm_filt": probe_trans_mm_filt,
        "probe_quat_filt": probe_quat_filt,
    }

    if T_C_U_all is not None:
        out["T_C_U"] = T_C_U_all            # US in camera frame (camera timestamps)
        out["T_U0_U"] = T_U0_U_all          # US in first-US frame (camera timestamps)

    if T_C_M_us is not None:
        # Poses interpolated to ultrasound frame timestamps (clarius)
        out["us_times_s"] = us_times
        out["T_C_M_us"] = T_C_M_us
        out["T_C_P_us"] = T_C_P_us

        # Optional metadata only (no image payload)
        out["num_us_frames"] = int(us_times.size)
        out["source_npz"] = str(source_npz_path) if source_npz_path is not None else None

        if T_C_U_us is not None:
            out["T_C_U_us"] = T_C_U_us
            out["T_U0_U_us"] = T_U0_U_us

    np.save(trial_dir / "probe_tracking.npy", out, allow_pickle=True)
    print(f"[POSE] Saved probe_tracking.npy in {trial_dir}")

def run_full_automation(cfg: AutomationConfig) -> None:
    """
    End-to-end:
      - ensure map exists (build it with mapping NPZ if missing)
      - run detection + tracking on all motion NPZ recordings
    """
    map_path = ensure_map(cfg)
    run_motion_detection_and_tracking(cfg, map_path)


def run_multi_marker_motion_automation(cfg: MultiMarkerAutomationConfig) -> None:
    npz_files = sorted(cfg.motion_npz_dir.glob(cfg.motion_npz_glob))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files in {cfg.motion_npz_dir} matching {cfg.motion_npz_glob}"
        )

    print(f"[MULTI] Found {len(npz_files)} motion NPZ recordings.")
    for npz_path in npz_files:
        print(f"\n[MULTI] === Processing {npz_path.name} ===")
        run_multi_detection_and_tracking_for_npz(cfg, npz_path)

def run_pose_composition_for_all_trials(
    cfg: AutomationConfig,
    calib_cfg: ProbeCalibrationConfig,
) -> None:
    # Trials are named trial_01, trial_02, ...
    for trial_dir in sorted(cfg.motion_npz_dir.glob("trial_*")):
        if not trial_dir.is_dir():
            continue
        # Try to recover the source recording for this trial (to align to ultrasound timestamps)
        source_npz_path = None
        meta_path = trial_dir / "trial_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                if "source_npz" in meta:
                    source_npz_path = Path(meta["source_npz"])
            except Exception as e:
                print(f"[POSE] WARNING: failed reading {meta_path}: {e}")

        compose_probe_poses_for_trial(
            trial_dir,
            marker_to_probe_path=calib_cfg.marker_to_probe_path,
            probe_to_us_path=calib_cfg.probe_to_us_path,
            source_npz_path=source_npz_path,
        )


if __name__ == "__main__":
    cfg = AutomationConfig(
        sta_and_map_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\3x3_2x3tag"),
        motion_npz_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\3x3_2x3tag\accuracy_rot_increment_z"),
        mapping_npz=None,
        camera_matrix_path=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_mtx.npy"),
        distortion_path=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_dist.npy"),
        pipeline_script=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\ReadMarker_SV.py"),
    )
    run_full_automation(cfg)

    # multi_cfg = MultiMarkerAutomationConfig(
    #     marker_root_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\multiple_tracking"),  # contains MarkerA, MarkerB, ...
    #     motion_npz_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\multiple_tracking\tracking"),
    #     camera_matrix_path=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_mtx.npy"),
    #     distortion_path=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_dist.npy"),
    #     pipeline_script=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\ReadMarker_SV_multi.py"),
    # )
    # run_multi_marker_motion_automation(multi_cfg)

    # calib_cfg = ProbeCalibrationConfig(
    #     marker_to_probe_path=r"C:\Users\samue\Desktop\Slicer\Results0412\T_M_P_from_slicer.npy",  # if None, use HARD_CODED_T_M_P
    #     probe_to_us_path=None,      # if None, use HARD_CODED_T_P_U
    # )
    # run_pose_composition_for_all_trials(cfg, calib_cfg)
