from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import json
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
    [0.0, 0.0, 1.0, 50.0],   # probe origin 50 mm "above" marker plane
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

    # Scale reconstruction by nearest-neighbour spacing (same idea as sfm_runner.py) :contentReference[oaicite:7]{index=7}
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
) -> None:
    tracking_path = trial_dir / "tracking_results.npy"
    if not tracking_path.is_file():
        print(f"[POSE] No tracking_results.npy in {trial_dir}, skipping")
        return

    # Load tracking results produced by track_from_map.run_tracking_from_records
    payload = np.load(tracking_path, allow_pickle=True).item()

    timestamps = np.asarray(payload["timestamps"], dtype=float)
    R_CM_all = np.asarray(payload["rotation_matrices"], dtype=float)
    t_CM_all = np.asarray(payload["translations_mm"], dtype=float)

    N = R_CM_all.shape[0]

    # Build homogeneous ^C T_M for each frame (marker in camera frame)
    T_C_M_all = np.repeat(np.eye(4)[None, :, :], N, axis=0)
    T_C_M_all[:, :3, :3] = R_CM_all
    T_C_M_all[:, :3, 3]  = t_CM_all

    # Load calibration: probe transducer in marker frame (^M T_P)
    T_M_P = _load_T_M_P(marker_to_probe_path)
    if T_M_P.shape != (4, 4):
        raise ValueError(f"T_M_P must be 4x4, got {T_M_P.shape}")

    # It's necessary to interpolate between frames since the sampling of the probe is different than the camera.


    # Compose probe poses in camera frame: ^C T_P = ^C T_M · ^M T_P
    T_C_P_all = T_C_M_all @ T_M_P

    # Extract probe translations and rotations
    probe_trans_mm = T_C_P_all[:, :3, 3]
    probe_rots = R.from_matrix(T_C_P_all[:, :3, :3])
    probe_euler_rad = probe_rots.as_euler("xyz", degrees=False)
    probe_euler_deg = np.degrees(probe_euler_rad)
    probe_quat = probe_rots.as_quat()

    # US frame poses
    T_C_U_all = None
    T_U0_U_all = None

    T_P_U = _load_T_P_U(probe_to_us_path)
    if T_P_U is not None:
        # ^C T_U = ^C T_M · ^M T_P · ^P T_U
        T_C_U_all = T_C_M_all @ T_M_P @ T_P_U

        # Rebase to first US frame
        T_C_U0 = T_C_U_all[0]
        T_C_U0_inv = np.linalg.inv(T_C_U0)
        T_U0_U_all = T_C_U0_inv[None, :, :] @ T_C_U_all


    # Build output payload
    out = {
        "timestamps": timestamps,
        "T_C_M": T_C_M_all,                 # marker in camera frame
        "T_C_P": T_C_P_all,                 # probe in camera frame
        "probe_trans_mm": probe_trans_mm,   # (N, 3)
        "probe_euler_rad": probe_euler_rad, # (N, 3)
        "probe_euler_deg": probe_euler_deg, # (N, 3)
        "probe_quat": probe_quat,           # (N, 4)
    }

    if T_C_U_all is not None:
        out["T_C_U"] = T_C_U_all            # US in camera frame
        out["T_U0_U"] = T_U0_U_all          # US in first-US frame (U*)

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
        compose_probe_poses_for_trial(
            trial_dir,
            marker_to_probe_path=calib_cfg.marker_to_probe_path,
            probe_to_us_path=calib_cfg.probe_to_us_path,
        )


if __name__ == "__main__":
    cfg = AutomationConfig(
        sta_and_map_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag_tris"),
        motion_npz_dir=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag_tris\accuracy_increment_x"),
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
