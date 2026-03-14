from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ReadMarker_SV import read_marker


@dataclass(slots=True)
class MapData:
    marker_ids: np.ndarray
    marker_points: np.ndarray
    origin: np.ndarray
    rotation: np.ndarray
    pose_ids: np.ndarray
    pose_rotations: np.ndarray
    pose_translations: np.ndarray

    def marker_dict(self) -> Dict[int, np.ndarray]:
        return {int(mid): self.marker_points[idx] for idx, mid in enumerate(self.marker_ids)}


@dataclass(slots=True)
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    inlier_ids: List[int]


@dataclass(slots=True)
class DetectionConfig:
    r: int = 5
    sigma: float = 3.0
    expect_factor: float = 20
    auto_scale: bool = True
    reference_max_side: float = 720.0
    contrast_threshold: float = 10.0
    id_base: int = 1
    column_major_ids: bool = True
    min_coverage_ratio: float = 0.55

    def estimate_candidates(self, sta: np.ndarray) -> int:
        expected_points = (sta.shape[0] + 1) * (sta.shape[1] + 1)
        return int(round(self.expect_factor * expected_points))


def load_camera_intrinsics(matrix_path: Path, distortion_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.load(matrix_path)
    distortion = np.load(distortion_path)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Expected camera matrix shape (3, 3), got {camera_matrix.shape}")
    return camera_matrix.astype(np.float64), distortion.astype(np.float64)


def load_map(npz_path: Path) -> MapData:
    data = np.load(npz_path)
    required = {
        "marker_ids",
        "marker_points",
        "rotation",
        "origin",
        "pose_ids",
        "pose_rotations",
        "pose_translations",
    }
    missing = required.difference(set(data.files))
    if missing:
        raise KeyError(f"Missing keys in map file: {sorted(missing)}")

    return MapData(
        marker_ids=data["marker_ids"].astype(np.int32),
        marker_points=data["marker_points"].astype(np.float64),
        origin=data["origin"].astype(np.float64),
        rotation=data["rotation"].astype(np.float64),
        pose_ids=data["pose_ids"].astype(np.int32),
        pose_rotations=data["pose_rotations"].astype(np.float64),
        pose_translations=data["pose_translations"].astype(np.float64),
    )


def load_marker_pattern(path: Path) -> np.ndarray:
    sta = np.load(path)
    if sta.ndim != 2:
        raise ValueError(f"Marker pattern must be a 2D array, got shape {sta.shape}")
    return sta.astype(np.float64)

def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in radians."""
    pitch = np.arcsin(-R[2, 0])

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return roll, pitch, yaw

@dataclass(slots=True)
class ImageLookup:
    by_index: Dict[int, np.ndarray]
    by_name: Dict[str, np.ndarray]

    def get(self, frame_idx: int, name: Optional[str]) -> Optional[np.ndarray]:
        image = self.by_index.get(int(frame_idx))
        if image is not None:
            return image
        if name is not None:
            return self.by_name.get(str(name))
        return None


def _normalize_npz_entry(entry: Any) -> Dict[str, Any]:
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
    raise TypeError("Unsupported NPZ frame entry type; expected dict-like objects.")


def _iter_npz_entries(npz_path: Path, data_key: Optional[str]) -> Iterator[Dict[str, Any]]:
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
                    f"Multiple arrays in {npz_path!s}: {keys}. Provide data_key to choose one."
                )
        if target_key not in npz:
            raise KeyError(f"Array {target_key!r} not found in {npz_path!s}. Available keys: {keys}")

        dataset = npz[target_key]

    if isinstance(dataset, np.ndarray) and dataset.dtype == object:
        for entry in dataset:
            yield _normalize_npz_entry(entry)
    elif isinstance(dataset, np.ndarray) and dataset.ndim == 3:
        for frame in dataset:
            yield {"img": frame}
    else:
        raise TypeError(
            "Unsupported NPZ dataset shape or dtype. Expected an object array of dict-like entries "
            "or a 3D array of images."
        )


def load_detection_records(npz_path: Path) -> List[Dict[str, Any]]:
    with np.load(npz_path, allow_pickle=True) as npz:
        if "frames" not in npz:
            raise KeyError(
                f"Detections NPZ at {npz_path!s} does not contain a 'frames' array produced by ReadMarker_FromNPZ."
            )
        frames = npz["frames"]

    records: List[Dict[str, Any]] = []
    for entry in frames:
        if isinstance(entry, dict):
            record = dict(entry)
        elif hasattr(entry, "item"):
            record = entry.item()
            if not isinstance(record, dict):
                raise TypeError("Unexpected detections entry; expected dict-like objects.")
        else:
            raise TypeError("Unexpected detections entry type; expected dict-like objects.")

        detections = np.asarray(record.get("detections", np.empty((0, 3))), dtype=np.float64)
        if detections.ndim != 2 or detections.shape[1] != 3:
            raise ValueError("Each detection entry must be an array of shape (N, 3) [id, x, y].")
        record["detections"] = detections
        records.append(record)
    return records


def load_npz_image_lookup(
    npz_path: Path,
    data_key: Optional[str] = None,
    image_key: str = "img",
    max_frames: Optional[int] = None,
) -> ImageLookup:
    by_index: Dict[int, np.ndarray] = {}
    by_name: Dict[str, np.ndarray] = {}

    for idx, entry in enumerate(_iter_npz_entries(npz_path, data_key)):
        if image_key not in entry:
            raise KeyError(f"Entry {idx} in {npz_path!s} does not contain key '{image_key}'.")
        image = np.asarray(entry[image_key])
        frame_idx = int(entry.get("index", idx))
        name = entry.get("name") or entry.get("frame") or f"frame_{idx:06d}"
        by_index[frame_idx] = image
        by_name[str(name)] = image
        if max_frames is not None and len(by_index) >= max_frames:
            break

    if not by_index:
        raise ValueError(f"No images loaded from {npz_path!s} using key '{image_key}'.")

    return ImageLookup(by_index=by_index, by_name=by_name)


class MapTracker:
    def __init__(self, map_data: MapData, camera_matrix: np.ndarray, distortion: np.ndarray) -> None:
        self.map = map_data
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.marker_lookup = map_data.marker_dict()
        self.last_rvec: Optional[np.ndarray] = None
        self.last_tvec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last_rvec = None
        self.last_tvec = None

    def estimate_pose(
        self,
        detections: Dict[int, np.ndarray],
        unidentified_corners: Optional[np.ndarray] = None,
        use_last_as_initial: bool = True,
        flags: int = cv2.SOLVEPNP_ITERATIVE,
        use_geometric_refinement: bool = False,
        match_threshold_px: float = 10.0,
    ) -> Optional[PoseEstimate]:
        """
        Estimate camera pose from marker detections.
        
        Args:
            detections: Dictionary mapping marker_id -> pixel coordinates [x, y]
            unidentified_corners: Optional Nx2 array of detected corners without IDs
            use_last_as_initial: Whether to use previous pose as initial guess
            flags: OpenCV solvePnP flags
            use_geometric_refinement: Enable Stage 2 refinement with unidentified corners
            match_threshold_px: Distance threshold for geometric matching (pixels)
        """
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        object_ids: list[int] = []
        for marker_id, pixel in detections.items():
            point = self.marker_lookup.get(marker_id)
            if point is None:
                continue
            object_points.append(point)
            image_points.append(pixel)
            object_ids.append(int(marker_id))

        if len(object_points) < 6:
            return None

        object_points_np = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points_np = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        rvec_init = self.last_rvec if use_last_as_initial and self.last_rvec is not None else None
        tvec_init = self.last_tvec if use_last_as_initial and self.last_tvec is not None else None
        use_guess = use_last_as_initial and self.last_rvec is not None

        # Stage 1: Initial pose estimation with RANSAC
        success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
            object_points_np,
            image_points_np,
            self.camera_matrix,
            self.distortion,
            rvec=rvec_init,           # Pass previous pose
            tvec=tvec_init,           # as initial guess
            useExtrinsicGuess=use_guess,
            reprojectionError=0.5,
            confidence=0.9995,
            iterationsCount=10000,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success or rvec_est is None or tvec_est is None:
            return None
        
        # Pose validation: check for unreasonable values
        distance = np.linalg.norm(tvec_est)
        if distance > 450.0:  # Adjust threshold based on your scene scale
            print(f"Rejected: distance {distance:.1f} exceeds limit")
            return None
        
        if rvec_init is not None:
            rotation_magnitude_init = np.linalg.norm(rvec_init)
            rotation_magnitude = np.linalg.norm(rvec_est)
            if rotation_magnitude_init - rotation_magnitude > 2 * np.pi:
                print(f"Rejected: rotation magnitude {rotation_magnitude:.1f} exceeds limit")
                return None
        
        # Temporal consistency check
        if self.last_tvec is not None:
            motion = np.linalg.norm(tvec_est - self.last_tvec)
            if motion > 70:  # Adjust based on expected object motion
                print(f"Rejected: motion {motion:.3f} too large between frames")
                return None
        
        # IF YOU WANT TO USE IT TURN ON (use_geometric_refinement = True)
        #  Stage 2: Geometric refinement with unidentified corners
        additional_inliers_added = 0
    
        if (use_geometric_refinement and unidentified_corners is not None and len(unidentified_corners) > 0):
            # Get all 3D points from the map
            all_map_3d = np.array([point for marker_points in self.map.marker_points for point in [marker_points]], dtype=np.float64).reshape(-1, 3)
            
            # Project all map points using pose that we previously calculated with PnP
            projected_2d, _ = cv2.projectPoints(
                all_map_3d,
                rvec_est,
                tvec_est,
                self.camera_matrix,
                self.distortion
            )
            projected_2d = projected_2d.reshape(-1, 2)
            
            # Match unidentified corners to projected map points
            additional_2d = []
            additional_3d = []
            
            for unid_corner in unidentified_corners:
                # Find nearest projected point
                distances = np.linalg.norm(projected_2d - unid_corner, axis=1)
                min_dist_idx = np.argmin(distances)
                
                if distances[min_dist_idx] < match_threshold_px:
                    additional_2d.append(unid_corner)
                    additional_3d.append(all_map_3d[min_dist_idx])
            
            # Re-solve PnP with augmented inlier set
            if len(additional_2d) > 0:
                combined_3d = np.vstack([object_points_np, additional_3d])
                combined_2d = np.vstack([image_points_np, additional_2d])
                
                # Refine pose with more inliers
                success_refined, rvec_refined, tvec_refined = cv2.solvePnP(
                    combined_3d,
                    combined_2d,
                    self.camera_matrix,
                    self.distortion,
                    rvec=rvec_est,
                    tvec=tvec_est,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success_refined:
                    reproj_pts, _ = cv2.projectPoints(combined_3d, rvec_refined, tvec_refined,self.camera_matrix, self.distortion)
                    reproj_errors = np.linalg.norm(reproj_pts.reshape(-1, 2) - combined_2d, axis=1)
                    
                    # Split errors: original vs newly added
                    original_errors = reproj_errors[:len(object_points_np)]
                    new_errors = reproj_errors[len(object_points_np):]
                    
                    # Thresholds
                    max_original_error = 0.5  # pixels (original corners should stay tight) a little above ransac
                    max_new_error = 1.0       # pixels (new inliers must also be tight)
                    
                    # Check if refinement made things worse
                    if (np.mean(original_errors) < max_original_error and np.all(new_errors < max_new_error)):
                        # Good refinement!
                        rvec_est = rvec_refined
                        tvec_est = tvec_refined
                        additional_inliers_added = len(additional_2d)
                        print(f"  ✓ Refinement accepted: "
                            f"orig_err={np.mean(original_errors):.3f}px, "
                            f"new_err={np.mean(new_errors):.3f}px")
                    else:
                        # Refinement is bad → reject it
                        additional_inliers_added = 0
                        print(f"  ✗ Refinement rejected: "
                            f"orig_err={np.mean(original_errors):.3f}px, "
                            f"new_err={np.mean(new_errors):.3f}px "
                            f"(max_orig={max_original_error}, max_new={max_new_error})")
            else:
                print(f"  No additional matches found, keeping Stage 1 pose")
        
        self.last_rvec = rvec_est
        self.last_tvec = tvec_est

        inlier_ids: List[int]
        if inliers is None:
            inlier_ids = object_ids.copy()
        else:
            inlier_indices = inliers.reshape(-1)
            inlier_ids = [object_ids[idx] for idx in inlier_indices if 0 <= idx < len(object_ids)]

        return PoseEstimate(rvec=rvec_est, tvec=tvec_est, inlier_ids=inlier_ids)

    def compute_reprojection_errors(
        self,
        detections: Dict[int, np.ndarray],
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[np.ndarray]:
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for marker_id, pixel in detections.items():
            map_point = self.marker_lookup.get(marker_id)
            if map_point is None:
                continue
            object_points.append(map_point)
            image_points.append(pixel)

        if not object_points:
            return None

        object_points_np = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points_np = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        projected, _ = cv2.projectPoints(object_points_np, rvec, tvec, self.camera_matrix, self.distortion)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points_np, axis=1)
        return errors


def detect_markers(
    image_gray: np.ndarray,
    sta: np.ndarray,
    config: DetectionConfig,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    expect_n = config.estimate_candidates(sta)
    pt_list, _ = read_marker(
        image_gray,
        sta,
        r=config.r,
        expect_n=expect_n,
        sigma=config.sigma,
        auto_scale=config.auto_scale,
        reference_max_side=config.reference_max_side,
        contrast_threshold=config.contrast_threshold,
        id_base=config.id_base,
        column_major_ids=config.column_major_ids,
        min_coverage_ratio=config.min_coverage_ratio,
        return_debug=False,
    )

    detections: Dict[int, np.ndarray] = {}
    for row, col, marker_id in pt_list:
        if np.isnan(marker_id):
            continue
        detections[int(marker_id)] = np.array([float(col), float(row)], dtype=np.float64)
    return detections, pt_list


def draw_detections(frame: np.ndarray, pt_list: np.ndarray) -> np.ndarray:
    if frame.ndim == 2 or frame.shape[2] == 1:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()
    marker_radius = max(3, int(round(min(frame.shape[0], frame.shape[1]) / 200)))
    font_scale = 0.5
    font_thickness = 1
    for row, col, marker_id in pt_list:
        center = (int(round(col)), int(round(row)))
        if np.isnan(marker_id):
            cv2.drawMarker(vis, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            continue
        cv2.circle(vis, center, marker_radius, (0, 255, 0), thickness=-1)
        cv2.putText(
            vis,
            str(int(marker_id)),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
            cv2.LINE_AA,
        )
    return vis


def open_capture(video_path: Optional[Path], camera_index: int) -> cv2.VideoCapture:
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        return cap
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return cap


def run_tracking_loop(
    tracker: MapTracker,
    sta: np.ndarray,
    config: DetectionConfig,
    video_path: Optional[Path],
    camera_index: Optional[int],
    display: bool,
    axis_length: float,
    min_markers: int,
    report_every: int,
    quit_key: str,
    use_pose_prior: bool,
    plot_output_dir: Optional[Path] = None,
) -> None:
    cap = open_capture(video_path, camera_index if camera_index is not None else 0)
    window_name = "Hydra Marker Tracking"
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                detections, pt_list = detect_markers(gray, sta, config)
            except Exception as exc:  # pragma: no cover - defensive path
                print(f"[frame {frame_idx}] detection failed: {exc}")
                tracker.reset()
                frame_idx += 1
                continue

            pose: Optional[PoseEstimate] = None
            if len(detections) >= min_markers:
                pose = tracker.estimate_pose(detections, use_last_as_initial=use_pose_prior)
            else:
                tracker.reset()

            if pose is not None:
                rvec = pose.rvec
                tvec = pose.tvec

                inlier_detections = {
                    marker_id: detections[marker_id] for marker_id in pose.inlier_ids if marker_id in detections
                }
                if frame_idx % report_every == 0:
                    errors = tracker.compute_reprojection_errors(inlier_detections, rvec, tvec)
                    err_msg = (
                        f"mean={errors.mean():.3f}px, max={errors.max():.3f}px" if errors is not None else "n/a"
                    )
                    print(
                        f"[frame {frame_idx}] markers={len(pose.inlier_ids)}/{len(detections)} "
                        f"t=({tvec.flatten()[0]:.3f}, {tvec.flatten()[1]:.3f}, {tvec.flatten()[2]:.3f}) "
                        f"rvec=({rvec.flatten()[0]:.3f}, {rvec.flatten()[1]:.3f}, {rvec.flatten()[2]:.3f}) "
                        f"reproj {err_msg}"
                    )
            elif frame_idx % report_every == 0:
                print(f"[frame {frame_idx}] insufficient markers ({len(detections)}) for pose.")

            if display:
                vis = draw_detections(frame, pt_list)
                if pose is not None:
                    cv2.drawFrameAxes(
                        vis, tracker.camera_matrix, tracker.distortion, pose.rvec, pose.tvec, axis_length
                    )
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1 if video_path is None else 10) & 0xFF
                if key == ord(quit_key):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if display:
            cv2.destroyWindow(window_name)


def run_tracking_from_records(
    tracker: MapTracker,
    records: List[Dict[str, Any]],
    image_lookup: Optional[ImageLookup],
    display: bool,
    axis_length: float,
    min_markers: int,
    report_every: int,
    quit_key: str,
    use_pose_prior: bool,
    display_delay_ms: int = 10,
    plot_output_dir: Optional[Path] = None,
    results_output_path: Optional[Path] = None,
) -> None:
    if not records:
        print("No detection records provided; nothing to track.")
        return

    if display and image_lookup is None:
        print("Display requested but no frame images provided; visualization disabled.")
        display = False

    if results_output_path is not None:
        results_output_path = Path(results_output_path)
    elif plot_output_dir is not None:
        results_output_path = Path(plot_output_dir) / "tracking_results.npy"

    window_name = "Hydra Marker Tracking (NPZ)"
    delay_ms = max(1, int(display_delay_ms))

    frame_indices: List[int] = []
    timestamps: List[float] = []
    translations_x: List[float] = []
    translations_y: List[float] = []
    translations_z: List[float] = []
    rotation_mats: List[np.ndarray] = []
    euler_x: List[float] = []
    euler_y: List[float] = []
    euler_z: List[float] = []
    euler_x_deg: List[float] = []
    euler_y_deg: List[float] = []
    euler_z_deg: List[float] = []
    rotation_magnitude_deg: List[float] = []
    reprojection_error: List[float] = []


    for seq_idx, record in enumerate(records):
        frame_idx = int(record.get("index", seq_idx))
        name = record.get("name") or f"frame_{frame_idx:06d}"
        timestamp = record.get("timestamp", float("nan"))
        detections_arr = np.asarray(record.get("detections", np.empty((0, 3))), dtype=np.float64)
        detections_map: Dict[int, np.ndarray] = {int(marker_id): np.array([float(x), float(y)], dtype=np.float64) for marker_id, x, y in detections_arr}
        unidentified_corners = np.asarray(record.get("unidentified_corners", np.empty((0, 2), dtype=np.float64)), dtype=np.float64)
        pose: Optional[PoseEstimate] = None
        error_value = float("nan")
        if len(detections_map) >= min_markers:
            pose = tracker.estimate_pose(detections_map, unidentified_corners=unidentified_corners, use_last_as_initial=use_pose_prior, use_geometric_refinement=False, match_threshold_px=1.0)
        else:
            tracker.reset()

        frame_indices.append(frame_idx)
        timestamps.append(timestamp)
        timestamp_str = f"{timestamp:.3f}" if np.isfinite(timestamp) else "n/a"

        if pose is not None:
            rvec = pose.rvec
            tvec = pose.tvec
            inlier_detections = {
                marker_id: detections_map[marker_id] for marker_id in pose.inlier_ids if marker_id in detections_map
            }
            errors = tracker.compute_reprojection_errors(inlier_detections, rvec, tvec)
            if errors is not None and errors.size:
                errors = errors.astype(np.float64)
                error_value = float(np.sqrt(np.mean(errors * errors)))
            if seq_idx % report_every == 0:
                err_msg = f"rms={error_value:.3f}px" if np.isfinite(error_value) else "n/a"
                print(
                    f"[{seq_idx}] frame={name} markers={len(pose.inlier_ids)}/{len(detections_map)} "
                    f"t=({tvec.flatten()[0]:.3f}, {tvec.flatten()[1]:.3f}, {tvec.flatten()[2]:.3f}) "
                    f"rvec=({rvec.flatten()[0]:.3f}, {rvec.flatten()[1]:.3f}, {rvec.flatten()[2]:.3f}) "
                    f"ts={timestamp_str} "
                    f"reproj {err_msg}"
                )
            flat_t = tvec.reshape(-1)
            flat_r = rvec.reshape(-1)
            translations_x.append(float(flat_t[0]))
            translations_y.append(float(flat_t[1]))
            translations_z.append(float(flat_t[2]))
            rotation_matrix, _ = cv2.Rodrigues(flat_r)
            rotation_mats.append(rotation_matrix.astype(float))
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
            euler_x.append(float(roll))
            euler_y.append(float(pitch))
            euler_z.append(float(yaw))
            roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
            euler_x_deg.append(float(roll_deg))
            euler_y_deg.append(float(pitch_deg))
            euler_z_deg.append(float(yaw_deg))
            rot_mag_deg = float(np.degrees(np.linalg.norm(flat_r)))
            rotation_magnitude_deg.append(rot_mag_deg)
        else:
            translations_x.append(float("nan"))
            translations_y.append(float("nan"))
            translations_z.append(float("nan"))
            rotation_mats.append(np.full((3, 3), np.nan, dtype=float))
            euler_x.append(float("nan"))
            euler_y.append(float("nan"))
            euler_z.append(float("nan"))
            euler_x_deg.append(float("nan"))
            euler_y_deg.append(float("nan"))
            euler_z_deg.append(float("nan"))
            rotation_magnitude_deg.append(float("nan"))
            if seq_idx % report_every == 0:
                print(
                    f"[{seq_idx}] frame={name} ts={timestamp_str} "
                    f"insufficient markers ({len(detections_map)}) for pose."
                )

        reprojection_error.append(error_value)

        if display and image_lookup is not None:
            image = image_lookup.get(frame_idx, record.get("name"))
            if image is not None:
                pt_visual = (
                    np.column_stack((detections_arr[:, 2], detections_arr[:, 1], detections_arr[:, 0]))
                    if detections_arr.size
                    else np.empty((0, 3), dtype=np.float64)
                )
                vis = draw_detections(image, pt_visual)
                if pose is not None:
                    cv2.drawFrameAxes(
                        vis, tracker.camera_matrix, tracker.distortion, pose.rvec, pose.tvec, axis_length
                    )
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord(quit_key):
                    break
            else:
                print(f"[{seq_idx}] frame={name} has no associated image for display.")

    if display:
        cv2.destroyWindow(window_name)

    frame_numbers_arr = np.asarray(frame_indices, dtype=np.int32)
    timestamps_arr_save = np.asarray(timestamps, dtype=np.float64)
    translations_arr = (np.column_stack((np.asarray(translations_x, dtype=np.float64), np.asarray(translations_y, dtype=np.float64), np.asarray(translations_z, dtype=np.float64))) if frame_numbers_arr.size else np.empty((0, 3), dtype=np.float64))
    rotation_matrix_arr = (np.stack(rotation_mats, axis=0) if rotation_mats else np.empty((0, 3, 3), dtype=float))
    euler_rad_arr = (np.column_stack((np.asarray(euler_x, dtype=np.float64), np.asarray(euler_y, dtype=np.float64), np.asarray(euler_z, dtype=np.float64))) if frame_numbers_arr.size else np.empty((0, 3), dtype=np.float64))
    euler_deg_arr = (np.column_stack((np.asarray(euler_x_deg, dtype=np.float64), np.asarray(euler_y_deg, dtype=np.float64), np.asarray(euler_z_deg, dtype=np.float64))) if frame_numbers_arr.size else np.empty((0, 3), dtype=np.float64))
    rotation_magnitude_arr = np.asarray(rotation_magnitude_deg, dtype=np.float64)
    reprojection_error_arr = np.asarray(reprojection_error, dtype=np.float64)

    results_payload = {
        "frame_numbers": frame_numbers_arr,
        "timestamps": timestamps_arr_save,
        "translations_mm": translations_arr,
        "rotation_matrices": rotation_matrix_arr,
        "euler_rad": euler_rad_arr,
        "euler_deg": euler_deg_arr,
        "rotation_magnitude_deg": rotation_magnitude_arr,
        "reprojection_error_px": reprojection_error_arr,
    }

    if results_output_path is not None:
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(results_output_path, results_payload, allow_pickle=True)
        print(f"Saved tracking results to {results_output_path}")

    if frame_indices:
        frame_indices_arr = np.asarray(frame_indices, dtype=float)
        timestamps_arr = np.asarray(timestamps, dtype=float)
        finite_time_mask = np.isfinite(timestamps_arr)
        if finite_time_mask.any():
            x_axis = timestamps_arr.copy()
            first_ts = float(np.nanmin(timestamps_arr[finite_time_mask]))
            if np.isfinite(first_ts):
                x_axis -= first_ts
            x_label = "Time (s)"
        else:
            x_axis = frame_indices_arr
            x_label = "Frame Index"

        fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        translation_arrays = [np.asarray(translations_x, dtype=float),
                              np.asarray(translations_y, dtype=float),
                              np.asarray(translations_z, dtype=float)]
        rotation_arrays = [np.asarray(euler_x_deg, dtype=float),
                           np.asarray(euler_y_deg, dtype=float),
                           np.asarray(euler_z_deg, dtype=float)]
        translation_min = min(np.nanmin(arr) for arr in translation_arrays if np.isfinite(arr).any())
        translation_max = max(np.nanmax(arr) for arr in translation_arrays if np.isfinite(arr).any())
        rotation_min = min(np.nanmin(arr) for arr in rotation_arrays if np.isfinite(arr).any())
        rotation_max = max(np.nanmax(arr) for arr in rotation_arrays if np.isfinite(arr).any())
        components = [
            (translations_x, "Translation X (mm)"),
            (translations_y, "Translation Y (mm)"),
            (translations_z, "Translation Z (mm)"),
            (euler_x_deg, "Rotation X (deg)"),
            (euler_y_deg, "Rotation Y (deg)"),
            (euler_z_deg, "Rotation Z (deg)"),
        ]
        for ax, (series, label) in zip(axes.flatten(), components):
            ax.plot(x_axis, series, linewidth=1.2)
            ax.set_ylabel(label)
            if "Translation" in label:
                ax.set_ylim(translation_min-np.abs(0.1*translation_min), translation_max+np.abs(0.1*translation_max))
            elif "Rotation" in label:
                ax.set_ylim(rotation_min-np.abs(0.1*rotation_min), rotation_max+np.abs(0.2*rotation_max))
            ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)
        axes[1, 2].set_xlabel(x_label)
        fig.suptitle("Marker Translation and Rotation Components Over Time")
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)

        fig_err, ax_err = plt.subplots(figsize=(10, 4))
        ax_err.plot(x_axis, reprojection_error, linewidth=1.2)
        ax_err.set_xlabel(x_label)
        ax_err.set_ylabel("RMS Reprojection Error (pixels)")
        ax_err.set_title("Reprojection Error Over Time")
        ax_err.grid(alpha=0.3, linestyle="--", linewidth=0.8)

        fig_rot, ax_rot = plt.subplots(figsize=(10, 4))
        ax_rot.plot(x_axis, rotation_magnitude_deg, linewidth=1.2)
        ax_rot.set_xlabel(x_label)
        ax_rot.set_ylabel("Rotation Magnitude (deg)")
        ax_rot.set_title("Angle Magnitude Over Time")
        ax_rot.grid(alpha=0.3, linestyle="--", linewidth=0.8)

        if plot_output_dir is not None:
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            pose_plot_path = plot_output_dir / "motion_components.png"
            error_plot_path = plot_output_dir / "reprojection_error.png"
            rotation_plot_path = plot_output_dir / "rotation_magnitude.png"
            fig.savefig(pose_plot_path, dpi=150)
            fig_err.savefig(error_plot_path, dpi=150)
            fig_rot.savefig(rotation_plot_path, dpi=150)
            print(f"Saved motion plot to {pose_plot_path}")
            print(f"Saved reprojection plot to {error_plot_path}")
            print(f"Saved rotation-magnitude plot to {rotation_plot_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track object pose using a pre-built Hydra marker map.")
    parser.add_argument("map_path", type=Path, help="Path to marker_map_aligned.npz produced by reconstruction.")
    parser.add_argument("--camera-matrix", type=Path, required=True, help="Path to camera matrix .npy.")
    parser.add_argument("--distortion", type=Path, required=True, help="Path to distortion coefficients .npy.")
    parser.add_argument("--sta", type=Path, help="Path to marker_sta.npy pattern (required for live detection).")
    parser.add_argument(
        "--detections-npz",
        type=Path,
        help="Path to detections_data.npz produced by ReadMarker_FromNPZ (pre-computed marker detections).",
    )
    parser.add_argument(
        "--frames-npz",
        type=Path,
        help="Optional NPZ containing original frames for visualization alongside detection playback.",
    )
    parser.add_argument(
        "--frames-data-key",
        type=str,
        default=None,
        help="Dataset key inside frames NPZ (autodetected when omitted).",
    )
    parser.add_argument(
        "--frames-image-key",
        type=str,
        default="img",
        help="Key holding the image array inside each NPZ frame entry.",
    )
    parser.add_argument(
        "--frames-max",
        type=int,
        default=None,
        help="Optional limit on the number of frames to load from the frames NPZ.",
    )
    parser.add_argument("--video", type=Path, help="Optional video file for tracking.")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index for live tracking (if no video).")
    parser.add_argument("--no-display", action="store_true", help="Disable visualization window.")
    parser.add_argument("--axis-length", type=float, default=50.0, help="Axis length for pose visualization.")
    parser.add_argument("--min-markers", type=int, default=6, help="Minimum markers required to compute pose.")
    parser.add_argument("--report-every", type=int, default=10, help="Console logging interval in frames.")
    parser.add_argument("--quit-key", type=str, default="q", help="Key to stop the visualization window.")
    parser.add_argument(
        "--display-delay-ms",
        type=int,
        default=10,
        help="Delay in milliseconds between frames when visualizing recorded detections.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        help="Path to .npy file for saving tracking results when processing detection records.",
    )
    parser.add_argument("--no-pose-prior", action="store_true", help="Disable warm-starting solvePnP with last pose.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.map_path.exists():
        raise FileNotFoundError(f"Map file not found: {args.map_path}")

    map_data = load_map(args.map_path)
    print(
        f"Loaded map with {map_data.marker_ids.size} markers "
        f"and {map_data.pose_ids.size} stored camera poses from {args.map_path}"
    )

    camera_matrix, distortion = load_camera_intrinsics(args.camera_matrix, args.distortion)
    #print(f"Camera matrix:\n{camera_matrix}")
    #print(f"Distortion coefficients shape: {distortion.shape}")

    tracker = MapTracker(map_data, camera_matrix, distortion)
    print("Tracker initialized.")

    if args.detections_npz is not None:
        if not args.detections_npz.exists():
            raise FileNotFoundError(f"Detections NPZ not found: {args.detections_npz}")
        records = load_detection_records(args.detections_npz)
        image_lookup = None
        if args.frames_npz is not None:
            if not args.frames_npz.exists():
                raise FileNotFoundError(f"Frames NPZ not found: {args.frames_npz}")
            image_lookup = load_npz_image_lookup(
                args.frames_npz,
                data_key=args.frames_data_key,
                image_key=args.frames_image_key,
                max_frames=args.frames_max,
            )
        run_tracking_from_records(
            tracker,
            records,
            image_lookup=image_lookup,
            display=not args.no_display,
            axis_length=args.axis_length,
            min_markers=args.min_markers,
            report_every=max(1, args.report_every),
            quit_key=args.quit_key,
            use_pose_prior=not args.no_pose_prior,
            display_delay_ms=args.display_delay_ms,
            results_output_path=args.results_output,
        )
        return

    if args.sta is None:
        raise ValueError("Live/video tracking requires --sta pointing to marker_sta.npy.")

    sta = load_marker_pattern(args.sta)
    detection_config = DetectionConfig()

    if args.video is None and args.camera < 0:
        print("No video or camera source specified; exiting after initialization.")
        return

    run_tracking_loop(
        tracker,
        sta,
        detection_config,
        video_path=args.video,
        camera_index=None if args.video is not None else args.camera,
        display=not args.no_display,
        axis_length=args.axis_length,
        min_markers=args.min_markers,
        report_every=max(1, args.report_every),
        quit_key=args.quit_key,
        use_pose_prior=not args.no_pose_prior,
    )


if __name__ == "__main__":
    main()
