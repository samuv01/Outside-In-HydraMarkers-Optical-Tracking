from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pyceres
import warnings


@dataclass(slots=True)
class CameraCalibration:
    """Holds camera intrinsics and lens distortion coefficients."""

    camera_matrix: np.ndarray  # 3x3
    distortion_coeffs: np.ndarray  # Nx1 or (N,)

    def validate(self) -> None:
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got {self.camera_matrix.shape}")
        if self.distortion_coeffs.ndim not in (1, 2):
            raise ValueError("distortion_coeffs must be a 1D or 2D array")


@dataclass(slots=True)
class MarkerDetection:
    """Single marker observation inside a frame."""

    marker_id: int
    image_point: np.ndarray  # (2,) or (2,1), (x, y) in pixels
    confidence: float = 1.0

    def as_tuple(self) -> Tuple[int, float, float, float]:
        x, y = map(float, np.asarray(self.image_point).ravel())
        return self.marker_id, x, y, float(self.confidence)


@dataclass(slots=True)
class FrameObservation:
    """All marker detections for one frame."""

    frame_id: int
    detections: Dict[int, MarkerDetection] = field(default_factory=dict)
    timestamp: Optional[float] = None
    metadata: Optional[Mapping[str, object]] = None

    def shared_ids(self, other: "FrameObservation") -> List[int]:
        """Return marker IDs seen in both frames."""
        return sorted(set(self.detections) & set(other.detections))

    def to_array(self) -> np.ndarray:
        """Return detections as structured array-like (id, x, y, confidence)."""
        rows = [det.as_tuple() for det in self.detections.values()]
        return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)

    def get_detection(self, marker_id: int) -> MarkerDetection:
        try:
            return self.detections[marker_id]
        except KeyError as ex:
            raise KeyError(f"Marker {marker_id} not found in frame {self.frame_id}") from ex


@dataclass(slots=True)
class CameraPose:
    """Rigid pose represented by rotation matrix and translation vector."""

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def as_homogeneous(self) -> np.ndarray:
        """Return 4x4 homogeneous transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass(slots=True)
class TriangulationResult:
    """Result of triangulating correspondences between two views."""

    marker_ids: np.ndarray
    points: np.ndarray
    depths_view1: np.ndarray
    depths_view2: np.ndarray
    reprojection_errors: np.ndarray


@dataclass(slots=True)
class SfMState:
    """Container for the state of the marker-based reconstruction."""

    calibration: CameraCalibration
    frames: List[FrameObservation]
    poses: MutableMapping[int, CameraPose] = field(default_factory=dict)
    marker_positions: MutableMapping[int, np.ndarray] = field(default_factory=dict)

    def frame_by_id(self, frame_id: int) -> FrameObservation:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(frame_id)

    def add_pose(self, frame_id: int, pose: CameraPose) -> None:
        self.poses[frame_id] = pose

    def add_marker_position(self, marker_id: int, position: np.ndarray) -> None:
        self.marker_positions[marker_id] = np.asarray(position, dtype=np.float64)

    def has_pose(self, frame_id: int) -> bool:
        return frame_id in self.poses

    def posed_frames(self) -> List[int]:
        return sorted(self.poses.keys())

    def get_marker_position(self, marker_id: int) -> Optional[np.ndarray]:
        return self.marker_positions.get(marker_id)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "SfMState":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, SfMState):
            raise TypeError(f"Unexpected object type in checkpoint: {type(obj)!r}")
        return obj


def load_calibration(camera_matrix_path = Path, distortion_path = Path) -> CameraCalibration:
    """
    Load camera calibration data from .npy files.

    Parameters
    ----------
    camera_matrix_path:
        Path to a 3x3 intrinsic matrix stored via numpy.save.
    distortion_path:
        Path to distortion coefficients stored via numpy.save.
    """
    camera_matrix = np.load(camera_matrix_path)
    distortion_coeffs = np.load(distortion_path)
    calibration = CameraCalibration(camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
    calibration.validate()
    return calibration


def load_frame_observations(npz_path : Path, image_key: str = "img") -> List[FrameObservation]:
    """
    Load marker frame observations from an NPZ archive produced by ReadMarker_FromNPZ.

    The NPZ is expected to contain an object array where each element is a dictionary
    with keys including:
        - "index" or "frame"
        - "detections": array-like of (marker_id, x, y) or (marker_id, row, col)
        - optional "timestamp"

    This loader is intentionally flexible; mismatches will raise informative errors.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Detections NPZ not found: {npz_path!s}")

    with np.load(npz_path, allow_pickle=True) as npz:
        if "frames" in npz.files:
            payload = npz["frames"]
        elif "detections" in npz.files:
            payload = npz["detections"]
        else:
            # Fallback: assume single array with structured entries
            payload = next(iter(npz.values()))

    frames: List[FrameObservation] = []
    for idx, entry in enumerate(np.asarray(payload).ravel()):
        if isinstance(entry, dict):
            frame_id = int(entry.get("index", entry.get("frame", idx)))
            detections_raw = entry.get("detections")
            timestamp = entry.get("timestamp")
        elif hasattr(entry, "item"):
            item = entry.item()
            if isinstance(item, dict):
                frame_id = int(item.get("index", item.get("frame", idx)))
                detections_raw = item.get("detections")
                timestamp = item.get("timestamp")
            else:
                raise TypeError(f"Unsupported entry type at index {idx}: {type(item)!r}")
        else:
            raise TypeError(f"Unsupported entry type at index {idx}: {type(entry)!r}")

        if detections_raw is None:
            raise ValueError(f"Frame {frame_id} is missing 'detections' data")

        detections_array = np.asarray(detections_raw)
        if detections_array.ndim != 2 or detections_array.shape[1] < 3:
            raise ValueError(
                f"Detections for frame {frame_id} must be Nx3 or Nx4 array (id,x,y[,conf]), "
                f"got shape {detections_array.shape}"
            )

        frame = FrameObservation(frame_id=frame_id, timestamp=float(timestamp) if timestamp is not None else None)
        for row in detections_array:
            marker_id = int(row[0])
            x, y = float(row[1]), float(row[2])
            confidence = float(row[3]) if row.shape[0] >= 4 else 1.0
            frame.detections[marker_id] = MarkerDetection(marker_id=marker_id, image_point=np.array([x, y]), confidence=confidence)
        frames.append(frame)

    frames.sort(key=lambda f: f.frame_id)
    return frames


def initialize_state(
    calibration: CameraCalibration,
    frames: Iterable[FrameObservation],
) -> SfMState:
    """
    Create an empty SfMState ready to be populated by the reconstruction pipeline.

    This function does not estimate poses or 3D structure; it simply packages inputs.
    """
    frame_list = list(frames)
    return SfMState(calibration=calibration, frames=frame_list)


@dataclass(slots=True)
class FilterConfig:
    """Parameters controlling detection filtering and robust bootstrap selection."""

    # Frame consistency parameters
    min_consensus: int = 100
    outlier_sigma: float = 3.0
    max_pixel_deviation: float = 700.0
    velocity_spike_multiplier: float = 80.0
    per_frame_median_threshold: float = 600.0
    per_frame_mean_threshold: float = 600.0
    per_frame_large_threshold: float = 800.0
    per_frame_large_ratio: float = 1.0
    frame_pair_distance_threshold: float = 80.0
    min_pair_consensus: int = 1

    # Epipolar voting thresholds
    epipolar_inlier_threshold: float = 0.8
    epipolar_ransac_threshold: float = 1e-3
    sample_strategy: str = "strided"

    # Bootstrap search settings
    min_shared_markers: int = 15
    min_bootstrap_inlier_ratio: float = 0.85
    max_pairs_to_test: int = 50

    # Validation thresholds
    max_median_error: float = 2.0
    max_single_error: float = 5.0
    min_bootstrap_points: int = 8


def _pose_to_rvec_tvec(pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(pose.rotation)
    tvec = pose.translation.reshape(3, 1)
    return rvec.astype(np.float64), tvec.astype(np.float64)


def _undistort_points(points: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """Undistort pixel coordinates into normalized camera coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 2)
    pts = pts.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, calibration.camera_matrix, calibration.distortion_coeffs)
    return undistorted.reshape(-1, 2)


def select_bootstrap_pair(state: SfMState) -> Tuple[FrameObservation, FrameObservation]:
    """Select a pair of frames to bootstrap the reconstruction."""
    if len(state.frames) < 2:
        raise ValueError("At least two frames are required for bootstrap.")

    base_frame = state.frames[0]
    best_frame: Optional[FrameObservation] = None
    best_shared = 0

    for candidate in state.frames[1:]:
        shared = len(base_frame.shared_ids(candidate))
        if shared > best_shared:
            best_shared = shared
            best_frame = candidate

    if best_frame is None or best_shared < 8:
        raise RuntimeError(
            "Failed to find a bootstrap pair with sufficient shared markers (need >= 8)."
        )

    return base_frame, best_frame


def estimate_relative_pose(
    frame_a: FrameObservation,
    frame_b: FrameObservation,
    calibration: CameraCalibration,
    *,
    ransac_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the relative pose between two frames using their shared marker detections.

    Returns
    -------
    R : np.ndarray
        3x3 rotation matrix taking points from frame A to frame B.
    t : np.ndarray
        3-vector translation (up to scale) from frame A to frame B.
    marker_ids : np.ndarray
        Marker IDs used for the final pose estimate.
    pts_a_norm : np.ndarray
        Nx2 normalized (undistorted) coordinates from frame A.
    pts_b_norm : np.ndarray
        Nx2 normalized (undistorted) coordinates from frame B.
    """
    shared_ids = frame_a.shared_ids(frame_b)
    if len(shared_ids) < 8:
        raise RuntimeError(
            f"Insufficient shared markers between frames {frame_a.frame_id} and "
            f"{frame_b.frame_id}: need >= 8, got {len(shared_ids)}."
        )

    pts_a = np.array([frame_a.get_detection(mid).image_point for mid in shared_ids], dtype=np.float64)
    pts_b = np.array([frame_b.get_detection(mid).image_point for mid in shared_ids], dtype=np.float64)

    pts_a_norm = _undistort_points(pts_a, calibration)
    pts_b_norm = _undistort_points(pts_b, calibration)

    E, mask = cv2.findEssentialMat(
        pts_a_norm,
        pts_b_norm,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    if E is None or mask is None:
        raise RuntimeError("findEssentialMat failed to compute a valid model.")
    
    _, R, t, pose_mask = cv2.recoverPose(E, pts_a_norm, pts_b_norm, mask=mask)
    # marker_ids corresponds to pts_a_norm / pts_b_norm (same ordering)
    for mid, inlier, a_norm, b_norm in zip(shared_ids, pose_mask.ravel().astype(bool), pts_a_norm, pts_b_norm, strict=False):
        if inlier:
            continue  # only inspect the ones that got dropped

        # Triangulate with the recovered pose
        homog = cv2.triangulatePoints(
            np.hstack([np.eye(3), np.zeros((3, 1))]),
            np.hstack([R, t.reshape(3, 1)]),
            a_norm.reshape(2, 1),
            b_norm.reshape(2, 1),
        )
        point_cam1 = (homog[:3] / homog[3]).ravel()
        point_cam2 = R @ point_cam1 + t.ravel()

        depth1 = point_cam1[2]
        depth2 = point_cam2[2]

        # Reproject to each image
        reproj1 = point_cam1[:2] / depth1
        reproj2 = point_cam2[:2] / depth2
        err1 = np.linalg.norm(reproj1 - a_norm)
        err2 = np.linalg.norm(reproj2 - b_norm)

    inlier_mask = pose_mask.ravel().astype(bool)
    if inlier_mask.sum() < 8:
        raise RuntimeError(
            f"Too few inliers after pose recovery: {inlier_mask.sum()} (need >= 8)."
        )

    marker_ids = np.asarray(shared_ids, dtype=np.int32)[inlier_mask]
    pts_a_inliers = pts_a_norm[inlier_mask]
    pts_b_inliers = pts_b_norm[inlier_mask]

    return R, t.ravel(), marker_ids, pts_a_inliers, pts_b_inliers


def triangulate_markers_two_view(
    marker_ids: np.ndarray,
    pts_view1_norm: np.ndarray,
    pts_view2_norm: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    reprojection_threshold: float = 1e-3,
) -> TriangulationResult:
    """
    Triangulate marker positions from two normalized views.

    Parameters
    ----------
    marker_ids:
        Array of marker IDs corresponding to the provided correspondences.
    pts_view1_norm / pts_view2_norm:
        Nx2 arrays of normalized (undistorted) coordinates.
    R, t:
        Relative pose from view1 to view2.
    reprojection_threshold:
        Maximum normalized reprojection error allowed for a triangulated point.
    """
    ids = np.asarray(marker_ids, dtype=np.int32)
    pts1 = np.asarray(pts_view1_norm, dtype=np.float64)
    pts2 = np.asarray(pts_view2_norm, dtype=np.float64)
    t_vec = np.asarray(t, dtype=np.float64).reshape(3, 1)

    if ids.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return TriangulationResult(
            marker_ids=ids,
            points=np.empty((0, 3), dtype=np.float64),
            depths_view1=empty,
            depths_view2=empty,
            reprojection_errors=empty,
        )

    P1 = np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
    P2 = np.hstack((R.astype(np.float64), t_vec))

    homog_points = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points = cv2.convertPointsFromHomogeneous(homog_points.T).reshape(-1, 3)

    depths_view1 = points[:, 2]
    points_cam2 = (R @ points.T + t_vec).T
    depths_view2 = points_cam2[:, 2]

    proj1 = points[:, :2] / depths_view1[:, None]
    proj2 = points_cam2[:, :2] / depths_view2[:, None]

    err1 = np.linalg.norm(proj1 - pts1, axis=1)
    err2 = np.linalg.norm(proj2 - pts2, axis=1)
    reproj_err = np.maximum(err1, err2)

    valid_mask = (
        (depths_view1 > 0)
        & (depths_view2 > 0)
        & np.isfinite(reproj_err)
        & (reproj_err <= reprojection_threshold)
    )

    return TriangulationResult(
        marker_ids=ids[valid_mask],
        points=points[valid_mask],
        depths_view1=depths_view1[valid_mask],
        depths_view2=depths_view2[valid_mask],
        reprojection_errors=reproj_err[valid_mask],
    )


def bootstrap_reconstruction(
    state: SfMState,
    *,
    frame_pair: Optional[Tuple[int, int]] = None,
    checkpoint_path: Optional[Path] = Path("checkpoint_bootstrap.pkl"),
    ransac_threshold: float = 1e-3,
    reprojection_threshold: float = 1e-3,
) -> TriangulationResult:
    """
    Perform the initial two-view bootstrap and populate the SfM state with poses and markers.
    """
    if frame_pair is None:
        frame_a, frame_b = select_bootstrap_pair(state)
    else:
        try:
            frame_a = state.frame_by_id(frame_pair[0])
            frame_b = state.frame_by_id(frame_pair[1])
        except KeyError as exc:
            raise RuntimeError(f"Specified bootstrap frame not found: {exc}") from exc

    R, t, marker_ids, pts_a_norm, pts_b_norm = estimate_relative_pose(
        frame_a,
        frame_b,
        state.calibration,
        ransac_threshold=ransac_threshold,
    )

    tri_result = triangulate_markers_two_view(
        marker_ids,
        pts_a_norm,
        pts_b_norm,
        R,
        t,
        reprojection_threshold=reprojection_threshold,
    )

    if tri_result.marker_ids.size == 0:
        raise RuntimeError("Bootstrap triangulation failed: no valid 3D points recovered.")

    if frame_a.frame_id not in state.poses:
        state.add_pose(frame_a.frame_id, CameraPose())
    if frame_b.frame_id not in state.poses:
        state.add_pose(frame_b.frame_id, CameraPose(rotation=R, translation=t))

    for marker_id, point in zip(tri_result.marker_ids, tri_result.points, strict=False):
        state.add_marker_position(int(marker_id), point)

    assert tri_result.marker_ids.size >= 10, "Too few markers triangulated during bootstrap."
    assert np.all(tri_result.depths_view1 > 0) and np.all(
        tri_result.depths_view2 > 0
    ), "Negative depth detected during bootstrap."

    if checkpoint_path is not None:
        state.save(checkpoint_path)

    return tri_result


def build_detection_index(
    observations: Iterable[FrameObservation],
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Convert FrameObservation objects into a frame -> marker -> (x, y) index.
    """
    index: Dict[int, Dict[int, Tuple[float, float]]] = {}
    marker_ids: List[int] = []
    total_detections = 0

    for obs in observations:
        markers: Dict[int, Tuple[float, float]] = {}
        for marker_id, detection in obs.detections.items():
            point = np.asarray(detection.image_point, dtype=np.float64).ravel()
            markers[marker_id] = (float(point[0]), float(point[1]))
            marker_ids.append(marker_id)
        index[obs.frame_id] = markers
        total_detections += len(markers)

    if marker_ids:
        print(
            f"Detection index built: {len(index)} frames, "
            f"{total_detections} detections, "
            f"marker ID range {min(marker_ids)}–{max(marker_ids)}"
        )
    else:
        print("Detection index built: no detections found.")

    return index


def filter_frame_consistency(
    index: Dict[int, Dict[int, Tuple[float, float]]],
    config: FilterConfig,
) -> Tuple[Dict[int, Dict[int, Tuple[float, float]]], List[int], Dict[int, List[str]]]:
    """
    Remove frames that contain marker observations inconsistent with the global consensus.
    """
    marker_tracks: Dict[int, List[Tuple[int, float, float]]] = {}
    for frame_id, markers in index.items():
        for marker_id, (x, y) in markers.items():
            marker_tracks.setdefault(marker_id, []).append((frame_id, x, y))

    bad_frames: Set[int] = set()
    frame_reasons: Dict[int, List[str]] = {}
    frame_distances: Dict[int, List[float]] = {}

    def flag_frame(frame_id: int, reason: str) -> None:
        bad_frames.add(frame_id)
        frame_reasons.setdefault(frame_id, []).append(reason)

    for marker_id, track in marker_tracks.items():
        if not track:
            continue
        track_sorted = sorted(track, key=lambda entry: entry[0])
        frames = np.array([t[0] for t in track_sorted], dtype=np.int32)
        positions = np.array([t[1:] for t in track_sorted], dtype=np.float64)

        distances = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
        for idx, dist in enumerate(distances):
            frame_id = int(frames[idx])
            frame_distances.setdefault(frame_id, []).append(float(dist))

        if len(track_sorted) >= config.min_consensus:
            sigma = float(np.std(distances))
            threshold = float(np.median(distances)) + config.outlier_sigma * (sigma if sigma > 0 else 1.0)
            for idx, dist in enumerate(distances):
                frame_id = int(frames[idx])
                if dist > threshold or dist > config.max_pixel_deviation:
                    flag_frame(frame_id, f"marker {marker_id} deviates {dist:.1f}px")

        if len(track_sorted) >= 3:
            velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            if velocities.size:
                median_vel = float(np.median(velocities))
                ref_velocity = median_vel if median_vel > 1e-3 else float(np.mean(velocities) + 1e-3)
                for idx, vel in enumerate(velocities):
                    if vel > config.velocity_spike_multiplier * ref_velocity:
                        flag_frame(int(frames[idx + 1]), f"marker {marker_id} velocity spike {vel:.1f}")

    # Per-frame aggregate deviation checks
    if len(frame_distances) >= config.min_consensus:
        for frame_id, distances in frame_distances.items():
            if not distances:
                continue
            distances_arr = np.asarray(distances, dtype=np.float64)
            median_dist = float(np.median(distances_arr))
            mean_dist = float(np.mean(distances_arr))
            large_ratio = float(np.mean(distances_arr > config.per_frame_large_threshold))

            if median_dist > config.per_frame_median_threshold:
                flag_frame(frame_id, f"median deviation {median_dist:.1f}px")
            if mean_dist > config.per_frame_mean_threshold:
                flag_frame(frame_id, f"mean deviation {mean_dist:.1f}px")
            if large_ratio > config.per_frame_large_ratio:
                flag_frame(frame_id, f"{large_ratio:.0%} markers deviate > {config.per_frame_large_threshold}px")

    # Pairwise frame consistency
    frame_ids_sorted = sorted(index.keys())
    pair_consensus: Dict[int, int] = {fid: 0 for fid in frame_ids_sorted}
    pair_fail_notes: Dict[int, List[str]] = {}

    for idx_a, frame_a in enumerate(frame_ids_sorted):
        markers_a = index.get(frame_a, {})
        for frame_b in frame_ids_sorted[idx_a + 1 :]:
            markers_b = index.get(frame_b, {})
            shared_ids = set(markers_a) & set(markers_b)
            if len(shared_ids) < max(config.min_shared_markers, 8):
                continue
            diffs = [
                np.linalg.norm(
                    np.asarray(markers_a[mid], dtype=np.float64)
                    - np.asarray(markers_b[mid], dtype=np.float64)
                )
                for mid in shared_ids
            ]
            if not diffs:
                continue
            mean_diff = float(np.mean(diffs))
            if mean_diff <= config.frame_pair_distance_threshold:
                pair_consensus[frame_a] += 1
                pair_consensus[frame_b] += 1
            else:
                pair_fail_notes.setdefault(frame_a, []).append(
                    f"distance to frame {frame_b} is {mean_diff:.1f}px"
                )
                pair_fail_notes.setdefault(frame_b, []).append(
                    f"distance to frame {frame_a} is {mean_diff:.1f}px"
                )

    for frame_id in frame_ids_sorted:
        if pair_consensus.get(frame_id, 0) < config.min_pair_consensus:
            flag_frame(frame_id, f"pair consensus {pair_consensus.get(frame_id, 0)} < {config.min_pair_consensus}")
            for note in pair_fail_notes.get(frame_id, []):
                frame_reasons.setdefault(frame_id, []).append(note)

    filtered_index: Dict[int, Dict[int, Tuple[float, float]]] = {
        frame_id: markers for frame_id, markers in index.items() if frame_id not in bad_frames
    }

    if bad_frames:
        detail = {fid: frame_reasons[fid] for fid in sorted(bad_frames)}
        print(f"Frame consistency filter removed {len(bad_frames)} frames: {detail}")
    else:
        print("Frame consistency filter retained all frames.")

    return filtered_index, sorted(bad_frames), frame_reasons


def _sample_frame_pairs(frames: Sequence[int], config: FilterConfig) -> List[Tuple[int, int]]:
    if len(frames) < 2:
        return []
    frames_sorted = sorted(frames)
    pairs: List[Tuple[int, int]] = []

    if config.sample_strategy == "adaptive":
        for i in range(min(20, len(frames_sorted))):
            for offset in (5, 10, 15):
                j = i + offset
                if j < len(frames_sorted):
                    pairs.append((frames_sorted[i], frames_sorted[j]))
    else:  # default to strided sampling
        max_pairs = max(1, config.max_pairs_to_test)
        step = max(1, len(frames_sorted) // max_pairs)
        for i in range(0, len(frames_sorted), step):
            for offset in range(5, 31, 5):
                j = i + offset
                if j < len(frames_sorted):
                    pairs.append((frames_sorted[i], frames_sorted[j]))

    # Deduplicate while preserving order
    unique_pairs = list(dict.fromkeys(pairs))
    if not unique_pairs:
        # Fall back to all distinct pairs for small datasets
        for i in range(len(frames_sorted)):
            for j in range(i + 1, len(frames_sorted)):
                unique_pairs.append((frames_sorted[i], frames_sorted[j]))
    return unique_pairs


def validate_markers_epipolar(
    index: Dict[int, Dict[int, Tuple[float, float]]],
    calibration: CameraCalibration,
    config: FilterConfig,
) -> Tuple[Set[int], Dict[int, float]]:
    """
    Evaluate marker consistency across frame pairs using epipolar geometry.
    Returns a set of unreliable markers and per-marker confidence scores.
    """
    frame_ids = sorted(index.keys())
    if len(frame_ids) < 2:
        return set(), {}

    pairs = _sample_frame_pairs(frame_ids, config)
    if not pairs:
        return set(), {}

    marker_votes: Dict[int, List[int]] = {}

    for frame_a, frame_b in pairs:
        markers_a = index.get(frame_a, {})
        markers_b = index.get(frame_b, {})
        shared_ids = sorted(set(markers_a) & set(markers_b))
        if len(shared_ids) < max(config.min_shared_markers, 8):
            continue

        pts_a = np.array([markers_a[mid] for mid in shared_ids], dtype=np.float64)
        pts_b = np.array([markers_b[mid] for mid in shared_ids], dtype=np.float64)

        pts_a_norm = _undistort_points(pts_a, calibration)
        pts_b_norm = _undistort_points(pts_b, calibration)

        E, mask = cv2.findEssentialMat(
            pts_a_norm,
            pts_b_norm,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=config.epipolar_ransac_threshold,
        )
        if E is None or mask is None:
            continue

        inliers = mask.ravel().astype(bool)
        for idx, marker_id in enumerate(shared_ids):
            votes = marker_votes.setdefault(marker_id, [0, 0])
            votes[1] += 1
            if inliers[idx]:
                votes[0] += 1

    bad_markers: Set[int] = set()
    marker_confidence: Dict[int, float] = {}

    for marker_id, (inliers, total) in marker_votes.items():
        confidence = inliers / total if total else 0.0
        marker_confidence[marker_id] = confidence
        if confidence < config.epipolar_inlier_threshold:
            bad_markers.add(marker_id)

    if bad_markers:
        preview = sorted(bad_markers)
        suffix = "..." if len(preview) > 10 else ""
        print(
            f"Epipolar voting rejected {len(bad_markers)} marker IDs: "
            f"{preview[:10]}{suffix}"
        )
    else:
        print("Epipolar voting retained all markers.")

    return bad_markers, marker_confidence


def remove_markers_from_index(
    index: Dict[int, Dict[int, Tuple[float, float]]],
    bad_markers: Set[int],
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    if not bad_markers:
        return {frame_id: dict(markers) for frame_id, markers in index.items()}

    cleaned: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for frame_id, markers in index.items():
        filtered = {mid: coords for mid, coords in markers.items() if mid not in bad_markers}
        if filtered:
            cleaned[frame_id] = filtered
    return cleaned


def index_to_observations(
    index: Dict[int, Dict[int, Tuple[float, float]]],
    original_frames: Iterable[FrameObservation],
) -> List[FrameObservation]:
    lookup = {frame.frame_id: frame for frame in original_frames}
    filtered_frames: List[FrameObservation] = []

    for frame_id in sorted(index.keys()):
        original = lookup.get(frame_id)
        if original is None:
            continue

        detections: Dict[int, MarkerDetection] = {}
        for marker_id, (x, y) in index[frame_id].items():
            if marker_id in original.detections:
                det = original.detections[marker_id]
                confidence = det.confidence
            else:
                confidence = 1.0
            detections[marker_id] = MarkerDetection(
                marker_id=marker_id,
                image_point=np.array([x, y], dtype=np.float64),
                confidence=confidence,
            )

        if detections:
            filtered_frames.append(
                FrameObservation(
                    frame_id=frame_id,
                    detections=detections,
                    timestamp=original.timestamp,
                    metadata=original.metadata,
                )
            )

    return filtered_frames


def preprocess_detections(
    observations: Iterable[FrameObservation],
    calibration: CameraCalibration,
    config: Optional[FilterConfig] = None,
) -> Tuple[List[FrameObservation], Dict[str, object]]:
    """
    Run detection filtering pipeline returning cleaned observations and diagnostics.
    """
    if config is None:
        config = FilterConfig()

    diagnostics: Dict[str, object] = {
        "removed_frames": [],
        "removed_markers": [],
        "marker_confidence": {},
        "frame_reasons": {},
    }

    index = build_detection_index(observations)
    filtered_index, bad_frames, frame_reasons = filter_frame_consistency(index, config)
    diagnostics["removed_frames"] = bad_frames
    diagnostics["frame_reasons"] = frame_reasons

    bad_markers, marker_confidence = validate_markers_epipolar(filtered_index, calibration, config)
    diagnostics["removed_markers"] = sorted(bad_markers)
    diagnostics["marker_confidence"] = marker_confidence

    cleaned_index = remove_markers_from_index(filtered_index, bad_markers)
    filtered_observations = index_to_observations(cleaned_index, observations)

    print(f"Preprocessing retained {len(filtered_observations)} frames out of {len(index)}.")

    return filtered_observations, diagnostics


def _evaluate_bootstrap_pair(
    frame_a: int,
    frame_b: int,
    index: Dict[int, Dict[int, Tuple[float, float]]],
    calibration: CameraCalibration,
    config: FilterConfig,
) -> Optional[Dict[str, float]]:
    markers_a = index.get(frame_a, {})
    markers_b = index.get(frame_b, {})
    shared_ids = sorted(set(markers_a) & set(markers_b))
    if len(shared_ids) < max(config.min_shared_markers, 8):
        return None

    pts_a = np.array([markers_a[mid] for mid in shared_ids], dtype=np.float64)
    pts_b = np.array([markers_b[mid] for mid in shared_ids], dtype=np.float64)

    pts_a_norm = _undistort_points(pts_a, calibration)
    pts_b_norm = _undistort_points(pts_b, calibration)

    E, mask = cv2.findEssentialMat(
        pts_a_norm,
        pts_b_norm,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=config.epipolar_ransac_threshold,
    )
    if E is None or mask is None:
        return None

    inliers = mask.ravel().astype(bool)
    if not inliers.any():
        return None

    inlier_ratio = float(inliers.mean())
    if inlier_ratio < config.min_bootstrap_inlier_ratio:
        return None

    parallax = np.linalg.norm(pts_a_norm[inliers] - pts_b_norm[inliers], axis=1)
    median_parallax = float(np.median(parallax)) if parallax.size else 0.0

    return {
        "shared_markers": float(len(shared_ids)),
        "inliers": float(inliers.sum()),
        "inlier_ratio": inlier_ratio,
        "median_parallax": median_parallax,
    }


def _bootstrap_candidate_score(metrics: Dict[str, float]) -> float:
    shared_score = min(metrics["shared_markers"], 40.0) / 40.0
    parallax_score = min(metrics["median_parallax"], 0.1) / 0.1 if metrics["median_parallax"] > 0 else 0.0
    return metrics["inlier_ratio"] * 10.0 + shared_score * 5.0 + parallax_score * 5.0


def find_bootstrap_pair(
    index: Dict[int, Dict[int, Tuple[float, float]]],
    calibration: CameraCalibration,
    config: FilterConfig,
) -> Tuple[Tuple[int, int], List[Tuple[int, int, Dict[str, float]]]]:
    frame_ids = sorted(index.keys())
    pairs = _sample_frame_pairs(frame_ids, config)
    candidates: List[Tuple[int, int, Dict[str, float]]] = []

    for frame_a, frame_b in pairs:
        metrics = _evaluate_bootstrap_pair(frame_a, frame_b, index, calibration, config)
        if metrics is None:
            continue
        candidates.append((frame_a, frame_b, metrics))

    if not candidates:
        raise RuntimeError("No valid bootstrap pairs found.")

    candidates.sort(key=lambda item: _bootstrap_candidate_score(item[2]), reverse=True)
    best_pair = (candidates[0][0], candidates[0][1])

    print(
        f"Selected bootstrap pair {best_pair} with inlier ratio "
        f"{candidates[0][2]['inlier_ratio']:.2f} and "
        f"{int(candidates[0][2]['shared_markers'])} shared markers."
    )

    return best_pair, candidates


def _point_depth(point: np.ndarray, pose: CameraPose) -> float:
    camera_coords = pose.rotation @ point + pose.translation
    return float(camera_coords[2])


def validate_bootstrap_state(state: SfMState, config: FilterConfig) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    errors_dict = compute_reprojection_errors(state)
    all_errors = np.concatenate(
        [errs for errs in errors_dict.values() if errs.size > 0], axis=0
    ) if errors_dict else np.empty(0)

    if all_errors.size:
        median_err = float(np.median(all_errors))
        max_err = float(np.max(all_errors))
    else:
        median_err = float("inf")
        max_err = float("inf")

    if not np.isfinite(median_err) or median_err > config.max_median_error:
        issues.append(f"Median reprojection error {median_err:.2f}px exceeds threshold.")
    if np.isfinite(max_err) and max_err > config.max_single_error:
        issues.append(f"Maximum reprojection error {max_err:.2f}px exceeds threshold.")

    negative_depths = 0
    for marker_id, position in state.marker_positions.items():
        for pose in state.poses.values():
            if _point_depth(position, pose) <= 0:
                negative_depths += 1
                break
    if negative_depths:
        issues.append(f"{negative_depths} markers have non-positive depth.")

    if len(state.marker_positions) < config.min_bootstrap_points:
        issues.append(f"Too few markers triangulated ({len(state.marker_positions)}).")

    return len(issues) == 0, issues


def bootstrap_with_fallback(
    frames: Iterable[FrameObservation],
    calibration: CameraCalibration,
    config: FilterConfig,
    ransac_threshold: float = 1e-3,
    reprojection_threshold: float = 1e-3,
    max_attempts: int = 5,
) -> Tuple[SfMState, TriangulationResult, Dict[str, object]]:
    index = build_detection_index(frames)
    best_pair, candidates = find_bootstrap_pair(index, calibration, config)

    attempts = candidates[:max_attempts]
    last_issue = "No attempts made."

    for idx, (frame_a, frame_b, metrics) in enumerate(attempts, start=1):
        print(f"Bootstrap attempt {idx}: frames {frame_a} & {frame_b}")
        state = initialize_state(calibration, frames)
        try:
            tri = bootstrap_reconstruction(
                state,
                frame_pair=(frame_a, frame_b),
                checkpoint_path=None,
                ransac_threshold=ransac_threshold,
                reprojection_threshold=reprojection_threshold,
            )
        except RuntimeError as exc:
            last_issue = str(exc)
            continue

        valid, issues = validate_bootstrap_state(state, config)
        if valid:
            diagnostics = {
                "selected_pair": (frame_a, frame_b),
                "pair_metrics": metrics,
                "candidates": attempts,
            }
            print("Bootstrap validation succeeded.")
            return state, tri, diagnostics
        last_issue = "; ".join(issues)
        print(f"Bootstrap validation failed: {last_issue}")

    raise RuntimeError(f"Bootstrap failed for all candidate pairs. Last issue: {last_issue}")


def _solve_pnp_for_frame(
    frame: FrameObservation,
    state: SfMState,
    *,
    min_points: int = 6,
    reprojection_threshold: float = 2.0,
) -> Optional[CameraPose]:
    """Estimate the camera pose for a frame using markers already in the map."""
    object_points: List[np.ndarray] = []
    image_points: List[np.ndarray] = []

    for marker_id, detection in frame.detections.items():
        point3d = state.get_marker_position(marker_id)
        if point3d is None:
            continue
        object_points.append(point3d.astype(np.float64))
        image_points.append(np.asarray(detection.image_point, dtype=np.float64))

    if len(object_points) < min_points:
        return None

    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        state.calibration.camera_matrix,
        state.calibration.distortion_coeffs,
        flags=cv2.SOLVEPNP_AP3P,
        reprojectionError=2.0,
        iterationsCount=200,
        confidence=0.999,
    )

    if not success or inliers is None or len(inliers) < min_points:
        return None

    inlier_idx = inliers.ravel()
    obj_inliers = obj[inlier_idx]
    img_inliers = img[inlier_idx]

    rvec, tvec = cv2.solvePnPRefineLM(
        obj_inliers,
        img_inliers,
        state.calibration.camera_matrix,
        state.calibration.distortion_coeffs,
        rvec,
        tvec,
    )

    projected, _ = cv2.projectPoints(
        obj_inliers,
        rvec,
        tvec,
        state.calibration.camera_matrix,
        state.calibration.distortion_coeffs,
    )
    reproj_errors = np.linalg.norm(
        projected.reshape(-1, 2) - img_inliers.reshape(-1, 2),
        axis=1,
    )
    median_error = float(np.median(reproj_errors))
    if median_error > reprojection_threshold:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    return CameraPose(rotation=rotation, translation=translation)


def _triangulate_marker_with_frame_pair(
    marker_id: int,
    frame_new: FrameObservation,
    frame_ref: FrameObservation,
    pose_new: CameraPose,
    pose_ref: CameraPose,
    calibration: CameraCalibration,
    reprojection_threshold: float = 2.0,
) -> Optional[np.ndarray]:
    det_new = frame_new.detections.get(marker_id)
    det_ref = frame_ref.detections.get(marker_id)
    if det_new is None or det_ref is None:
        return None

    undist_new = _undistort_points(det_new.image_point, calibration)
    undist_ref = _undistort_points(det_ref.image_point, calibration)

    P_new = np.hstack((pose_new.rotation, pose_new.translation.reshape(3, 1)))
    P_ref = np.hstack((pose_ref.rotation, pose_ref.translation.reshape(3, 1)))

    point_h = cv2.triangulatePoints(P_ref, P_new, undist_ref.T, undist_new.T)
    point3d = cv2.convertPointsFromHomogeneous(point_h.T).reshape(3)

    depths = np.array(
        [
            pose_ref.rotation[2] @ point3d + pose_ref.translation[2],
            pose_new.rotation[2] @ point3d + pose_new.translation[2],
        ],
        dtype=np.float64,
    )

    if np.any(depths <= 0):
        return None

    rvec_ref, tvec_ref = _pose_to_rvec_tvec(pose_ref)
    rvec_new, tvec_new = _pose_to_rvec_tvec(pose_new)

    proj_ref, _ = cv2.projectPoints(point3d.reshape(1, 3), rvec_ref, tvec_ref, calibration.camera_matrix, calibration.distortion_coeffs)
    proj_new, _ = cv2.projectPoints(point3d.reshape(1, 3), rvec_new, tvec_new, calibration.camera_matrix, calibration.distortion_coeffs)

    err_ref = np.linalg.norm(np.asarray(det_ref.image_point) - proj_ref.reshape(-1, 2)[0])
    err_new = np.linalg.norm(np.asarray(det_new.image_point) - proj_new.reshape(-1, 2)[0])

    if max(err_ref, err_new) > reprojection_threshold:
        return None

    return point3d


def triangulate_new_markers(
    frame: FrameObservation,
    state: SfMState,
    reprojection_threshold: float = 2.0,
) -> int:
    """Triangulate and add markers observed in the given frame but not yet in the map."""
    pose_new = state.poses[frame.frame_id]
    added = 0

    for marker_id, detection in frame.detections.items():
        if state.get_marker_position(marker_id) is not None:
            continue

        for posed_frame_id in state.posed_frames():
            if posed_frame_id == frame.frame_id:
                continue
            frame_ref = state.frame_by_id(posed_frame_id)
            if marker_id not in frame_ref.detections:
                continue

            pose_ref = state.poses[posed_frame_id]
            point3d = _triangulate_marker_with_frame_pair(
                marker_id,
                frame,
                frame_ref,
                pose_new,
                pose_ref,
                state.calibration,
                reprojection_threshold=reprojection_threshold,
            )
            if point3d is not None:
                state.add_marker_position(marker_id, point3d)
                added += 1
                break

    return added


def incremental_pose_estimation(
    state: SfMState,
    min_points: int = 6,
    reprojection_threshold: float = 2.0,
    checkpoint_path: Optional[Path] = Path("checkpoint_incremental.pkl"),
) -> None:
    """
    Solve camera poses for remaining frames using PnP and expand the marker map incrementally.
    """
    frames_queue = [frame for frame in state.frames if not state.has_pose(frame.frame_id)]
    if not frames_queue:
        return

    attempts_without_progress = 0
    while frames_queue:
        frame = frames_queue.pop(0)
        pose = _solve_pnp_for_frame(
            frame,
            state,
            min_points=min_points,
            reprojection_threshold=reprojection_threshold,
        )
        if pose is None:
            frames_queue.append(frame)
            attempts_without_progress += 1
            if attempts_without_progress >= len(frames_queue):
                break
            continue

        state.add_pose(frame.frame_id, pose)
        triangulate_new_markers(frame, state, reprojection_threshold=reprojection_threshold)
        attempts_without_progress = 0

    assert len(state.poses) >= 3, "Too few frames registered after incremental stage."

    if checkpoint_path is not None:
        state.save(checkpoint_path)


def _build_bundle_adjustment_problem(
    state: SfMState,
    frame_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    posed_ids = state.posed_frames()
    if not posed_ids:
        raise ValueError("No camera poses available for bundle adjustment.")

    anchor_id = posed_ids[0]

    if frame_ids is None:
        frames_considered = posed_ids
    else:
        frames_considered = sorted({fid for fid in frame_ids if state.has_pose(fid)})
        if anchor_id not in frames_considered:
            frames_considered.insert(0, anchor_id)

    frames_considered = [fid for fid in frames_considered if state.has_pose(fid)]
    if len(frames_considered) < 2:
        raise ValueError("Need at least two posed frames for bundle adjustment.")

    frame_ids_opt = [fid for fid in frames_considered if fid != anchor_id]
    frame_index_opt = {fid: idx for idx, fid in enumerate(frame_ids_opt)}

    marker_ids_set = set()
    for fid in frames_considered:
        frame = state.frame_by_id(fid)
        for marker_id in frame.detections:
            if state.get_marker_position(marker_id) is not None:
                marker_ids_set.add(marker_id)
    if not marker_ids_set:
        raise ValueError("No markers available for bundle adjustment.")

    marker_ids = sorted(marker_ids_set)
    marker_index = {mid: idx for idx, mid in enumerate(marker_ids)}

    obs_frame_ids: List[int] = []
    obs_frame_opt_idx: List[int] = []
    obs_marker_idx: List[int] = []
    obs_points: List[np.ndarray] = []

    fixed_pose_rt: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for fid in frames_considered:
        is_opt = fid in frame_index_opt
        if not is_opt:
            fixed_pose_rt[fid] = _pose_to_rvec_tvec(state.poses[fid])
        frame = state.frame_by_id(fid)
        for marker_id in marker_ids:
            if marker_id not in frame.detections:
                continue
            if state.get_marker_position(marker_id) is None:
                continue
            obs_frame_ids.append(fid)
            obs_frame_opt_idx.append(frame_index_opt.get(fid, -1))
            obs_marker_idx.append(marker_index[marker_id])
            obs_points.append(np.asarray(frame.detections[marker_id].image_point, dtype=np.float64))

    if not obs_points:
        raise ValueError("No valid observations for bundle adjustment.")

    return {
        "anchor_id": anchor_id,
        "frame_ids_opt": frame_ids_opt,
        "frame_index_opt": frame_index_opt,
        "fixed_pose_rt": fixed_pose_rt,
        "marker_ids": marker_ids,
        "marker_index": marker_index,
        "obs_frame_ids": np.asarray(obs_frame_ids, dtype=np.int32),
        "obs_frame_opt_idx": np.asarray(obs_frame_opt_idx, dtype=np.int32),
        "obs_marker_idx": np.asarray(obs_marker_idx, dtype=np.int32),
        "obs_points": np.asarray(obs_points, dtype=np.float64),
    }


@dataclass(slots=True)
class PyCeresOptions:
    """
    Light wrapper around the Ceres solver configuration.

    Parameters
    ----------
    linear_solver : pyceres.LinearSolverType
        Linear solver backend used by Ceres.
    loss : Optional[str]
        One of {'huber', 'cauchy', 'soft_l1', 'trivial', None}.
    loss_scale : float
        Scaling factor passed to the chosen loss.
    max_iterations : int
        Maximum number of solver iterations.
    report_full : bool
        If True, print the full solver report to stdout.
    """

    linear_solver: Optional["pyceres.LinearSolverType"] = None
    loss: Optional[str] = "huber"
    loss_scale: float = 1.0
    max_iterations: int = 100
    report_full: bool = False


def _make_loss_function(loss: Optional[str], scale: float) -> "pyceres.LossFunction":
    if loss is None:
        return pyceres.TrivialLoss()
    normalized = loss.lower()
    if normalized in {"trivial", "none"}:
        return pyceres.TrivialLoss()
    if normalized == "huber":
        return pyceres.HuberLoss(scale)
    if normalized == "cauchy":
        return pyceres.CauchyLoss(scale)
    if normalized in {"soft_l1", "softl1"}:
        return pyceres.SoftLOneLoss(scale)
    raise ValueError(f"Unsupported loss function '{loss}'.")


class _ReprojectionCost(pyceres.CostFunction):
    __slots__ = ("_observed", "_camera_matrix", "_distortion", "_eps")

    def __init__(
        self,
        observed_xy: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])
        self._observed = np.asarray(observed_xy, dtype=np.float64)
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self._distortion = np.asarray(distortion, dtype=np.float64)
        self._eps = float(eps)

    def _project(self, camera_vec: np.ndarray, point_vec: np.ndarray) -> np.ndarray:
        rvec = camera_vec[:3].reshape(3, 1)
        tvec = camera_vec[3:].reshape(3, 1)
        point = point_vec.reshape(1, 3)
        projected, _ = cv2.projectPoints(
            point,
            rvec,
            tvec,
            self._camera_matrix,
            self._distortion,
        )
        return projected.reshape(-1, 2)[0]

    def _residual(self, camera_vec: np.ndarray, point_vec: np.ndarray) -> np.ndarray:
        reprojection = self._project(camera_vec, point_vec)
        return reprojection - self._observed

    def _finite_difference(
        self,
        camera_vec: np.ndarray,
        point_vec: np.ndarray,
        wrt_camera: bool,
    ) -> np.ndarray:
        base = camera_vec if wrt_camera else point_vec
        dim = base.shape[0]
        jac = np.zeros((2, dim), dtype=np.float64)
        for col in range(dim):
            delta = np.zeros_like(base)
            delta[col] = self._eps
            if wrt_camera:
                res_plus = self._residual(camera_vec + delta, point_vec)
                res_minus = self._residual(camera_vec - delta, point_vec)
            else:
                res_plus = self._residual(camera_vec, point_vec + delta)
                res_minus = self._residual(camera_vec, point_vec - delta)
            jac[:, col] = (res_plus - res_minus) / (2.0 * self._eps)
        return jac

    def Evaluate(self, parameters, residuals, jacobians) -> bool:  # noqa: N802 (Ceres API)
        camera_vec = np.asarray(parameters[0], dtype=np.float64)
        point_vec = np.asarray(parameters[1], dtype=np.float64)
        residual = self._residual(camera_vec, point_vec)
        residuals[0] = residual[0]
        residuals[1] = residual[1]

        if jacobians is not None:
            if jacobians[0] is not None:
                jac_cam = self._finite_difference(camera_vec, point_vec, wrt_camera=True).reshape(-1)
                for idx, value in enumerate(jac_cam):
                    jacobians[0][idx] = value
            if jacobians[1] is not None:
                jac_pt = self._finite_difference(camera_vec, point_vec, wrt_camera=False).reshape(-1)
                for idx, value in enumerate(jac_pt):
                    jacobians[1][idx] = value
        return True


def _initialize_camera_blocks(state: SfMState, problem: Dict[str, object]) -> Dict[int, np.ndarray]:
    blocks: Dict[int, np.ndarray] = {}
    for frame_id in problem["obs_frame_ids"]:
        fid = int(frame_id)
        if fid not in blocks:
            pose = state.poses[fid]
            rvec, tvec = _pose_to_rvec_tvec(pose)
            block = np.ascontiguousarray(np.concatenate((rvec.ravel(), tvec.ravel()), axis=0), dtype=np.float64)
            blocks[fid] = block
    anchor_id = int(problem["anchor_id"])
    if anchor_id not in blocks:
        pose = state.poses[anchor_id]
        rvec, tvec = _pose_to_rvec_tvec(pose)
        blocks[anchor_id] = np.ascontiguousarray(np.concatenate((rvec.ravel(), tvec.ravel()), axis=0), dtype=np.float64)
    for frame_id in problem["frame_ids_opt"]:
        fid = int(frame_id)
        if fid not in blocks:
            pose = state.poses[fid]
            rvec, tvec = _pose_to_rvec_tvec(pose)
            blocks[fid] = np.ascontiguousarray(np.concatenate((rvec.ravel(), tvec.ravel()), axis=0), dtype=np.float64)
    return blocks


def _initialize_marker_blocks(state: SfMState, problem: Dict[str, object]) -> Dict[int, np.ndarray]:
    blocks: Dict[int, np.ndarray] = {}
    for marker_id in problem["marker_ids"]:
        position = state.marker_positions[int(marker_id)]
        blocks[int(marker_id)] = np.ascontiguousarray(position.astype(np.float64).ravel(), dtype=np.float64)
    return blocks


def _update_state_from_blocks(
    state: SfMState,
    problem: Dict[str, object],
    camera_blocks: Dict[int, np.ndarray],
    marker_blocks: Dict[int, np.ndarray],
) -> None:
    for frame_id in problem["frame_ids_opt"]:
        block = camera_blocks[int(frame_id)]
        rvec = block[:3].reshape(3, 1)
        tvec = block[3:].reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        state.add_pose(int(frame_id), CameraPose(rotation=rotation, translation=tvec.copy()))

    for marker_id in problem["marker_ids"]:
        block = marker_blocks[int(marker_id)]
        state.add_marker_position(int(marker_id), block.copy())


def bundle_adjustment_pyceres(
    state: SfMState,
    ceres_opts: Optional[PyCeresOptions] = None,
    *,
    frame_ids: Optional[Sequence[int]] = None,
    checkpoint_path: Optional[Path] = Path("checkpoint_ba.pkl"),
) -> "pyceres.SolverSummary":
    """
    Run bundle adjustment using the Ceres Solver Python bindings.

    The function updates ``state`` in-place and optionally serializes the new state.
    """
    options = ceres_opts or PyCeresOptions()
    ba_problem = _build_bundle_adjustment_problem(state, frame_ids=frame_ids)
    camera_blocks = _initialize_camera_blocks(state, ba_problem)
    marker_blocks = _initialize_marker_blocks(state, ba_problem)

    problem = pyceres.Problem()

    frames_free = {int(fid) for fid in ba_problem["frame_ids_opt"]}
    for frame_id, block in camera_blocks.items():
        problem.add_parameter_block(block, block.size)
        if frame_id not in frames_free:
            problem.set_parameter_block_constant(block)

    for block in marker_blocks.values():
        problem.add_parameter_block(block, block.size)

    obs_frame_ids = ba_problem["obs_frame_ids"]
    obs_marker_idx = ba_problem["obs_marker_idx"]
    obs_points = ba_problem["obs_points"]
    marker_ids = ba_problem["marker_ids"]

    for idx, frame_id in enumerate(obs_frame_ids):
        cam_block = camera_blocks[int(frame_id)]
        marker_id = int(marker_ids[int(obs_marker_idx[idx])])
        point_block = marker_blocks[marker_id]
        cost_function = _ReprojectionCost(
            obs_points[idx],
            state.calibration.camera_matrix,
            state.calibration.distortion_coeffs,
        )
        loss_function = _make_loss_function(options.loss, options.loss_scale)
        problem.add_residual_block(cost_function, loss_function, [cam_block, point_block])

    solver_options = pyceres.SolverOptions()
    solver_options.max_num_iterations = options.max_iterations
    solver_options.linear_solver_type = options.linear_solver or pyceres.LinearSolverType.DENSE_SCHUR
    solver_options.minimizer_progress_to_stdout = options.report_full

    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)

    if options.report_full:
        print(summary.FullReport())

    if not summary.IsSolutionUsable():
        raise RuntimeError(f"Ceres failed: {summary.message}")

    _update_state_from_blocks(state, ba_problem, camera_blocks, marker_blocks)

    median_error, _ = compute_median_mean_reprojection_error(state)
    if not np.isfinite(median_error) or median_error >= 1.0:
        raise RuntimeError(
            f"High reprojection error after Ceres bundle adjustment: {median_error:.3f}"
        )

    if checkpoint_path is not None:
        state.save(checkpoint_path)

    return summary


def bundle_adjustment(
    state: SfMState,
    *,
    frame_ids: Optional[Sequence[int]] = None,
    options: Optional[PyCeresOptions] = None,
    checkpoint_path: Optional[Path] = Path("checkpoint_ba.pkl"),
) -> "pyceres.SolverSummary":
    """
    Convenience wrapper mirroring the previous SciPy signature while delegating to Ceres.
    """
    return bundle_adjustment_pyceres(
        state,
        frame_ids=frame_ids,
        ceres_opts=options,
        checkpoint_path=checkpoint_path,
    )


def align_state_to_object_frame(
    state: SfMState,
    *,
    origin_marker_id: Optional[int] = None,
    x_axis_marker_id: Optional[int] = None,
    y_axis_marker_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-express marker positions and camera poses in an object-fixed frame.

    The new frame is derived either from user-specified reference markers or, when those
    are not provided, from the principal components of the marker cloud:
        - origin at the selected marker or the centroid
        - x-axis from the origin towards ``x_axis_marker_id`` (when specified) or the
          largest PCA component
        - y-axis from the origin towards ``y_axis_marker_id`` (when specified) or the
          second PCA component
        - z-axis completing a right-handed basis (aimed towards the cameras)

    Returns
    -------
    rotation : np.ndarray
        The 3x3 rotation matrix applied to world coordinates (object = R * (world - origin)).
    origin : np.ndarray
        The translation vector (centroid or chosen marker) that was subtracted before rotation.
    """
    if not state.marker_positions:
        raise ValueError("No marker positions available to align.")

    marker_positions = state.marker_positions
    points_world = np.stack([pos.copy() for pos in marker_positions.values()], axis=0)
    centroid_world = points_world.mean(axis=0)

    # Helper to build axes from explicit marker IDs.
    def _axes_from_markers(
        origin_vec: np.ndarray,
        x_vec_world: np.ndarray,
        y_vec_world: np.ndarray,
    ) -> Optional[np.ndarray]:
        eps = 1e-9
        x_vec = x_vec_world - origin_vec                           # vector from origin to X marker
        x_norm = np.linalg.norm(x_vec)                             # length of vector
        if x_norm < eps:                                           # error if same point
            return None
        x_axis_local = x_vec / x_norm                              # normalize to unit vector

        y_vec = y_vec_world - origin_vec                           # vector from origin to Y marker
        y_vec -= np.dot(y_vec, x_axis_local) * x_axis_local        # remove X component (Gram-Schmidt)
        y_norm = np.linalg.norm(y_vec)                             # length after orthogonalization
        if y_norm < eps:                                           # error if too close to X
            return None
        y_axis_local = y_vec / y_norm                              # normalize to unit vector

        z_axis_local = np.cross(x_axis_local, y_axis_local)
        z_norm = np.linalg.norm(z_axis_local)
        if z_norm < eps:
            return None
        z_axis_local /= z_norm

        return np.stack((x_axis_local, y_axis_local, z_axis_local), axis=0)

    # Choose origin in world coordinates.
    if origin_marker_id is not None:
        origin_candidate = marker_positions.get(origin_marker_id)
        if origin_candidate is None:
            warnings.warn(
                f"Marker ID {origin_marker_id} not found; falling back to centroid origin.",
                RuntimeWarning,
                stacklevel=2,
            )
            origin_world = centroid_world.copy()
        else:
            origin_world = origin_candidate.copy()
    else:
        origin_world = centroid_world.copy()

    rotation: Optional[np.ndarray] = None
    if x_axis_marker_id is not None and y_axis_marker_id is not None:
        x_marker = marker_positions.get(x_axis_marker_id)
        y_marker = marker_positions.get(y_axis_marker_id)
        if x_marker is None or y_marker is None:
            missing = [
                str(marker_id)
                for marker_id, marker in ((x_axis_marker_id, x_marker), (y_axis_marker_id, y_marker))
                if marker is None
            ]
            warnings.warn(
                f"Marker(s) {', '.join(missing)} not found; falling back to PCA alignment.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            rotation = _axes_from_markers(origin_world, x_marker, y_marker)
            if rotation is None:
                warnings.warn(
                    "Marker-based axis construction failed due to degeneracy; falling back to PCA alignment.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if rotation is None:
        centered = points_world - centroid_world
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        x_axis, y_axis, z_axis = vh

        if state.poses:
            camera_centers = []
            for pose in state.poses.values():
                center = -pose.rotation.T @ pose.translation
                camera_centers.append(center)
            camera_centroid = np.mean(camera_centers, axis=0)
            if np.dot(z_axis, camera_centroid - centroid_world) < 0:
                z_axis = -z_axis
        else:
            if z_axis[2] < 0:
                z_axis = -z_axis

        y_axis = np.cross(z_axis, x_axis)
        if np.linalg.norm(y_axis) < 1e-9:
            raise ValueError("Degenerate marker configuration; cannot define object frame.")
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        z_axis /= np.linalg.norm(z_axis)

        rotation = np.stack((x_axis, y_axis, z_axis), axis=0)

    origin = origin_world.copy()

    # Update marker positions.
    new_positions: Dict[int, np.ndarray] = {}
    for marker_id, position in state.marker_positions.items():
        new_positions[marker_id] = rotation @ (position - origin)
    for marker_id, position in new_positions.items():
        state.marker_positions[marker_id] = position

    # Update camera poses (object frame -> camera).
    new_poses: Dict[int, CameraPose] = {}
    for frame_id, pose in state.poses.items():
        new_rotation = pose.rotation @ rotation.T
        new_translation = pose.rotation @ origin + pose.translation
        new_poses[frame_id] = CameraPose(rotation=new_rotation, translation=new_translation)
    state.poses.update(new_poses)

    return rotation, origin


def compute_reprojection_errors(
    state: SfMState,
    frame_ids: Optional[Iterable[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Compute pixel reprojection errors for the specified frames.

    Returns a dictionary mapping frame_id to an array of per-marker errors.
    """
    if frame_ids is None:
        frames_iterable = state.poses.keys()
    else:
        frames_iterable = frame_ids

    errors: Dict[int, np.ndarray] = {}
    K = state.calibration.camera_matrix
    dist = state.calibration.distortion_coeffs

    for frame_id in frames_iterable:
        pose = state.poses.get(frame_id)
        if pose is None:
            continue
        frame = state.frame_by_id(frame_id)
        rvec, tvec = _pose_to_rvec_tvec(pose)

        frame_errors: List[float] = []
        for marker_id, detection in frame.detections.items():
            point3d = state.marker_positions.get(marker_id)
            if point3d is None:
                continue
            projected, _ = cv2.projectPoints(
                point3d.reshape(1, 3),
                rvec,
                tvec,
                K,
                dist,
            )
            projected_xy = projected.reshape(-1, 2)[0]
            observed_xy = np.asarray(detection.image_point, dtype=np.float64).ravel()
            frame_errors.append(float(np.linalg.norm(observed_xy - projected_xy)))

        errors[frame_id] = np.asarray(frame_errors, dtype=np.float64)

    return errors


def compute_median_mean_reprojection_error(
    state: SfMState,
    frame_ids: Optional[Iterable[int]] = None,
) -> float:
    """Return the median reprojection error across the specified frames."""
    per_frame = compute_reprojection_errors(state, frame_ids=frame_ids)
    all_errors = np.concatenate(
        [errs for errs in per_frame.values() if errs.size > 0], axis=0
    ) if per_frame else np.empty(0)
    if all_errors.size == 0:
        return float("nan")
    median = float(np.median(all_errors))
    mean = float(np.mean(all_errors))
    return median, mean


# Backwards-compatibility alias retained for historical imports.
compute_median_reprojection_error = compute_median_mean_reprojection_error


__all__ = [
    "CameraCalibration",
    "MarkerDetection",
    "FrameObservation",
    "CameraPose",
    "TriangulationResult",
    "SfMState",
    "PyCeresOptions",
    "load_calibration",
    "load_frame_observations",
    "initialize_state",
    "select_bootstrap_pair",
    "estimate_relative_pose",
    "triangulate_markers_two_view",
    "bootstrap_reconstruction",
    "FilterConfig",
    "build_detection_index",
    "filter_frame_consistency",
    "validate_markers_epipolar",
    "remove_markers_from_index",
    "index_to_observations",
    "preprocess_detections",
    "find_bootstrap_pair",
    "validate_bootstrap_state",
    "bootstrap_with_fallback",
    "align_state_to_object_frame",
    "compute_reprojection_errors",
    "compute_median_reprojection_error",
    "incremental_pose_estimation",
    "bundle_adjustment_pyceres",
    "bundle_adjustment",
]
