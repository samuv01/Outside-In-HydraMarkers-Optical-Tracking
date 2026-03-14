from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from track_from_map_multi import (
    MapTracker,
    MultiMapTracker,
    load_camera_intrinsics,
    load_detection_records,
    load_multi_maps,
    rotation_matrix_to_euler_angles,
    run_tracking_from_records,
    load_npz_image_lookup,
)

def main() -> None:
    map_path_1 = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\10x10_3x3tag_2x5tag_NUMBER_1\mapping\sfm_outputs\marker_map_aligned.npz"
    )
    map_path_2 = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\10x10_3x3tag_2x5tag_NUMBER_2\mapping\sfm_outputs\marker_map_aligned.npz"
    )
    maps_paths_dict: Dict[str, Path] = {"object_1": map_path_1, "object_2": map_path_2}

    detections_path_1 = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\10x10_3x3tag_2x5tag_NUMBER_1\tracking\npz_marker_results_MARKER_1\detections_data.npz"
    )
    detections_path_2 = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\10x10_3x3tag_2x5tag_NUMBER_2\tracking\npz_marker_results_MARKER_2\detections_data.npz"
    )
    detections_dict_paths: Dict[str, Path] = {
        "object_1": detections_path_1,
        "object_2": detections_path_2,
    }

    camera_matrix_path = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_mtx.npy"
    )
    distortion_path = Path(
        r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_dist.npy"
    )

    frames_npz_paths = {
    "object_1": Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\tracking_rec_test_20251028_091809.npz"),
    "object_2": Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\MULTI_TRACK\tracking_rec_test_20251028_091809.npz"),
}

    display = False

    required_paths = [
        (camera_matrix_path, "camera matrix"),
        (distortion_path, "distortion"),
    ]
    for object_id, path in maps_paths_dict.items():
        required_paths.append((path, f"{object_id} map"))
    for object_id, path in detections_dict_paths.items():
        required_paths.append((path, f"{object_id} detections"))

    for path, label in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")

    camera_matrix, distortion = load_camera_intrinsics(camera_matrix_path, distortion_path)
    maps_dict = load_multi_maps(maps_paths_dict)

    print("Loaded maps:")
    for object_id, map_data in maps_dict.items():
        print(
            f"  {object_id}: {map_data.marker_ids.size} markers, "
            f"{map_data.pose_ids.size} reference poses from {maps_paths_dict[object_id]}"
        )

    detections_dict_all: Dict[str, List[dict]] = {
        object_id: load_detection_records(path) for object_id, path in detections_dict_paths.items()
    }
    for object_id, records in detections_dict_all.items():
        print(f"{object_id}: loaded {len(records)} detection records from {detections_dict_paths[object_id]}")

    tracker = MultiMapTracker(maps_dict, camera_matrix, distortion)
    print("Multi-object tracker initialized.")

    object_ids = list(maps_dict.keys())
    max_frames = max((len(detections_dict_all[obj]) for obj in object_ids), default=0)
    if max_frames == 0:
        print("No detection frames available; exiting.")
        return

    results = {
        object_id: {
            "frame_indices": [],
            "timestamps": [],
            "translation_x": [],
            "translation_y": [],
            "translation_z": [],
            "rotation_x": [],
            "rotation_y": [],
            "rotation_z": [],
            "euler_x_rad": [],
            "euler_y_rad": [],
            "euler_z_rad": [],
            "euler_x_deg": [],
            "euler_y_deg": [],
            "euler_z_deg": [],
            "rms_error": [],
            "frame_to_idx": {},
            "inliers": [],
        }
        for object_id in object_ids
    }
    

    frame_timestamps: Dict[str, float] = {}

    for frame_idx in range(max_frames):
        detections_this_frame: Dict[str, Dict[int, np.ndarray]] = {}
        frame_reference: Dict[str, int] = {}

        for object_id in object_ids:
            records = detections_dict_all[object_id]
            if frame_idx < len(records):
                record = records[frame_idx]
                detections_arr = np.asarray(record.get("detections", np.empty((0, 3))), dtype=np.float64)
                detections_map = {
                    int(marker_id): np.array([float(x), float(y)], dtype=np.float64)
                    for marker_id, x, y in detections_arr
                }
                frame_reference[object_id] = int(record.get("index", frame_idx))
                frame_timestamps[object_id] = float(record.get("timestamp", float("nan")))
            else:
                detections_map = {}
                frame_reference[object_id] = frame_idx
                frame_timestamps[object_id] = float("nan")
            detections_this_frame[object_id] = detections_map

        poses = tracker.estimate_poses_multi(detections_this_frame, frame_ids=frame_reference)
        errors = tracker.compute_reprojection_errors_multi(detections_this_frame, poses)

        for object_id in object_ids:
            pose = poses.get(object_id)
            metrics = errors.get(object_id, {})
            rms_error = float(metrics.get("rms", float("nan")))
            frame_ref = frame_reference[object_id]
            timestamp = frame_timestamps.get(object_id, float("nan"))
            result = results[object_id]
            frame_to_idx = result["frame_to_idx"]
            frame_to_idx[frame_ref] = len(result["frame_indices"])
            result["frame_indices"].append(frame_ref)
            result["timestamps"].append(timestamp)

            tracker_obj = tracker.trackers[object_id]
            if pose is not None:
                tvec = pose.tvec.reshape(-1)
                rvec = pose.rvec.reshape(-1)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
                roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])
                inlier_count = len(pose.inlier_ids)
                result["translation_x"].append(float(tvec[0]))
                result["translation_y"].append(float(tvec[1]))
                result["translation_z"].append(float(tvec[2]))
                result["rotation_x"].append(float(rvec[0]))
                result["rotation_y"].append(float(rvec[1]))
                result["rotation_z"].append(float(rvec[2]))
                result["euler_x_rad"].append(float(roll))
                result["euler_y_rad"].append(float(pitch))
                result["euler_z_rad"].append(float(yaw))
                result["euler_x_deg"].append(float(roll_deg))
                result["euler_y_deg"].append(float(pitch_deg))
                result["euler_z_deg"].append(float(yaw_deg))
                result["rms_error"].append(rms_error)
                result["inliers"].append(float(inlier_count))

                if np.isfinite(timestamp):
                    print(f"[Frame {frame_ref}] {object_id} (ts={timestamp:.3f}s):")
                else:
                    print(f"[Frame {frame_ref}] {object_id}:")
                print(
                    f"  tvec: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})"
                )
                print(
                    f"  rvec: ({rvec[0]:.3f}, {rvec[1]:.3f}, {rvec[2]:.3f})"
                )
                print(
                    f"  euler_rad: ({roll:.3f}, {pitch:.3f}, {yaw:.3f})"
                )
                print(
                    f"  euler_deg: ({roll_deg:.3f}, {pitch_deg:.3f}, {yaw_deg:.3f})"
                )
                print(f"  inliers: {inlier_count}")
                if np.isfinite(rms_error):
                    print(f"  reproj_rms: {rms_error:.3f}px")
                else:
                    print("  reproj_rms: n/a")
            else:
                nan = float("nan")
                result["translation_x"].append(nan)
                result["translation_y"].append(nan)
                result["translation_z"].append(nan)
                result["rotation_x"].append(nan)
                result["rotation_y"].append(nan)
                result["rotation_z"].append(nan)
                result["euler_x_rad"].append(nan)
                result["euler_y_rad"].append(nan)
                result["euler_z_rad"].append(nan)
                result["euler_x_deg"].append(nan)
                result["euler_y_deg"].append(nan)
                result["euler_z_deg"].append(nan)
                result["rms_error"].append(rms_error)
                result["inliers"].append(float("nan"))
                detection_count = len(detections_this_frame[object_id])
                tracker_obj = tracker.trackers[object_id]
                required = getattr(tracker_obj, "min_required_markers", 6)
                ts_suffix = f" ts={timestamp:.3f}s" if np.isfinite(timestamp) else ""
                if detection_count < required:
                    print(
                        f"[Frame {frame_ref}] {object_id}:{ts_suffix} insufficient markers "
                        f"({detection_count}/{required}) for pose."
                    )
                elif getattr(tracker_obj, "awaiting_reentry", False):
                    print(
                        f"[Frame {frame_ref}] {object_id}:{ts_suffix} pose pending confirmation after marker dropout "
                        f"(detections={detection_count})."
                    )
                else:
                    print(
                        f"[Frame {frame_ref}] {object_id}:{ts_suffix} pose solution unavailable "
                        f"(detections={detection_count})."
                    )

            pending_values = getattr(tracker_obj, "confirmed_pending_pose_values", [])
            if pending_values:
                for entry in pending_values:
                    frame_pending = entry.get("frame")
                    idx = frame_to_idx.get(frame_pending)
                    if idx is None:
                        continue
                    pending_tvec = np.asarray(entry.get("tvec"), dtype=float).reshape(-1)
                    pending_rvec = np.asarray(entry.get("rvec"), dtype=float).reshape(-1)
                    result["translation_x"][idx] = float(pending_tvec[0])
                    result["translation_y"][idx] = float(pending_tvec[1])
                    result["translation_z"][idx] = float(pending_tvec[2])
                    result["rotation_x"][idx] = float(pending_rvec[0])
                    result["rotation_y"][idx] = float(pending_rvec[1])
                    result["rotation_z"][idx] = float(pending_rvec[2])
                    rot_pending, _ = cv2.Rodrigues(pending_rvec.reshape(3, 1))
                    roll_p, pitch_p, yaw_p = rotation_matrix_to_euler_angles(rot_pending)
                    result["euler_x_rad"][idx] = float(roll_p)
                    result["euler_y_rad"][idx] = float(pitch_p)
                    result["euler_z_rad"][idx] = float(yaw_p)
                    result["euler_x_deg"][idx] = float(np.degrees(roll_p))
                    result["euler_y_deg"][idx] = float(np.degrees(pitch_p))
                    result["euler_z_deg"][idx] = float(np.degrees(yaw_p))
                    result["rms_error"][idx] = float("nan")
                    result["inliers"][idx] = float(entry.get("inliers", float("nan")))
                tracker_obj.confirmed_pending_frames = []
                tracker_obj.confirmed_pending_pose_values = []
            else:
                tracker_obj.confirmed_pending_frames = []
                tracker_obj.confirmed_pending_pose_values = []

    for object_id in object_ids:
        result = results[object_id]
        result.pop("frame_to_idx", None)
        frames = result["frame_indices"]
        timestamps_series = np.asarray(result.get("timestamps", []), dtype=float)
        translations_x = result["translation_x"]
        translations_y = result["translation_y"]
        translations_z = result["translation_z"]
        euler_x_deg = result["euler_x_deg"]
        euler_y_deg = result["euler_y_deg"]
        euler_z_deg = result["euler_z_deg"]
        rms_errors = result["rms_error"]
        inliers_series = np.asarray(result["inliers"], dtype=float)
        finite_inliers = inliers_series[np.isfinite(inliers_series)]
        if finite_inliers.size:
            mean_inliers = float(np.mean(finite_inliers))
            std_inliers = float(np.std(finite_inliers, ddof=0))
            print(
                f"{object_id}: inlier count mean={mean_inliers:.2f}, std={std_inliers:.2f} "
                f"over {finite_inliers.size} frames."
            )
        else:
            print(f"{object_id}: no valid inlier counts recorded.")
        
        if not frames:
            print(f"No valid frames for {object_id}; skipping plots.")
            continue
        
        timestamps_mask = np.isfinite(timestamps_series)
        if timestamps_mask.any():
            x_axis = timestamps_series.copy()
            first_ts = float(np.nanmin(timestamps_series[timestamps_mask]))
            if np.isfinite(first_ts):
                x_axis -= first_ts
            x_label = "Time (s)"
        else:
            x_axis = np.asarray(frames, dtype=float)
            x_label = "Frame Index"

        # === Plot 1: 6-Component Pose Plot ===
        fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
        
        translation_arrays = [np.asarray(translations_x, dtype=float),
                            np.asarray(translations_y, dtype=float),
                            np.asarray(translations_z, dtype=float)]
        rotation_arrays = [np.asarray(euler_x_deg, dtype=float),
                        np.asarray(euler_y_deg, dtype=float),
                        np.asarray(euler_z_deg, dtype=float)]
        
        # Compute per-component limits
        valid_trans = [arr[np.isfinite(arr)] for arr in translation_arrays]
        valid_rots = [arr[np.isfinite(arr)] for arr in rotation_arrays]
        
        translation_min = min(arr.min() for arr in valid_trans) if valid_trans and any(len(arr) > 0 for arr in valid_trans) else 0
        translation_max = max(arr.max() for arr in valid_trans) if valid_trans and any(len(arr) > 0 for arr in valid_trans) else 1
        rotation_min = min(arr.min() for arr in valid_rots) if valid_rots and any(len(arr) > 0 for arr in valid_rots) else -3.15
        rotation_max = max(arr.max() for arr in valid_rots) if valid_rots and any(len(arr) > 0 for arr in valid_rots) else 3.15
        
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
                t_margin = np.abs(0.1 * translation_min) if translation_min != 0 else 1
                ax.set_ylim(translation_min - t_margin, translation_max + t_margin)
            elif "Rotation" in label:
                r_margin = np.abs(0.1 * rotation_min) if rotation_min != 0 else 0.5
                ax.set_ylim(rotation_min - r_margin, rotation_max + r_margin * 1.2)
            ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        
        axes[1, 0].set_xlabel(x_label)
        axes[1, 1].set_xlabel(x_label)
        axes[1, 2].set_xlabel(x_label)
        fig.suptitle(f"{object_id}: Translation and Rotation Components Over Time")
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
        plt.show()
    
        # === Plot 2: Reprojection Error ===
        fig_err, ax_err = plt.subplots(figsize=(10, 4))
        ax_err.plot(x_axis, rms_errors, linewidth=1.2, color='red')
        ax_err.set_xlabel(x_label)
        ax_err.set_ylabel("RMS Reprojection Error (pixels)")
        ax_err.set_title(f"{object_id}: Reprojection Error Over Time")
        ax_err.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        plt.show()

    if display:
        for object_id in object_ids:
            frames_npz = frames_npz_paths.get(object_id)
            if frames_npz and frames_npz.exists():
                image_lookup = load_npz_image_lookup(frames_npz, data_key="data", image_key="img")
            else:
                print(f"{object_id}: frames NPZ missing – skipping playback.")
                continue

            run_tracking_from_records(
                tracker=tracker.trackers[object_id],
                records=detections_dict_all[object_id],
                image_lookup=image_lookup,
                display=True,
                axis_length=50.0,
                min_markers=6,
                report_every=1,
                quit_key="q",
                use_pose_prior=True,
            )


if __name__ == "__main__":
    main()


