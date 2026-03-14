from __future__ import annotations
from typing import Optional
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sfm import (
    FilterConfig,
    bootstrap_with_fallback,
    bundle_adjustment_pyceres,
    compute_median_mean_reprojection_error,
    compute_reprojection_errors,
    align_state_to_object_frame,
    incremental_pose_estimation,
    load_calibration,
    load_frame_observations,
    preprocess_detections,
)

# Could be good to have this function back in future for more robust spacing estimation
# def estimate_marker_spacing_from_neighbors(
#     pts: np.ndarray,
#     *,
#     neighbour_scale: float = 1.2,
# ) -> Optional[float]:
#     """
#     Robustly estimate the grid spacing from a cloud of marker points.

#     - first estimate a base spacing from nearest-neighbour distances
#     - then, for each marker, look at neighbours within `neighbour_scale * base_spacing`
#       and average the few closest ones
#     - finally, average those per-marker means.

#     Returns
#     -------
#     spacing_mm : float or None
#         Estimated spacing in the same units as `pts`, or None if it fails.
#     """
#     pts = np.asarray(pts, dtype=float)
#     if pts.shape[0] <= 1:
#         return None

#     tree = cKDTree(pts)
#     nn_dists, _ = tree.query(pts, k=2)
#     base_spacing = float(np.median(nn_dists[:, 1]))
#     if not np.isfinite(base_spacing) or base_spacing <= 0.0:
#         return None

#     cutoff = base_spacing * neighbour_scale

#     per_marker_means: list[float] = []

#     for idx in range(pts.shape[0]):
#         dists, idxs = tree.query(pts[idx], k=min(6, pts.shape[0]))
#         dists = np.atleast_1d(dists)
#         idxs = np.atleast_1d(idxs)

#         mask = idxs != idx
#         neighbour_dists_full = np.asarray(dists[mask], dtype=float)

#         neighbour_dists_full = neighbour_dists_full[neighbour_dists_full <= cutoff]
#         if neighbour_dists_full.size == 0:
#             continue

#         neighbour_dists_full.sort()
#         target_neighbors = min(4, neighbour_dists_full.size)
#         neighbour_dists = neighbour_dists_full[:target_neighbors]

#         per_marker_means.append(float(neighbour_dists.mean()))

#     if not per_marker_means:
#         return None

#     spacing = float(np.mean(per_marker_means))

#     # quick sanity check: simple NN median vs neighbour-based
#     simple_nn_spacing = float(np.median(nn_dists[:, 1]))
#     print(f"Simple NN spacing: {simple_nn_spacing:.6f} mm, "
#         f"Neighbour-based spacing: {spacing:.6f} mm")

#     return spacing


def main() -> None:
    calibration = load_calibration(
        Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_mtx.npy"),
        Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\calib_data\basler_dart\camera_dist.npy"),
    )
    
    detections_path = Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag_probe\mapping\detections_data.npz")
    output_dir = detections_path.parent / "sfm_outputs"
    raw_frames = load_frame_observations(detections_path)
    print(f"Loaded {len(raw_frames)} raw frames")

    config = FilterConfig(min_shared_markers=40, min_bootstrap_inlier_ratio=0.8)
    origin_marker_id = 17  # Set to your chosen ID0; use None to default to centroid.
    x_axis_marker_id = 59       # Set to your chosen ID for x-axis direction; use None to default to PCA.
    y_axis_marker_id = 21  # Set to your chosen ID for y-axis direction; use None to default to PCA.
    cell_size_mm = 8.000     # [mm] - Physical spacing between adjacent cells; set to None to skip rescaling.
    filtered_frames, diagnostics = preprocess_detections(raw_frames, calibration, config=config)
    print(f"Preprocessing diagnostics: {diagnostics}")

    state, tri_result, bootstrap_diag = bootstrap_with_fallback(
        filtered_frames,
        calibration,
        config=config,
    )
    print(
        f"Bootstrap triangulated {len(tri_result.marker_ids)} ids "
        f"using frames {bootstrap_diag['selected_pair']} \n"
        f"Bootstrap survived ids: \n {tri_result.marker_ids}"
    )

    incremental_pose_estimation(
        state,
        checkpoint_path=output_dir / "checkpoint_incremental.pkl",
    )

    errors_before = compute_reprojection_errors(state)
    median_before, mean_before = compute_median_mean_reprojection_error(state)
    print(f"Registered frames after incremental: {sorted(state.poses.keys())}")
    print(f"Markers in map after incremental: {len(state.marker_positions)}")
    print({fid: errs.mean() for fid, errs in errors_before.items() if errs.size})
    print(f"Median reprojection error (pre-BA): {median_before}")
    print(f"Mean reprojection error (pre-BA): {mean_before}")

    print("\nRunning bundle adjustment...", flush=True)
    ba_summary = bundle_adjustment_pyceres(
        state,
        checkpoint_path=output_dir / "checkpoint_ba.pkl",
    )
    print("Ceres bundle adjustment summary:")
    print(f"  message: {str(getattr(ba_summary, 'message', '')).strip()}")
    initial_cost = getattr(ba_summary, "initial_cost", None)
    if initial_cost is not None:
        print(f"  initial cost: {initial_cost:.6e}")
    final_cost = getattr(ba_summary, "final_cost", None)
    if final_cost is not None:
        print(f"  final cost: {final_cost:.6e}")
    iterations = getattr(ba_summary, "iterations", None)
    if iterations is not None:
        print(f"  iterations: {iterations}")

    errors_after = compute_reprojection_errors(state)
    median_after, mean_after = compute_median_mean_reprojection_error(state)
    print({fid: errs.mean() for fid, errs in errors_after.items() if errs.size})
    print(f"Median reprojection error (post-BA): {median_after}")
    print(f"Mean reprojection error (post-BA): {mean_after}")

    # this step is done to change the reference frame of the 
    rotation, origin = align_state_to_object_frame(state, origin_marker_id=origin_marker_id, x_axis_marker_id=x_axis_marker_id, y_axis_marker_id=y_axis_marker_id)

    marker_ids = np.array(sorted(state.marker_positions.keys()), dtype=np.int32)
    marker_points = np.stack([state.marker_positions[mid] for mid in marker_ids], axis=0)

    pose_ids = np.array(sorted(state.poses.keys()), dtype=np.int32)
    pose_rotations = np.stack([state.poses[fid].rotation for fid in pose_ids], axis=0)
    pose_translations = np.stack([state.poses[fid].translation for fid in pose_ids], axis=0)

    scale = 1.0
    if cell_size_mm is not None and marker_points.shape[0] >= 2:
        tree = cKDTree(marker_points)
        dists, _ = tree.query(marker_points, k=2)
        nearest = dists[:, 1]
        finite = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        if finite.size:
            median_spacing = float(np.median(finite))
            scale = cell_size_mm / median_spacing
            marker_points *= scale
            pose_translations *= scale
            origin *= scale
            print(
                f"Rescaled reconstruction: median spacing {median_spacing:.6f} -> "
                f"{cell_size_mm:.6f} mm (scale factor {scale:.6f})."
            )
        else:
            print("Skipping scale application: insufficient valid nearest-neighbour distances.")
    elif cell_size_mm is not None:
        print("Skipping scale application: need at least two markers to estimate spacing.")

    # if cell_size_mm is not None and marker_points.shape[0] >= 2:
    #     spacing = estimate_marker_spacing_from_neighbors(marker_points, neighbour_scale=1.2)

    #     if spacing is not None and spacing > 0.0:
    #         scale = cell_size_mm / spacing
    #         marker_points *= scale
    #         pose_translations *= scale
    #         origin *= scale
    #         print(
    #             f"Rescaled reconstruction (neighbour-based): "
    #             f"{spacing:.6f} -> {cell_size_mm:.6f} mm (scale factor {scale:.6f})."
    #         )
    #     else:
    #         print("Skipping scale application: unable to estimate marker spacing from neighbours.")
    # elif cell_size_mm is not None:
    #     print("Skipping scale application: need at least two markers to estimate spacing.")


    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "marker_map_aligned.npz"
    np.savez(map_path, marker_ids=marker_ids, marker_points=marker_points, rotation=rotation, origin=origin, pose_ids=pose_ids, pose_rotations=pose_rotations, pose_translations=pose_translations)
    print(f"Saved aligned marker map to {map_path}")

    # Extract scaled data for visualization
    marker_positions = marker_points.copy()
    ids = marker_ids.copy()

    camera_centers = np.array([-rotation.T @ translation for rotation, translation in zip(pose_rotations, pose_translations)],dtype=np.float64)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot markers
    ax.scatter(marker_positions[:, 0], marker_positions[:, 1], marker_positions[:, 2], c=ids, cmap="viridis", s=8,label="IDs")

    # Plot cameras
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], color="red", s=5,label="Camera Pose")

    label_step = 3
    for idx, (x, y, z) in enumerate(marker_positions):
        if idx % label_step == 0:
            ax.text(x, y, z, str(ids[idx]), fontsize=6, color="black", va="bottom", ha="center")

    # for idx, (x, y, z) in enumerate(camera_centers):
    #     label = str(pose_ids[idx])
    #     ax.text(x, y, z, label, color="red", fontsize=7)

    # Reference frame at origin marker
    axis_length = float(cell_size_mm * 5.0) if cell_size_mm is not None else 50.0
    if origin_marker_id in marker_ids:
        origin_index = int(np.where(marker_ids == origin_marker_id)[0][0])
        origin_point = marker_positions[origin_index]
    else:
        origin_point = np.zeros(3, dtype=np.float64)
    axes_directions = [
        (np.array([axis_length, 0.0, 0.0], dtype=np.float64), "X", "tab:red"),
        (np.array([0.0, axis_length, 0.0], dtype=np.float64), "Y", "tab:green"),
        (np.array([0.0, 0.0, axis_length], dtype=np.float64), "Z", "tab:blue"),
    ]
    for direction, label, color in axes_directions:
        ax.quiver(origin_point[0], origin_point[1], origin_point[2], direction[0], direction[1], direction[2], color=color, arrow_length_ratio=0.1,linewidth=2.0,label=f"{label}-axis")
        ax.text(*(origin_point + direction),f"{label}",color=color,fontsize=10,weight="bold")

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Marker SFM Reconstruction')
    ax.legend()

    graph_path = output_dir / "3d_marker_map.png"
    fig.savefig(graph_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
