from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial import cKDTree

def inspect_marker_pca(npz_path: str | Path) -> None:
    data = np.load(npz_path)

    # All data in the NPZ is already expressed in the aligned object frame.
    pts_obj = data["marker_points"]          # object-frame marker coords
    R = data["rotation"]                     # rows: object axes in BA world
    origin = data["origin"]                  # BA world origin that was subtracted
    poses_obj = data["pose_translations"]    # object→camera translations after alignment

    # Reconstruct the original BA world coordinates that SVD was run on.
    # world = R.T @ object + origin
    pts_world = (R.T @ pts_obj.T).T + origin
    camera_world = (R.T @ poses_obj.T).T + origin

    centroid = pts_world.mean(axis=0)
    centered = pts_world - centroid
    _, singular_vals, vh = np.linalg.svd(centered, full_matrices=False)

    print("Singular values (spread along PCA axes):", singular_vals)
    print("Principal axes (rows of vh) in BA world frame:")
    print(vh)

    # Flip logic used by align_state_to_object_frame for reference
    z_axis = vh[2].copy()
    cam_centroid = camera_world.mean(axis=0) if len(camera_world) else None
    if cam_centroid is not None and np.dot(z_axis, cam_centroid - centroid) < 0:
        z_axis = -z_axis
    elif cam_centroid is None and z_axis[2] < 0:
        z_axis = -z_axis
    y_axis = np.cross(z_axis, vh[0])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    print("\nReconstructed right-handed basis from SVD (world frame):")
    print("x_axis:", x_axis)
    print("y_axis:", y_axis)
    print("z_axis:", z_axis)

    print("\nAlignment matrix rows (object axes in world frame):")
    print(R)

    # Compare directions: dot products should be ±1 when they agree
    dots = R @ np.stack((x_axis, y_axis, z_axis), axis=1)
    print("\nDot products between stored axes and PCA-based axes:")
    print(np.diag(dots))

    print("\nWorld centroid used for alignment:", centroid)
    print("Stored origin (centroid or chosen marker):", origin)

def print_marker_block(npz_path: str | Path, ids_to_check: list[int]) -> None:
    data = np.load(npz_path)
    pts = data["marker_points"]            # already in the aligned object frame
    marker_ids = data["marker_ids"]

    index_of = {int(mid): idx for idx, mid in enumerate(marker_ids)}
    missing = [mid for mid in ids_to_check if mid not in index_of]
    if missing:
        print(f"Missing markers: {missing}")
        return

    print("Marker coordinates:")
    for mid in ids_to_check:
        coord = pts[index_of[mid]]
        print(f"  {mid:3d}: {coord}")

    ref = ids_to_check[0]
    ref_coord = pts[index_of[ref]]
    print(f"\nDifferences relative to marker {ref}:")
    for mid in ids_to_check[1:]:
        diff = pts[index_of[mid]] - ref_coord
        print(f"  {mid:3d} - {ref:3d}: Δx={diff[0]: .4f}, Δy={diff[1]: .4f}, Δz={diff[2]: .4f}")


def report_marker_neighbors(
    pts: np.ndarray,
    ids: np.ndarray,
    *,
    neighbour_scale: float = 1.2,
) -> None:
    tree = cKDTree(pts)
    if pts.shape[0] <= 1:
        print("Not enough markers to compute neighbours.")
        return

    nn_dists, _ = tree.query(pts, k=2)
    base_spacing = float(np.median(nn_dists[:, 1]))
    if not np.isfinite(base_spacing) or base_spacing <= 0.0:
        print("Unable to determine reference spacing for neighbours.")
        return

    cutoff = base_spacing * neighbour_scale
    type_stats: dict[str, dict[str, list[float]]] = {
        "corner": {"means": [], "stds": []},
        "edge": {"means": [], "stds": []},
        "internal": {"means": [], "stds": []},
    }

    for idx, marker_id in enumerate(ids):
        marker_id_int = int(marker_id)
        dists, idxs = tree.query(pts[idx], k=min(6, len(pts)))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        mask = idxs != idx
        neighbour_dists_full = np.asarray(dists[mask], dtype=np.float64)
        close_mask = neighbour_dists_full <= cutoff
        close_count = int(close_mask.sum())

        if close_count <= 2:
            pos_type = "corner"
            target_neighbors = 2
        elif close_count == 3:
            pos_type = "edge"
            target_neighbors = 3
        else:
            pos_type = "internal"
            target_neighbors = 4

        neighbour_idxs = idxs[mask][:target_neighbors]
        neighbour_dists = neighbour_dists_full[:target_neighbors]

        print(f"Marker {marker_id_int} ({pos_type}):")
        if neighbour_idxs.size == 0:
            print("  (no neighbours found)")
            print()
            continue

        for n_idx, dist in zip(neighbour_idxs, neighbour_dists):
            neighbour_id = int(ids[int(n_idx)])
            print(f"  - Neighbour ID {neighbour_id}: {float(dist):.3f} mm")

        mean_dist = float(neighbour_dists.mean())
        std_dist = float(neighbour_dists.std(ddof=0)) if neighbour_dists.size > 1 else 0.0
        print(f"  mean distance: {mean_dist:.3f} mm")
        print(f"  std distance: {std_dist:.3f} mm")
        type_stats[pos_type]["means"].append(mean_dist)
        type_stats[pos_type]["stds"].append(std_dist)
        print()

    print("Overall neighbourhood statistics by marker class:")
    total_count = 0
    total_mean_sum = 0.0
    total_std_sum = 0.0
    all_means: list[float] = []
    for pos_type, stats in type_stats.items():
        if not stats["means"]:
            print(f"  {pos_type}: no markers")
            continue
        means_arr = np.asarray(stats["means"], dtype=np.float64)
        stds_arr = np.asarray(stats["stds"], dtype=np.float64)
        class_mean = float(means_arr.mean())
        class_std = float(means_arr.std(ddof=0)) if means_arr.size > 1 else 0.0
        class_mean_std = float(stds_arr.mean()) if stds_arr.size else 0.0
        print(f"  {pos_type}:")
        print(f"    mean of neighbour means: {class_mean:.3f} mm")
        print(f"    std of neighbour means: {class_std:.3f} mm")
        print(f"    mean of neighbour stds:  {class_mean_std:.3f} mm")
        total_count += means_arr.size
        total_mean_sum += float(means_arr.sum())
        total_std_sum += float(stds_arr.sum())
        all_means.extend(means_arr.tolist())

    if total_count:
        overall_weighted_mean = total_mean_sum / total_count
        overall_mean_std = total_std_sum / total_count
        overall_mean_dispersion = float(np.std(np.asarray(all_means, dtype=np.float64), ddof=0)) if len(all_means) > 1 else 0.0
        print("\nWeighted summary across all markers:")
        print(f"  weighted mean distance: {overall_weighted_mean:.3f} mm")
        print(f"  weighted std of distances: {overall_mean_dispersion:.3f} mm")
        print(f"  weighted mean of per-marker stds: {overall_mean_std:.3f} mm")


def main() -> None:
    result = np.load(Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\mapping\sfm_outputs_modified\marker_map_aligned.npz"))
    
    #inspect_marker_pca(Path("3x45_3x3tag/npz_marker_results_deodorant_mapping_2/sfm_outputs/marker_map_aligned.npz"))

    #print_marker_block("3x45_3x3tag/npz_marker_results_deodorant_mapping_2/sfm_outputs/marker_map_aligned.npz",[25, 26, 27, 28, 29])

    pts = result["marker_points"]  # Nx3
    ids = result["marker_ids"]
    poses = result["pose_translations"]  # Mx3 camera centres
    rotation = result["rotation"]
    rotations_cam = result["pose_rotations"]
    pose_ids = result["pose_ids"]

    centers_obj = []
    for R, t in zip(rotations_cam, poses):
        centers_obj.append(-(R.T @ t))
    centers_obj = np.vstack(centers_obj)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=ids, cmap=cm.viridis, s=8, label="IDs")
    ax.scatter(centers_obj[:, 0], centers_obj[:, 1], centers_obj[:, 2], color="red", s=25, label="Camera Pose")

    label_step = 1  # Increase to thin labels if the plot becomes cluttered.
    for idx, (x, y, z) in enumerate(pts):
        if idx % label_step == 0:
            ax.text(x, y, z, str(ids[idx]), fontsize=6, color="black", va="bottom", ha="center")

    # for idx, (x, y, z) in enumerate(centers_obj):
    #     label = str(pose_ids[idx])          # or use idx if you prefer 0..N-1
    #     ax.text(x, y, z, label, color="red", fontsize=7)


    # # Draw the reference frame at the origin used during alignment.
    # origin_idx = np.where(ids == 1)[0][0]
    # origin_pt = pts[origin_idx]

    # axis_len = 10.0
    # for vec, colour, label in zip(rotation, ("#d62728", "#2ca02c", "#1f77b4"), "XYZ"):
    #     end = origin_pt + axis_len * vec
    #     ax.plot([origin_pt[0], end[0]], [origin_pt[1], end[1]], [origin_pt[2], end[2]], color=colour, linewidth=2)
    #     ax.text(end[0], end[1], end[2], label, color=colour, fontsize=8, fontweight="bold")

    def marker_distance(id_a: int, id_b: int) -> float:
        idx_a = np.where(ids == id_a)[0][0]
        print(idx_a)
        print(pts[idx_a])
        idx_b = np.where(ids == id_b)[0][0]
        print(id_b)
        print(pts[idx_b])
        return float(np.linalg.norm(pts[idx_a] - pts[idx_b]))

    #print(f"Distance 1001↔1011: {marker_distance(1, 2):.3f} mm")

    print("\nNeighbour summary:\n")
    report_marker_neighbors(pts, ids)


    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Marker 1 Reconstruction")
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
