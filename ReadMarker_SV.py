# Pipeline single frame detection

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
from scipy.spatial import Delaunay, cKDTree

@dataclass
class RefinedPoints:
    points: np.ndarray  # Nx2 (row, col)
    ledges: np.ndarray  # Nx2 (degrees: [white_jump, black_jump])


@dataclass
class CornerDetectionResult:
    points: np.ndarray
    response: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray


@dataclass(frozen=True)
class _RadiusCache:
    r: int
    patch_size: int
    patch_area: int
    shape: Tuple[int, int]
    u: np.ndarray
    v: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    A: np.ndarray
    A3: np.ndarray
    A_pinv: np.ndarray


def _ensure_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32, copy=False)
    return image.astype(np.float32)


@lru_cache(maxsize=None)
def _get_radius_cache(r: int) -> _RadiusCache:
    if r <= 0:
        raise ValueError("radius must be positive")

    axis = np.arange(-r, r + 1, dtype=np.int16)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    ut = u.ravel().astype(np.float64)
    vt = v.ravel().astype(np.float64)
    A = np.column_stack((ut**2, ut * vt, vt**2, np.ones_like(ut)))
    ATA = A.T @ A
    A_pinv = np.linalg.solve(ATA, A.T)

    return _RadiusCache(
        r=r,
        patch_size=2 * r + 1,
        patch_area=A.shape[0],
        shape=u.shape,
        u=u.astype(np.float64),
        v=v.astype(np.float64),
        dx=u.ravel().astype(np.int32),
        dy=v.ravel().astype(np.int32),
        A=A,
        A3=A[:, :3],
        A_pinv=A_pinv,
    )


def _refine_points(
    gx: np.ndarray,
    gy: np.ndarray,
    points: np.ndarray,
    cache: _RadiusCache,
    iterations: int,
) -> np.ndarray:
    height, width = gx.shape
    refined = points.astype(np.float64, copy=True)

    for _ in range(iterations):
        if refined.size == 0:
            break

        iy = np.rint(refined[:, 0]).astype(np.int32)
        ix = np.rint(refined[:, 1]).astype(np.int32)
        inside = (
            (iy >= cache.r)
            & (iy < height - cache.r)
            & (ix >= cache.r)
            & (ix < width - cache.r)
        )
        if not inside.any():
            return np.empty((0, 2), dtype=np.float64)

        iy = iy[inside]
        ix = ix[inside]

        rows_idx = (iy[:, None] + cache.dy[None, :]).astype(np.intp)
        cols_idx = (ix[:, None] + cache.dx[None, :]).astype(np.intp)

        gm = gy[rows_idx, cols_idx]
        gn = gx[rows_idx, cols_idx]

        rows = rows_idx.astype(np.float64)
        cols = cols_idx.astype(np.float64)
        p_vec = rows * gm + cols * gn

        gm2 = gm * gm
        gn2 = gn * gn
        gmg = gm * gn

        g11 = gm2.sum(axis=1)
        g22 = gn2.sum(axis=1)
        g12 = gmg.sum(axis=1)

        rhs1 = (gm * p_vec).sum(axis=1)
        rhs2 = (gn * p_vec).sum(axis=1)

        det = g11 * g22 - g12 * g12
        valid = np.abs(det) > 1.0e-9
        if not valid.any():
            return np.empty((0, 2), dtype=np.float64)

        det = det[valid]
        g11 = g11[valid]
        g22 = g22[valid]
        g12 = g12[valid]
        rhs1 = rhs1[valid]
        rhs2 = rhs2[valid]

        row_sol = (rhs1 * g22 - g12 * rhs2) / det
        col_sol = (g11 * rhs2 - g12 * rhs1) / det
        refined = np.column_stack((row_sol, col_sol))

        mask = (
            (refined[:, 0] < cache.r + 2)
            | (refined[:, 0] > height - cache.r - 3)
            | (refined[:, 1] < cache.r + 2)
            | (refined[:, 1] > width - cache.r - 3)
        )
        refined = refined[~mask]

    return refined


def _poly_features(image: np.ndarray, points: np.ndarray, cache: _RadiusCache):
    n_points = points.shape[0]
    ledges = np.full((n_points, 2), np.nan, dtype=np.float64)
    correlations = np.full(n_points, np.nan, dtype=np.float64)
    ang_bias = np.full(n_points, np.nan, dtype=np.float64)

    if n_points == 0:
        return ledges, correlations, ang_bias

    patch_size = cache.patch_size
    patch_shape = cache.shape

    patches = []
    idx_map = []
    for idx, (row, col) in enumerate(points):
        patch = cv2.getRectSubPix(
            image,
            (patch_size, patch_size),
            (float(col), float(row)),
        )
        if patch is None or patch.shape != patch_shape:
            continue
        patches.append(patch.astype(np.float64, copy=False).reshape(-1))
        idx_map.append(idx)

    if not patches:
        return ledges, correlations, ang_bias

    idx_arr = np.asarray(idx_map, dtype=np.int32)
    patch_matrix = np.stack(patches, axis=0)

    coeffs = patch_matrix @ cache.A_pinv.T
    valid_coeffs = np.all(np.isfinite(coeffs), axis=1)
    if not np.any(valid_coeffs):
        return ledges, correlations, ang_bias

    a = coeffs[:, 2]
    b = coeffs[:, 1]
    c = coeffs[:, 0]

    discriminant = b * b - 4.0 * a * c
    root_mask = (np.abs(a) > 1.0e-12) & (discriminant >= 0.0) & valid_coeffs

    root1 = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    root2 = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    if root_mask.any():
        sqrt_disc = np.sqrt(discriminant[root_mask])
        denom = 2.0 * a[root_mask]
        root1[root_mask] = (-b[root_mask] + sqrt_disc) / denom
        root2[root_mask] = (-b[root_mask] - sqrt_disc) / denom

    theta0 = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    theta1 = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    valid_theta = np.isfinite(root1) & np.isfinite(root2)
    if valid_theta.any():
        theta0[valid_theta] = np.rad2deg(np.arctan(root1[valid_theta]))
        theta1[valid_theta] = np.rad2deg(np.arctan(root2[valid_theta]))

    templates = (cache.A3 @ coeffs[:, :3].T).T
    np.sign(templates, out=templates)
    templates[templates == 0.0] = 1.0

    templates_centered = templates - templates.mean(axis=1, keepdims=True)
    patches_centered = patch_matrix - patch_matrix.mean(axis=1, keepdims=True)

    ss_templates = np.einsum("ij,ij->i", templates_centered, templates_centered)
    ss_patches = np.einsum("ij,ij->i", patches_centered, patches_centered)
    corr_mask = (ss_templates > 1.0e-12) & (ss_patches > 1.0e-12)

    corr_values = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    if corr_mask.any():
        cov = np.einsum("ij,ij->i", templates_centered, patches_centered)
        corr_values[corr_mask] = cov[corr_mask] / np.sqrt(
            ss_templates[corr_mask] * ss_patches[corr_mask]
        )

    bias_values = np.full(coeffs.shape[0], np.nan, dtype=np.float64)
    ledge_values = np.full((coeffs.shape[0], 2), np.nan, dtype=np.float64)
    valid = root_mask & corr_mask & valid_theta
    if valid.any():
        theta0_v = theta0[valid]
        theta1_v = theta1[valid]
        angle_diff = np.rad2deg(
            np.abs(
                _angdiff(
                    np.deg2rad(theta0_v),
                    np.deg2rad(theta1_v),
                )
            )
        )
        bias_values[valid] = np.abs(angle_diff - 90.0)

        sign_k = np.sign(theta0_v * theta1_v * c[valid])
        max_angles = np.maximum(theta0_v, theta1_v)
        min_angles = np.minimum(theta0_v, theta1_v)
        ledge_values[valid] = np.column_stack(
            (
                np.where(sign_k >= 0.0, max_angles, min_angles),
                np.where(sign_k >= 0.0, min_angles, max_angles),
            )
        )

    correlations[idx_arr] = corr_values
    ang_bias[idx_arr] = bias_values
    ledges[idx_arr] = ledge_values

    return ledges, correlations, ang_bias


# def _cluster_points(points: np.ndarray, ledges: np.ndarray, threshold: float = 1.0):
#     n = points.shape[0]
#     if n == 0:
#         return points, ledges

#     parent = np.arange(n, dtype=np.int32)

#     def find(x: int) -> int:
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a: int, b: int) -> None:
#         ra = find(a)
#         rb = find(b)
#         if ra != rb:
#             parent[rb] = ra

#     cell_size = max(threshold, 1.0)
#     cells: Dict[Tuple[int, int], List[int]] = {}
#     scaled = points / cell_size
#     coords = np.floor(scaled).astype(np.int32)

#     for idx, (ci, cj) in enumerate(coords):
#         cells.setdefault((ci, cj), []).append(idx)

#     neighbor_offsets = [
#         (-1, -1),
#         (-1, 0),
#         (-1, 1),
#         (0, -1),
#         (0, 0),
#         (0, 1),
#         (1, -1),
#         (1, 0),
#         (1, 1),
#     ]
#     for (ci, cj), indices in cells.items():
#         for di, dj in neighbor_offsets:
#             neigh_key = (ci + di, cj + dj)
#             neigh_indices = cells.get(neigh_key)
#             if neigh_indices is None:
#                 continue
#             for i in indices:
#                 for j in neigh_indices:
#                     if neigh_key == (ci, cj) and j <= i:
#                         continue
#                     if (
#                         max(
#                             abs(points[i, 0] - points[j, 0]),
#                             abs(points[i, 1] - points[j, 1]),
#                         )
#                         <= threshold
#                     ):
#                         union(i, j)

#     clusters: Dict[int, List[int]] = {}
#     for idx in range(n):
#         root = find(idx)
#         clusters.setdefault(root, []).append(idx)

#     merged_points = []
#     merged_ledges = []
#     for indices in clusters.values():
#         merged_points.append(points[indices].mean(axis=0))
#         merged_ledges.append(ledges[indices].mean(axis=0))

#     return np.vstack(merged_points), np.vstack(merged_ledges)


def pre_filter(
    image: np.ndarray,
    r: int = 5,
    expect_n: int = 100,
    sigma: float = 3.0,
    response_threshold: float = 0.1,
) -> CornerDetectionResult:
    if image.ndim != 2:
        raise ValueError("pre_filter expects a single-channel (grayscale) image")

    img = _ensure_float32(image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)

    magnitude = cv2.magnitude(gx, gy)
    if sigma > 0.0:
        gpow = cv2.GaussianBlur(magnitude, (0, 0), sigma)
        gx_blur = cv2.GaussianBlur(gx, (0, 0), sigma)
        gy_blur = cv2.GaussianBlur(gy, (0, 0), sigma)
        gsum = cv2.magnitude(gx_blur, gy_blur)
    else:
        gpow = magnitude
        gsum = magnitude

    response = cv2.subtract(gpow, gsum)
    kernel = np.ones((3, 3), dtype=np.float32)
    dilated = cv2.dilate(response, kernel, borderType=cv2.BORDER_REFLECT101)
    response[dilated != response] = 0.0
    np.maximum(response, 0.0, out=response)
    if response_threshold > 0.0:
        response[response < response_threshold] = 0.0

    response[:r, :] = 0.0
    response[-r:, :] = 0.0
    response[:, :r] = 0.0
    response[:, -r:] = 0.0

    valid = response > 0.0
    if not np.any(valid):
        empty = np.empty((0, 2), dtype=np.float64)
        return CornerDetectionResult(
            points=empty,
            response=response,
            grad_x=gx.astype(np.float64, copy=False),
            grad_y=gy.astype(np.float64, copy=False),
        )

    values = response[valid]
    top_k = int(min(expect_n, values.size))
    if top_k <= 0:
        top_k = values.size
    if top_k == values.size:
        mask = valid
    else:
        kth = values.size - top_k
        threshold = float(np.partition(values, kth)[kth])
        mask = response >= threshold

    rows, cols = np.nonzero(mask)
    points = np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))

    return CornerDetectionResult(
        points=points,
        response=response,
        grad_x=gx.astype(np.float64, copy=False),
        grad_y=gy.astype(np.float64, copy=False),
    )


def pt_refine(
    image: np.ndarray,
    candidates: np.ndarray,
    r: int,
    iterations: int = 2,
    *,
    grad_x: np.ndarray | None = None,
    grad_y: np.ndarray | None = None,
) -> RefinedPoints:
    if candidates.size == 0:
        return RefinedPoints(points=np.empty((0, 2)), ledges=np.empty((0, 2)))

    cache = _get_radius_cache(int(r))

    if grad_x is None or grad_y is None:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    else:
        gx = np.array(grad_x, dtype=np.float64, copy=False)
        gy = np.array(grad_y, dtype=np.float64, copy=False)
        if gx.shape != image.shape or gy.shape != image.shape:
            raise ValueError("Gradient arrays must match image dimensions")

    refined = _refine_points(gx, gy, candidates, cache, iterations)
    if refined.size == 0:
        return RefinedPoints(points=np.empty((0, 2)), ledges=np.empty((0, 2)))

    ledges, correlations, ang_bias = _poly_features(image, refined, cache)

    valid_mask = (
        ~np.isnan(ledges).any(axis=1)
        & np.isfinite(correlations)
        & np.isfinite(ang_bias)
        & (ang_bias <= 20.0)
    )
    if valid_mask.any():
        corr_slice = correlations[valid_mask]
        corr_threshold = corr_slice.max() - 0.2
        valid_mask &= correlations >= corr_threshold

    refined = refined[valid_mask]
    ledges = ledges[valid_mask]

    if refined.shape[0] < 2:
        return RefinedPoints(points=refined, ledges=ledges)

    if refined.shape[0] >= 2:
        tree = linkage(refined, method='average', metric='chebyshev')
        labels = fcluster(tree, t=2.0, criterion='distance')
        
        merged_points = []
        merged_ledges = []
        for label in np.unique(labels):
            mask = labels == label
            merged_points.append(refined[mask].mean(axis=0))
            merged_ledges.append(ledges[mask].mean(axis=0))
        
        merged_points = np.vstack(merged_points)
        merged_ledges = np.vstack(merged_ledges)
    else:
        merged_points = refined
        merged_ledges = ledges

    return RefinedPoints(points=merged_points, ledges=merged_ledges)

def _angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    if a_flat.std() == 0 or b_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


# def pre_filter(image: np.ndarray, r: int = 5, expect_n: int = 100, sigma: float = 3.0) -> np.ndarray:
#     if image.ndim != 2:
#         raise ValueError("pre_filter expects a single-channel (grayscale) image")
    
#     # Use fast OpenCV for gradients and blur
#     gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
#     gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
#     gpow = cv2.GaussianBlur(np.hypot(gx, gy), (0,0), sigma)
#     gx_blur = cv2.GaussianBlur(gx, (0,0), sigma)
#     gy_blur = cv2.GaussianBlur(gy, (0,0), sigma)
#     gsum = np.hypot(gx_blur, gy_blur)
#     response = gpow - gsum

#     # Use OpenCV dilate instead of scipy maximum_filter
#     dilated = cv2.dilate(response, np.ones((3,3),np.float64), borderType=cv2.BORDER_REFLECT101)
#     response[dilated != response] = 0.0

#     response[:r, :] = 0.0
#     response[-r:, :] = 0.0
#     response[:, :r] = 0.0
#     response[:, -r:] = 0.0

#     valid = response >= 0.1
#     if not np.any(valid):
#         return np.empty((0, 2), dtype=np.float64)

#     # Vectorized top-k selection
#     values = response[valid]
#     top_k = min(expect_n, values.size)
#     idx = values.size - top_k
#     threshold = np.partition(values, idx)[idx]
#     rows, cols = np.nonzero(response >= threshold)
#     return np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))


# def pt_refine(image: np.ndarray, candidates: np.ndarray, r: int, iterations: int = 2) -> RefinedPoints:
#     if candidates.size == 0:
#         return RefinedPoints(points=np.empty((0, 2)), ledges=np.empty((0, 2)))

#     gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
#     gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)

#     points = candidates.astype(np.float64).copy()
#     height, width = image.shape

#     for _ in range(iterations):
#         for idx, (row_f, col_f) in enumerate(points):
#             row = int(round(row_f))
#             col = int(round(col_f))
#             if row - r < 0 or row + r >= height or col - r < 0 or col + r >= width:
#                 continue

#             rows, cols = np.mgrid[row - r : row + r + 1, col - r : col + r + 1]
#             gm = gy[row - r : row + r + 1, col - r : col + r + 1]
#             gn = gx[row - r : row + r + 1, col - r : col + r + 1]

#             g = np.column_stack((gm.ravel(), gn.ravel()))
#             mn = np.column_stack((rows.ravel(), cols.ravel()))
#             p_vec = np.sum(mn * g, axis=1)

#             try:
#                 sol, *_ = np.linalg.lstsq(g, p_vec, rcond=None)
#             except np.linalg.LinAlgError:
#                 continue

#             points[idx] = sol

#         mask = (
#             (points[:, 0] < r + 2)
#             | (points[:, 0] > height - r - 3)
#             | (points[:, 1] < r + 2)
#             | (points[:, 1] > width - r - 3)
#         )
#         points = points[~mask]
#         if points.size == 0:
#             return RefinedPoints(points=np.empty((0, 2)), ledges=np.empty((0, 2)))

#     u, v = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
#     ut = u.ravel()
#     vt = v.ravel()
#     A = np.column_stack((ut**2, ut * vt, vt**2, np.ones_like(ut)))

#     ledges = np.zeros((points.shape[0], 2))
#     correlations = np.zeros(points.shape[0])
#     ang_bias = np.zeros(points.shape[0])

#     grid_y = np.arange(height)
#     grid_x = np.arange(width)
#     interpolator = RegularGridInterpolator((grid_y, grid_x), image, bounds_error=False, fill_value=np.nan)

#     for idx, (row, col) in enumerate(points):
#         iy = int(round(row))
#         ix = int(round(col))

#         y_window = slice(iy - r - 1, iy + r + 2)
#         x_window = slice(ix - r - 1, ix + r + 2)
#         if (
#             y_window.start < 0
#             or y_window.stop > height
#             or x_window.start < 0
#             or x_window.stop > width
#         ):
#             ledges[idx, :] = np.nan
#             continue

#         sample_points = np.column_stack(((row + v).ravel(), (col + u).ravel()))
#         samples = interpolator(sample_points).reshape(u.shape)
#         if np.isnan(samples).any():
#             ledges[idx, :] = np.nan
#             continue

#         try:
#             c, *_ = np.linalg.lstsq(A, samples.ravel(), rcond=None)
#         except np.linalg.LinAlgError:
#             ledges[idx, :] = np.nan
#             continue

#         poly = np.array([c[2], c[1], c[0]])
#         roots = np.roots(poly)
#         if not np.all(np.isreal(roots)) or roots.size != 2:
#             ledges[idx, :] = np.nan
#             continue

#         theta = np.rad2deg(np.arctan(roots.real))
#         template = np.sign(A[:, :3] @ c[:3])
#         template[template == 0] = 1
#         correlations[idx] = _corr2(template.reshape(samples.shape), samples)

#         angle_diff = np.rad2deg(np.abs(_angdiff(np.deg2rad(theta[0]), np.deg2rad(theta[1]))))
#         ang_bias[idx] = abs(angle_diff - 90.0)

#         sign_k = np.sign(theta[0] * theta[1] * c[0])
#         if sign_k >= 0:
#             ledges[idx, :] = [max(theta), min(theta)]
#         else:
#             ledges[idx, :] = [min(theta), max(theta)]

#     valid_mask = (
#         ~np.isnan(ledges).any(axis=1)
#         & (correlations >= (correlations.max() - 0.2))
#         & (ang_bias <= 20.0)
#     )
#     points = points[valid_mask]
#     ledges = ledges[valid_mask]

#     if points.shape[0] < 2:
#         return RefinedPoints(points=points, ledges=ledges)

#     tree = linkage(points, method="average", metric="chebyshev")
#     cluster_labels = fcluster(tree, t=2.0, criterion="distance")

#     merged_points = []
#     merged_ledge = []
#     for label in np.unique(cluster_labels):
#         mask = cluster_labels == label
#         merged_points.append(points[mask].mean(axis=0))
#         merged_ledge.append(ledges[mask].mean(axis=0))
#     print(len(merged_points), "corners after merging clusters")
#     return RefinedPoints(points=np.vstack(merged_points), ledges=np.vstack(merged_ledge))


def _filter_sparse_clusters(
    refined: RefinedPoints,
    *,
    min_cluster_points: int,
    radius_scale: float = 3.5,
) -> RefinedPoints:
    """
    Suppress refined points that belong to small, isolated clusters.

    We build a neighbourhood graph using a cKDTree and keep only clusters whose
    cardinality exceeds `min_cluster_points`. If every cluster is small we leave
    the input untouched.
    """
    points = refined.points
    ledges = refined.ledges
    count = points.shape[0]
    if count == 0 or count < max(3, min_cluster_points):
        return refined

    k_neigh = min(count - 1, 6)
    if k_neigh <= 0:
        return refined

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k_neigh + 1)
    # Column 0 is distance to itself (0); grab the first non-zero neighbour.
    nn_dists = dists[:, 1]
    nn_dists = nn_dists[np.isfinite(nn_dists) & (nn_dists > 0)]
    if nn_dists.size == 0:
        return refined

    radius = float(np.median(nn_dists) * radius_scale)
    if not np.isfinite(radius) or radius <= 0:
        return refined

    adjacency = tree.query_ball_point(points, radius)
    visited = np.zeros(count, dtype=bool)
    clusters: List[List[int]] = []

    for idx in range(count):
        if visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        clusters.append(component)

    large_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_points]
    if not large_clusters:
        return refined

    keep_mask = np.zeros(count, dtype=bool)
    for cluster in large_clusters:
        keep_mask[cluster] = True

    if keep_mask.sum() == count:
        return refined

    return RefinedPoints(points=points[keep_mask], ledges=ledges[keep_mask])

def _filter_irregular_edges(
    refined: RefinedPoints,
    grids: List[np.ndarray],
    *,
    angle_tol_deg: float = 10.0,
    length_tol: float = 0.3,
    min_edges: int = 2,
) -> RefinedPoints:
    """
    Remove points with irregular edge patterns in detected grids.
    
    Validates grid topology by checking edge lengths and angles against
    median values. Points with too many irregular edges are filtered out.
    """
    points = refined.points
    ledges = refined.ledges
    total_points = points.shape[0]
    
    if total_points == 0:
        return refined

    keep_mask = np.ones(total_points, dtype=bool)
    angle_tol = float(np.deg2rad(angle_tol_deg))

    def _principal_angle(angles: np.ndarray) -> float:
        """Compute principal angle from a set of angles using circular mean."""
        if angles.size == 0:
            return 0.0
        vectors = np.exp(1j * 2.0 * angles)
        mean = np.mean(vectors)
        if mean == 0:
            return 0.0
        return 0.5 * float(np.angle(mean))

    # Process each grid
    for grid in grids:
        if grid.size == 0:
            continue

        rows, cols = grid.shape
        valid_mask = ~np.isnan(grid)
        
        if np.count_nonzero(valid_mask) < max(4, min_edges + 1):
            continue

        # Collect horizontal and vertical edge statistics
        horiz_lengths: List[float] = []
        horiz_angles: List[float] = []
        vert_lengths: List[float] = []
        vert_angles: List[float] = []

        for r in range(rows):
            for c in range(cols):
                idx_val = grid[r, c]
                if np.isnan(idx_val):
                    continue
                idx = int(idx_val)
                
                # Horizontal edge (right neighbor)
                if c + 1 < cols and not np.isnan(grid[r, c + 1]):
                    neighbour = int(grid[r, c + 1])
                    vec = points[neighbour] - points[idx]
                    length = float(np.linalg.norm(vec))
                    if length > 0.0:
                        horiz_lengths.append(length)
                        horiz_angles.append(float(np.arctan2(vec[0], vec[1])))
                
                # Vertical edge (bottom neighbor)
                if r + 1 < rows and not np.isnan(grid[r + 1, c]):
                    neighbour = int(grid[r + 1, c])
                    vec = points[neighbour] - points[idx]
                    length = float(np.linalg.norm(vec))
                    if length > 0.0:
                        vert_lengths.append(length)
                        vert_angles.append(float(np.arctan2(vec[0], vec[1])))

        if len(horiz_lengths) < 4 or len(vert_lengths) < 4:
            continue

        # Calculate reference values (median length and principal angle)
        med_h_len = float(np.median(horiz_lengths))
        med_v_len = float(np.median(vert_lengths))
        
        if med_h_len <= 0.0 or med_v_len <= 0.0:
            continue

        ref_h_ang = _principal_angle(np.asarray(horiz_angles))
        ref_v_ang = _principal_angle(np.asarray(vert_angles))

        # Evaluate each point's edges
        for r in range(rows):
            for c in range(cols):
                idx_val = grid[r, c]
                if np.isnan(idx_val):
                    continue
                idx = int(idx_val)
                bad_edges = 0
                total_edges = 0

                def _evaluate(neigh_r: int, neigh_c: int, ref_len: float, ref_ang: float) -> None:
                    """Check if an edge to a neighbor is irregular."""
                    nonlocal bad_edges, total_edges
                    
                    if neigh_r < 0 or neigh_r >= rows or neigh_c < 0 or neigh_c >= cols:
                        return
                    
                    neigh_val = grid[neigh_r, neigh_c]
                    if np.isnan(neigh_val):
                        return
                    
                    neighbour_idx = int(neigh_val)
                    vec = points[neighbour_idx] - points[idx]
                    length = float(np.linalg.norm(vec))
                    
                    if length <= 0.0:
                        return
                    
                    angle = float(np.arctan2(vec[0], vec[1]))
                    total_edges += 1
                    
                    # Check length and angle deviation
                    length_dev = abs(length - ref_len) / ref_len
                    angle_dev = min(
                        abs(_angdiff(angle, ref_ang)),
                        abs(_angdiff(angle, ref_ang + np.pi)),
                    )
                    
                    if length_dev > length_tol or angle_dev > angle_tol:
                        bad_edges += 1

                # Evaluate all 4 neighbors
                _evaluate(r, c + 1, med_h_len, ref_h_ang)  # Right
                _evaluate(r, c - 1, med_h_len, ref_h_ang)  # Left
                _evaluate(r + 1, c, med_v_len, ref_v_ang)  # Bottom
                _evaluate(r - 1, c, med_v_len, ref_v_ang)  # Top

                # Mark point for removal if too many bad edges
                if total_edges >= min_edges and bad_edges >= min_edges:
                    keep_mask[idx] = False

    if keep_mask.all():
        return refined

    return RefinedPoints(points=points[keep_mask], ledges=ledges[keep_mask])


def pt_struct(points: np.ndarray, ledges: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    if points.shape[0] < 3:
        return [], np.empty((0, 2), dtype=np.int32)

    pts_xy = points[:, ::-1]
    tri = Delaunay(pts_xy)

    edge_set = {tuple(sorted(pair)) for simplex in tri.simplices for pair in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2]))}
    edges = np.array(sorted(edge_set), dtype=np.int32)
    if edges.size == 0:
        return [], np.empty((0, 2), dtype=np.int32)

    remove_mask = np.zeros(edges.shape[0], dtype=bool)
    min_rad = np.deg2rad(30.0)
    for idx, (ia, ib) in enumerate(edges):
        ledge_a = np.deg2rad(ledges[ia])
        ledge_b = np.deg2rad(ledges[ib])
        vec = points[ib] - points[ia]
        dir_a = np.arctan2(vec[0], vec[1])
        dir_b = np.arctan2(-vec[0], -vec[1])

        ledge_diff = min(
            abs(_angdiff(ledge_a[0], ledge_b[1])),
            abs(_angdiff(ledge_a[1], ledge_b[0])),
        )

        ledge_a_diff = min(
            abs(_angdiff(dir_a, ledge_a[0])),
            abs(_angdiff(dir_a, ledge_a[0] + np.pi)),
            abs(_angdiff(dir_a, ledge_a[1])),
            abs(_angdiff(dir_a, ledge_a[1] + np.pi)),
        )

        ledge_b_diff = min(
            abs(_angdiff(dir_b, ledge_b[0])),
            abs(_angdiff(dir_b, ledge_b[0] + np.pi)),
            abs(_angdiff(dir_b, ledge_b[1])),
            abs(_angdiff(dir_b, ledge_b[1] + np.pi)),
        )

        if ledge_diff > min_rad or ledge_a_diff > min_rad or ledge_b_diff > min_rad:
            remove_mask[idx] = True

    edges = edges[~remove_mask]

    if edges.size == 0:
        return [], np.empty((0, 2), dtype=np.int32)

    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    order = np.argsort(lengths)
    edges = edges[order]
    lengths = lengths[order]

    shortest = np.full(edges.shape[0], np.inf)
    keep = np.ones(edges.shape[0], dtype=bool)
    for idx, (length, (ia, ib)) in enumerate(zip(lengths, edges)):
        if length > 1.7 * shortest[idx]: # this needs refinement because shortest is not the best
            keep[idx] = False
            continue
        mask = (
            (edges[:, 0] == ia)
            | (edges[:, 1] == ia)
            | (edges[:, 0] == ib)
            | (edges[:, 1] == ib)
        )
        shortest[mask] = np.minimum(shortest[mask], length)
    edges = edges[keep]
    if edges.size == 0:
        return [], np.empty((0, 2), dtype=np.int32)

    edges_bi = np.vstack((edges, edges[:, ::-1]))
    sort_idx = np.lexsort((edges_bi[:, 1], edges_bi[:, 0]))
    edges_bi = edges_bi[sort_idx]

    compass = np.sort(ledges, axis=1)
    compass = np.concatenate((compass, compass + 180.0), axis=1)

    matrix_label = np.zeros((points.shape[0], 3), dtype=np.int32)
    shift = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)

    queue: List[int] = []
    matrix_num = 0
    if edges_bi.size:
        queue.append(int(edges_bi[0, 0]))
        matrix_num = 1
        matrix_label[queue[0]] = np.array([matrix_num, 0, 0], dtype=np.int32)

    finished = False
    while not finished:
        if queue:
            qu_a = queue.pop(0)
            neighbours = [int(nb) for nb in edges_bi[edges_bi[:, 0] == qu_a, 1] if matrix_label[nb, 0] == 0]
            if not neighbours:
                continue

            queue.extend(neighbours)
            for nb in neighbours:
                vec = points[nb] - points[qu_a]
                dir_a = float(np.arctan2(vec[0], vec[1]))
                dir_b = float(np.arctan2(-vec[0], -vec[1]))
                comp_a = np.deg2rad(compass[qu_a])
                comp_b = np.deg2rad(compass[nb])

                include_a = np.abs(_angdiff(dir_a, comp_a))
                include_b = np.abs(_angdiff(dir_b, comp_b))

                matched_dir_a = int(np.argmin(include_a))
                matrix_label[nb, 0] = matrix_label[qu_a, 0]
                matrix_label[nb, 1:] = matrix_label[qu_a, 1:] + shift[matched_dir_a]

                expected = (matched_dir_a + 2) % 4
                matched_dir_b = int(np.argmin(include_b))
                compass[nb] = np.roll(compass[nb], expected - matched_dir_b)
        else:
            unset = np.where(matrix_label[:, 0] == 0)[0]
            if unset.size == 0:
                finished = True
                continue
            queue.append(int(unset[0]))
            matrix_num += 1
            matrix_label[queue[0]] = np.array([matrix_num, 0, 0], dtype=np.int32)

    if matrix_num == 0:
        return [], edges

    index_column = np.arange(points.shape[0], dtype=np.int32)[:, None]
    matrix_label_full = np.hstack((matrix_label, index_column))

    grids: List[np.ndarray] = []
    for num in range(1, matrix_num + 1):
        label = matrix_label_full[matrix_label_full[:, 0] == num]
        if label.size == 0:
            grids.append(np.empty((0, 0)))
            continue

        rows = label[:, 1] - label[:, 1].min()
        cols = label[:, 2] - label[:, 2].min()
        grid = np.full((rows.max() + 1, cols.max() + 1), np.nan, dtype=float)
        grid[rows, cols] = label[:, 3]
        grids.append(grid)

    return grids, edges


def _match_map(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Work in float space to avoid clip/overflow warnings when writing sentinel values.
    A_map = A.astype(np.float32, copy=True)
    A_map[np.isnan(A_map)] = 0
    A_map[A_map == 0] = -1

    B_map = B.astype(np.float32, copy=True)
    B_map[np.isnan(B_map)] = 0
    B_map[B_map == 0] = -1

    return convolve2d(np.rot90(A_map, 2), B_map, mode="full")


def pt_identify(
    image: np.ndarray,
    sta: np.ndarray,
    grids: Sequence[np.ndarray],
    points: np.ndarray,
    contrast_threshold: float = 10,
    id_base: int = 1,
    column_major: bool = True,
    return_debug: bool = False,
) -> np.ndarray:
    function_start = perf_counter()

    if len(grids) == 0:
        if points.size == 0:
            result = np.empty((0, 3))
        else:
            ids = np.full(points.shape[0], np.nan)
            result = np.column_stack((points, ids))
        if return_debug:
            empty_centers = np.empty((0, 2), dtype=np.float64)
            elapsed_single = perf_counter() - function_start
            debug_payload: dict[str, object] = {
                "grids": [],
                "dot_tables": [],
                "low_confidence_indices": [],
                "match_metrics": [],
                "ambiguous": False,
                "ambiguous_details": [],
                "dot_centers_yes": empty_centers,
                "dot_centers_no": empty_centers,
                "timings": {
                    "dot_detection": 0.0,
                    "identification": elapsed_single,
                    "total": elapsed_single,
                },
            }
            return result, debug_payload
        return result

    sta_m, sta_n = sta.shape
    expected_points = (sta_m + 1) * (sta_n + 1)
    min_target_ratio = 0.15
    min_target_coverage = max(1, int(round(min_target_ratio * expected_points)))
    order = "F" if column_major else "C"
    id_matrix = np.arange(id_base, id_base + (sta_m + 1) * (sta_n + 1), dtype=float).reshape(sta_m + 1, sta_n + 1, order=order)
    min_valid_id = float(np.nanmin(id_matrix))
    max_valid_id = float(np.nanmax(id_matrix))

    height, width = image.shape
    interpolator = RegularGridInterpolator(
        (np.arange(height), np.arange(width)),
        image,
        bounds_error=False,
        fill_value=np.nan,
    )

    def _sample(coords: np.ndarray) -> np.ndarray:
        values = interpolator(coords)
        if np.isnan(values).any():
            rr = np.clip(np.rint(coords[:, 0]).astype(int), 0, height - 1)
            cc = np.clip(np.rint(coords[:, 1]).astype(int), 0, width - 1)
            fallback = image[rr, cc]
            values = np.where(np.isnan(values), fallback, values)
        return values

    dot_tables: List[np.ndarray | None] = []
    pt_arrays: List[np.ndarray | None] = []
    low_confidence: set[int] = set()
    match_metrics: List[dict[str, object]] = []
    ambiguous = False
    ambiguous_details: List[Dict[str, object]] = []
    yes_dot_centers: List[np.ndarray] = []
    no_dot_centers: List[np.ndarray] = []
    dot_detection_start = perf_counter()

    for grid in grids:
        M, N = grid.shape
        if M < 2 or N < 2:
            dot_tables.append(None)
            pt_arrays.append(None)
            continue

        dot_table = np.zeros((M - 1, N - 1), dtype=float)
        pt_array = np.full((M, N, 2), np.nan, dtype=float)
        valid = ~np.isnan(grid)
        pt_array[valid] = points[grid[valid].astype(int)]
        pt_arrays.append(pt_array)

        for im in range(M - 1):
            for jn in range(N - 1):
                block = grid[im : im + 2, jn : jn + 2]
                if np.isnan(block).any():
                    dot_table[im, jn] = np.nan
                    continue

                corners = pt_array[im : im + 2, jn : jn + 2].reshape(-1, 2)
                dot_center = corners.mean(axis=0)

                # block = grid[im:im+2, jn:jn+2]
                # mask = ~np.isnan(block)
                # if mask.sum() < 3:
                #     dot_table[im, jn] = np.nan
                #     continue

                # # Infer missing corner if exactly one is NaN
                # corners = pt_array[im:im+2, jn:jn+2].reshape(-1, 2)
                # if mask.sum() == 3:
                #     coords = np.argwhere(mask)
                #     all_corners = {(0, 0), (0, 1), (1, 0), (1, 1)}
                #     present = {tuple(c) for c in coords}
                #     missing = all_corners.difference(present).pop()  # this is (row, col)

                #     # simple surrogate: average the two adjacent known corners
                #     if missing == (0,0):
                #         guess = (corners[1] + corners[2]) / 2
                #     elif missing == (0,1):
                #         guess = (corners[0] + corners[3]) / 2
                #     elif missing == (1,0):
                #         guess = (corners[0] + corners[3]) / 2
                #     else:  # (1,1)
                #         guess = (corners[1] + corners[2]) / 2
                #     corners[missing[0]*2 + missing[1]] = guess

                # dot_center = corners.mean(axis=0)

                sample_background = corners * 0.8 + dot_center * 0.2
                sample_foreground = corners * 0.2 + dot_center * 0.8

                bg_vals = _sample(sample_background)
                fg_vals = _sample(sample_foreground)

                has_contrast = (
                    not np.isnan(bg_vals).any()
                    and not np.isnan(fg_vals).any()
                    and abs(np.nanmean(bg_vals) - np.nanmean(fg_vals)) > contrast_threshold
                )
                if has_contrast:
                    dot_table[im, jn] = 1
                    yes_dot_centers.append(dot_center.astype(np.float64))
                else:
                    dot_table[im, jn] = 0
                    no_dot_centers.append(dot_center.astype(np.float64))

        dot_tables.append(dot_table)
    
    dot_detection_end = perf_counter()
    id_list = np.full(points.shape[0], np.nan)
    for grid_idx, (grid, dots) in enumerate(zip(grids, dot_tables)):
        if dots is None or dots.size == 0:
            continue

        target_coverage = int(np.count_nonzero(~np.isnan(grid)))
        if target_coverage < min_target_coverage:
            ambiguous = True
            id_list[:] = np.nan
            indices_sparse = grid[~np.isnan(grid)].astype(int)
            low_confidence.update(int(idx) for idx in indices_sparse.tolist())
            ambiguous_details.append(
                {
                    "grid_index": int(grid_idx),
                    "reason": "sparse_grid",
                    "target_coverage": int(target_coverage),
                    "min_required": int(min_target_coverage),
                }
            )
            break
        best_array: np.ndarray | None = None
        best_score = -np.inf
        best_coverage = -1

        for rot in range(4):
            rotated_dots = np.rot90(dots, rot)
            match_map = _match_map(rotated_dots, sta)
            order_flat = np.argsort(match_map, axis=None)[::-1]
            rotated_arr = np.rot90(grid, rot)
            arr_m, arr_n = rotated_arr.shape

            for flat_idx in order_flat:
                score = float(match_map.flat[flat_idx])
                if score <= 0 and best_coverage >= target_coverage:
                    break

                match_m = int(flat_idx // match_map.shape[1]) + 1
                match_n = int(flat_idx % match_map.shape[1]) + 1

                row_start = max(match_m - arr_m + 2, 1)
                row_end = min(match_m + 1, sta_m + 1)
                col_start = max(match_n - arr_n + 2, 1)
                col_end = min(match_n + 1, sta_n + 1)
                if row_start > row_end or col_start > col_end:
                    continue

                matched = id_matrix[row_start - 1 : row_end, col_start - 1 : col_end]

                array_indices = np.full(rotated_arr.shape, np.nan)
                arr_row_start = max(arr_m - match_m, 1)
                arr_row_end = min(sta_m + arr_m - match_m, arr_m)
                arr_col_start = max(arr_n - match_n, 1)
                arr_col_end = min(sta_n + arr_n - match_n, arr_n)

                array_indices[
                    arr_row_start - 1 : arr_row_end,
                    arr_col_start - 1 : arr_col_end,
                ] = matched

                array_ind = np.rot90(array_indices, -rot)
                coverage = int(np.count_nonzero(~np.isnan(array_ind)))

                if coverage == 0:
                    continue

                if coverage > best_coverage or (coverage == best_coverage and score > best_score):
                    best_array = array_ind
                    best_coverage = coverage
                    best_score = score

                if coverage >= target_coverage:
                    break

        if best_array is None:
            continue

        best_sorted = None
        arr_sorted = None
        best_score_align = -np.inf
        for rot in range(4):
            cand = np.rot90(best_array, rot)
            arr_cand = np.rot90(grid, rot)
            for flipped in (False, True):
                cand_cur = np.fliplr(cand) if flipped else cand

                arr_cur = np.fliplr(arr_cand) if flipped else arr_cand
                mask = ~np.isnan(cand_cur)

                if mask.any():
                    rows = min(cand_cur.shape[0], id_matrix.shape[0])
                    cols = min(cand_cur.shape[1], id_matrix.shape[1])
                    cand_trim = cand_cur[:rows, :cols]
                    mask_trim = mask[:rows, :cols]
                    id_view = id_matrix[:rows, :cols]
                    score_align = np.count_nonzero(np.isclose(cand_trim[mask_trim], id_view[mask_trim]))
                else:
                    score_align = 0

                if score_align > best_score_align:
                    best_score_align = score_align
                    best_sorted = cand_cur
                    arr_sorted = arr_cur

        if best_sorted is None or arr_sorted is None:
            continue
        
        assigned_map = np.full(arr_sorted.shape, np.nan)
        used_ids: set[int] = set()
        for r_idx in range(arr_sorted.shape[0]):
            for c_idx in range(arr_sorted.shape[1]):
                point_idx = arr_sorted[r_idx, c_idx]
                if np.isnan(point_idx):
                    continue
                point_idx = int(point_idx)
                value = best_sorted[r_idx, c_idx]
                if np.isnan(value):
                    continue
                id_list[point_idx] = value
                assigned_map[r_idx, c_idx] = value
                used_ids.add(int(round(value)))

        known_rows: list[list[float]] = []
        known_ids: list[float] = []
        for r_idx in range(arr_sorted.shape[0]):
            for c_idx in range(arr_sorted.shape[1]):
                assigned_val = assigned_map[r_idx, c_idx]
                if np.isnan(assigned_val):
                    continue
                known_rows.append([float(r_idx), float(c_idx), 1.0])
                known_ids.append(float(assigned_val))

        if len(known_rows) >= 3:
            A = np.asarray(known_rows, dtype=float)
            b = np.asarray(known_ids, dtype=float)
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
            row_coeff, col_coeff, offset = coeffs

            for r_idx in range(arr_sorted.shape[0]):
                for c_idx in range(arr_sorted.shape[1]):
                    point_idx = arr_sorted[r_idx, c_idx]
                    if np.isnan(point_idx):
                        continue
                    point_idx = int(point_idx)
                    if not np.isnan(id_list[point_idx]):
                        continue

                    predicted = row_coeff * r_idx + col_coeff * c_idx + offset
                    predicted_round = int(round(predicted))

                    if predicted_round < min_valid_id or predicted_round > max_valid_id:
                        continue
                    if predicted_round in used_ids:
                        continue
                    if abs(predicted - predicted_round) > 0.35:
                        continue

                    neighbour_ids: list[float] = []
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r_idx + dr
                            cc = c_idx + dc
                            if rr < 0 or rr >= arr_sorted.shape[0] or cc < 0 or cc >= arr_sorted.shape[1]:
                                continue
                            neighbour_val = assigned_map[rr, cc]
                            if np.isnan(neighbour_val):
                                continue
                            neighbour_ids.append(float(neighbour_val))

                    if len(neighbour_ids) < 2:
                        continue

                    id_list[point_idx] = float(predicted_round)
                    assigned_map[r_idx, c_idx] = float(predicted_round)
                    used_ids.add(predicted_round)

    result = np.column_stack((points, id_list))
    identification_end = perf_counter()
    dot_detection_time = float(max(dot_detection_end - dot_detection_start, 0.0))
    identification_time = float(max(identification_end - dot_detection_end, 0.0))
    total_time = float(max(identification_end - function_start, 0.0))
    if return_debug:
        dot_yes = np.asarray(yes_dot_centers, dtype=np.float64) if yes_dot_centers else np.empty((0, 2), dtype=np.float64)
        dot_no = np.asarray(no_dot_centers, dtype=np.float64) if no_dot_centers else np.empty((0, 2), dtype=np.float64)
        debug_payload: dict[str, object] = {
            "grids": list(grids),
            "dot_tables": dot_tables,
            "low_confidence_indices": sorted(low_confidence),
            "match_metrics": match_metrics,
            "ambiguous": ambiguous,
            "ambiguous_details": ambiguous_details,
            "dot_centers_yes": dot_yes,
            "dot_centers_no": dot_no,
            "timings": {
                "dot_detection": dot_detection_time,
                "identification": identification_time,
                "total": total_time,
            },
        }
        return result, debug_payload
    return result


def read_marker(
    image: np.ndarray,
    sta: np.ndarray,
    r: int = 5,
    expect_n: int = 1500,
    sigma: float = 3.0,
    contrast_threshold: float = 20,
    id_base: int = 1,
    column_major_ids: bool = True,
    *,
    return_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict[str, object]]:

    t_start = perf_counter()
    detection = pre_filter(image, r=r, expect_n=expect_n, sigma=sigma)
    candidates = detection.points
    t_after_pre = perf_counter()
    refined = pt_refine(image, candidates, r=r, grad_x=detection.grad_x, grad_y=detection.grad_y)
    t_after_refine = perf_counter()

    expected_points = (sta.shape[0] + 1) * (sta.shape[1] + 1)
    min_cluster_points = max(10, int(round(0.08 * expected_points)))
    refined = _filter_sparse_clusters(refined, min_cluster_points=min_cluster_points)
    t_after_cluster = perf_counter()
    points = np.rot90(refined.points, k=1)
    grids, edges = pt_struct(refined.points, refined.ledges)
    t_after_struct = perf_counter()

    #refined = _filter_irregular_edges(refined,grids, angle_tol_deg=15.0,length_tol=0.3, min_edges=2)
    
    pt_result = pt_identify(image,sta,grids,refined.points,contrast_threshold=contrast_threshold,id_base=id_base,column_major=column_major_ids,return_debug=return_debug)
    t_after_identify = perf_counter()

    if return_debug:
        pt_list, debug = pt_result
        timings = dict(debug.get("timings", {}))
        timings["corner_detection"] = float(max(t_after_pre - t_start, 0.0))
        timings["refinement"] = float(max(t_after_refine - t_after_pre, 0.0))
        timings["cluster_filter"] = float(max(t_after_cluster - t_after_refine, 0.0))
        timings["struct"] = float(max(t_after_struct - t_after_cluster, 0.0))
        timings["total"] = float(max(t_after_identify - t_start, 0.0))
        debug["timings"] = timings
        debug["refined_points"] = np.asarray(refined.points, dtype=np.float64)
        debug["refined_points_rotated"] = np.asarray(points, dtype=np.float64)
        return pt_list, edges, debug
    return pt_result, edges

def _draw_overlay(
    image: np.ndarray,
    pt_list: np.ndarray,
    edges: np.ndarray,
    *,
    font_scale: float | None = None,
    font_thickness: int | None = None,
    dot_debug: dict[str, object] | None = None,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("draw_overlay expects a single-channel (grayscale) image")

    if image.dtype == np.uint8:
        gray = image
    else:
        max_val = float(np.nanmax(image))
        if max_val <= 1.0:
            gray = np.clip(image, 0.0, 1.0) * 255.0
        else:
            gray = np.clip(image, 0.0, 255.0)
        gray = gray.astype(np.uint8)

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    scale_factor = max(image.shape) / 720.0
    print('Scale factor:', scale_factor)
    if font_scale is None:
        font_scale = max(0.6, 0.2 * scale_factor)
    if font_thickness is None:
        font_thickness = min(2, int(round(scale_factor)))
    marker_radius = max(3, int(round(1 * scale_factor)))
    cross_size = max(3, int(round(8 * scale_factor)))

    ambiguous = bool(dot_debug.get("ambiguous", False)) if dot_debug else False
    edge_color = (180, 120, 0) if not ambiguous else (0, 165, 255)
    dot_color = (180, 120, 0) if not ambiguous else (0, 165, 255)
    text_color = (0, 165, 255) if not ambiguous else (0, 165, 255)
    text_outline_color = (0, 0, 0)
    text_outline_thickness = max(font_thickness + 2, 2)

    for edge in edges:
        a, b = edge.astype(int)
        pa = (int(round(pt_list[a, 1])), int(round(pt_list[a, 0])))
        pb = (int(round(pt_list[b, 1])), int(round(pt_list[b, 0])))
        cv2.line(display, pa, pb, edge_color, 1, cv2.LINE_AA)

    low_conf_set: set[int] = set()
    if dot_debug:
        low_conf_indices = dot_debug.get("low_confidence_indices", [])
        if low_conf_indices is not None:
            try:
                low_conf_set = {int(idx) for idx in np.asarray(low_conf_indices).ravel()}
            except Exception:
                low_conf_set = {int(idx) for idx in low_conf_indices}

    for idx, (row, col, marker_id) in enumerate(pt_list):
        center = (int(round(col)), int(round(row)))
        if idx in low_conf_set:
            cv2.drawMarker(display, center, (0, 165, 255), cv2.MARKER_TILTED_CROSS, cross_size, 2)
        elif np.isnan(marker_id):
            cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, cross_size, 2)
        else:
            cv2.circle(display, center, marker_radius, dot_color, thickness=-1)
            cv2.putText(
                display,
                str(int(marker_id)),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_outline_color,
                text_outline_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                str(int(marker_id)),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
    if dot_debug:
        grids = dot_debug.get("grids", [])
        dot_tables = dot_debug.get("dot_tables", [])
        cell_radius = max(1, int(round(1 * scale_factor)))
        cell_cross = max(1, int(round(1 * scale_factor)))

        for grid, dot_table in zip(grids, dot_tables):
            if dot_table is None:
                continue

            for im in range(dot_table.shape[0]):
                for jn in range(dot_table.shape[1]):
                    value = dot_table[im, jn]
                    if np.isnan(value):
                        color = (0, 165, 255)  # orange for missing data
                    elif value >= 0.5:
                        color = (255, 255, 0)  # cian for detected dot
                    else:
                        color = (0, 0, 255)  # red for rejected dot

                    block = grid[im : im + 2, jn : jn + 2]
                    if np.isnan(block).any():
                        continue

                    point_indices = block.astype(int).ravel()
                    coords = pt_list[point_indices, :2]
                    center_row, center_col = coords.mean(axis=0)
                    center = (int(round(center_col)), int(round(center_row)))

                    if np.isnan(value):
                        cv2.drawMarker(display, center, color, cv2.MARKER_CROSS, cell_cross, 2)
                    elif value >= 0.5:
                        cv2.circle(display, center, cell_radius, color, thickness=-10)
                    else:
                        cv2.circle(display, center, cell_radius, color, thickness=-10)

    return display

# # Load the grayscale board image and the sta pattern generated by GenerateMarker_SV.py.
# img = cv2.imread("marker_board_rotated90CLOCK.png", cv2.IMREAD_GRAYSCALE)
# if img is None:
#     raise FileNotFoundError("marker_board.png not found")
# sta = np.load("marker_sta.npy")
# print("Loaded marker pattern array: ", sta)

# # this scaling part is here but it's not used. Autoscaling is done in read_marker
# height, width = img.shape[:2]
# long_side = max(height, width)
# print(f"Image size: {width}×{height}px, longest side = {long_side}px")
# if long_side < 720:
#     print("Longer edge is below 720px.")
# else:
#     print("Longer edge is 720px or more.")
# scale = 720.0 / max(img.shape)
# new_width = int(round(img.shape[1] * scale))
# img_resized = cv2.resize(img, (new_width, 720))
# img_resized = img_resized.astype(np.float64) / 255.0

# # Compute suggested detection parameters: fixed r = 5, sigma = 3.0, expect_n based on sta size.
# r = 5
# expect_n = int(20 * (sta.shape[0] + 1) * (sta.shape[1] + 1))
# sigma = 3.0
# pt_list, edges, debug_info = read_marker(img, sta, r, expect_n, sigma, return_debug=True)
# print(f"Detected {pt_list.shape[0]} points and identified {int(np.count_nonzero(~np.isnan(pt_list[:, 2])))} points.")

# np.savez("dot_centers_output.npz",yes_dot=debug_info["dot_centers_yes"],no_dot=debug_info["dot_centers_no"])
# np.savez("corners_output.npz", refined_points=debug_info["refined_points"],refined_points_rotated=debug_info["refined_points_rotated"])

# overlay = _draw_overlay(img, pt_list, edges, dot_debug=debug_info)
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# SAVE_PATH = Path("hydra_read_overlay.png")
# cv2.imwrite(str(SAVE_PATH), overlay)
# print(f"Overlay saved to {SAVE_PATH!s}")
