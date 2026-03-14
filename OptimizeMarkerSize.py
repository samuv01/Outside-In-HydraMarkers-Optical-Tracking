#!/usr/bin/env python3
"""
Marker parameter search & optimization (with image-area and printability checks)

Features:
- Detectability: min window in pixels (blur kernel vs detection patch)
- Optics mapping: convert min pixels -> required cell/feature size (mm)
- Printer limit: skip combos where feature < min_print_mm
- Image fraction: projected marker area <= max_image_fraction of the image
- Working range: honor closest/farthest distances when mapping to pixels
- Patch sizing: require (2r+1) to span a fraction of the projected cell
- Capacity rule (per user): square (n+1)^2; rect (gx+1)*(gy+1)
- Objectives: max_ids | max_robustness | balanced
- Geometry: square | rect | both
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil, floor
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ----------------------------
# Core computations / helpers
# ----------------------------
def compute_min_pixels(sigma: float, r: int) -> int:
    """Compute min window in pixels for detectability."""
    kernel_size = 2 * ceil(3 * sigma) + 1
    patch_size = 2 * r + 1
    return max(kernel_size, patch_size)


def required_feature_size_mm(min_pixels: int,
                             pixel_size_um: float,
                             working_distance_mm: float,
                             focal_length_mm: float) -> float:
    """
    Convert min pixels needed to a required physical cell/feature size (mm).
    feature_size_mm = (min_pixels * pixel_size_mm * working_distance_mm) / focal_length_mm
    """
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be > 0")
    pixel_size_mm = pixel_size_um / 1000.0
    return (min_pixels * pixel_size_mm * working_distance_mm) / focal_length_mm


def grid_counts(marker_w_mm: float, marker_h_mm: float, feature_mm: float) -> Tuple[int, int]:
    """How many whole features fit in each direction (integer packing)."""
    if feature_mm <= 0:
        return 0, 0
    gx = max(0, floor(marker_w_mm / feature_mm))
    gy = max(0, floor(marker_h_mm / feature_mm))
    return gx, gy


def actual_cell_pitch_mm(marker_w_mm: float, marker_h_mm: float, gx: int, gy: int, geometry: str) -> float:
    """Resulting (actual) cell pitch in mm for chosen grid."""
    if gx <= 0 or gy <= 0:
        return 0.0
    if geometry == "square":
        n = min(gx, gy)
        return min(marker_w_mm / n, marker_h_mm / n)
    else:
        return min(marker_w_mm / gx, marker_h_mm / gy)


def score_candidate(ids: int, min_pixels: int, objective: str, 
                   ids_min: float = 0.0, ids_max: float = 1.0,
                   px_min: float = 0.0, px_max: float = 1.0) -> float:
    """
    Bigger score = better; ties prefer the secondary term.
    
    For balanced objective: uses L2 compromise programming (Euclidean distance to utopia).
    Requires normalization bounds (ids_min, ids_max, px_min, px_max).
    """
    if objective == "max_ids":
        return ids * 1e9 + min_pixels
    elif objective == "max_robustness":
        return min_pixels * 1e9 + ids
    else:  # "balanced"
        # Normalize objectives to [0, 1]
        eps = 1e-12
        ids_norm = (ids - ids_min) / max(ids_max - ids_min, eps)
        px_norm = (min_pixels - px_min) / max(px_max - px_min, eps)
        
        # Clamp to [0, 1] range
        ids_norm = max(0.0, min(1.0, ids_norm))
        px_norm = max(0.0, min(1.0, px_norm))
        
        # Euclidean distance from utopia point (1, 1)
        # Utopia = best possible for both objectives
        distance = ((1.0 - ids_norm)**2 + (1.0 - px_norm)**2)**0.5
        
        # Return negative distance (closer to utopia = higher score)
        return -distance




def projected_marker_size_px(marker_w_mm: float, marker_h_mm: float,
                             focal_length_mm: float, working_distance_mm: float,
                             pixel_size_um: float) -> Tuple[float, float]:
    """
    Thin-lens orthogonal projection (small-angle approximation):
        width_px  = (f * S_marker) / (D * p)
        height_px = (f * S_marker) / (D * p)
    where p is pixel size in mm/pixel.
    """
    pixel_size_mm = pixel_size_um / 1000.0
    width_px = (focal_length_mm * marker_w_mm) / (working_distance_mm * pixel_size_mm)
    height_px = (focal_length_mm * marker_h_mm) / (working_distance_mm * pixel_size_mm)
    return width_px, height_px


# ----------------------------
# Inputs / search container
# ----------------------------
@dataclass
class SearchInputs:
    # Marker / optics
    marker_width_mm: float = 30.0
    marker_height_mm: float = 30.0
    pixel_size_um: float = 3.45
    working_distance_mm: float = 100.0
    working_distance_min_mm: float = 100.0
    working_distance_max_mm: float = 100.0
    focal_length_mm: float = 4.0

    # Image / sensor
    image_width_px: int = 1920     # Basler daA1920-160uc default
    image_height_px: int = 1080
    max_image_fraction: float = 0.25  # marker area <= this fraction of image area

    # Targets & ranges
    min_ids_required: int = 50
    sigma_min: float = 0.6
    sigma_max: float = 3.0
    sigma_step: float = 0.1
    r_min: int = 1
    r_max: int = 12

    # Strategy
    objective: str = "max_ids"     # "max_ids" | "max_robustness" | "balanced"
    grid_geometry: str = "both"    # "square" | "rect" | "both"
    balance_w: float = 0.5  # 0..1, higher = favor IDs, lower = favor robustness
    min_patch_fraction: float = 0.105  # fraction of projected cell width that (2r+1) must cover

    # Fabrication
    min_print_mm: float = 0.06     # minimum printable feature size (mm)


# ----------------------------
# Main search
# ----------------------------
def run_search(params: SearchInputs) -> Dict[str, Any]:
    # --- Global image-area constraint (same for all candidates with fixed optics/size) ---
    min_dist = float(params.working_distance_min_mm)
    max_dist = float(params.working_distance_max_mm)
    if min_dist <= 0.0 or max_dist <= 0.0:
        raise ValueError("working distances must be > 0")
    if min_dist > max_dist:
        raise ValueError("working_distance_min_mm cannot exceed working_distance_max_mm")

    pixel_size_mm = params.pixel_size_um / 1000.0

    w_px, h_px = projected_marker_size_px(
        params.marker_width_mm, params.marker_height_mm,
        params.focal_length_mm, min_dist, params.pixel_size_um
    )
    marker_area_px = w_px * h_px
    image_area_px = params.image_width_px * params.image_height_px
    frac = marker_area_px / image_area_px if image_area_px > 0 else 1.0

    if frac > params.max_image_fraction:
        raise ValueError(
            f"Marker projects to {marker_area_px:.0f} px^2 ({frac*100:.1f}% of image), "
            f"exceeding max allowed {params.max_image_fraction*100:.1f}%. "
            "Reduce marker size, increase working distance, or adjust focal length."
        )

    # If area constraint passed, proceed with sigma/r sweep
    sigmas = np.round(np.arange(params.sigma_min, params.sigma_max + 1e-12, params.sigma_step), 6)
    rs = list(range(params.r_min, params.r_max + 1))

    rows: List[Dict[str, Any]] = []
    filtered_printability = 0  # count configs below printer limit

    for sigma in sigmas:
        for r in rs:
            min_pix = compute_min_pixels(float(sigma), int(r))
            feat_mm = required_feature_size_mm(
                min_pix,
                params.pixel_size_um,
                max_dist,
                params.focal_length_mm,
            )

            # --- Printer resolution check ---
            if feat_mm < params.min_print_mm:
                filtered_printability += 1
                continue

            gx, gy = grid_counts(params.marker_width_mm, params.marker_height_mm, feat_mm)

            # Square candidate (n+1 rule)
            n_sq = max(0, min(gx, gy))
            ids_sq = (n_sq + 1) * (n_sq + 1) if n_sq > 0 else 0
            cell_sq = actual_cell_pitch_mm(params.marker_width_mm, params.marker_height_mm, gx, gy, "square") if n_sq > 0 else 0.0

            # Rect candidate (n+1 rule generalized)
            ids_rect = (gx + 1) * (gy + 1) if gx > 0 and gy > 0 else 0
            cell_rect = actual_cell_pitch_mm(params.marker_width_mm, params.marker_height_mm, gx, gy, "rect") if gx > 0 and gy > 0 else 0.0

            # Record per requested geometry (and meeting min IDs)
            min_patch_pixels = 2 * int(r) + 1

            allow_square = (
                params.grid_geometry in ("square", "both")
                and ids_sq >= params.min_ids_required
                and n_sq > 0
            )
            if allow_square and cell_sq > 0.0:
                cell_sq_px_min = (params.focal_length_mm * cell_sq) / (min_dist * pixel_size_mm)
                if params.min_patch_fraction > 0.0 and cell_sq_px_min > 0.0:
                    required_patch = params.min_patch_fraction * cell_sq_px_min
                    if min_patch_pixels < required_patch:
                        allow_square = False
            if allow_square:
                rows.append(dict(
                    sigma=float(sigma), r=int(r), min_pixels=int(min_pix),
                    req_cell_mm=float(feat_mm),
                    grid_x=int(n_sq), grid_y=int(n_sq),
                    ids=int(ids_sq), geometry="square",
                    actual_cell_mm=float(cell_sq),
                    marker_area_frac=float(frac),
                ))
            allow_rect = (
                params.grid_geometry in ("rect", "both")
                and ids_rect >= params.min_ids_required
                and gx > 0 and gy > 0
            )
            if allow_rect and cell_rect > 0.0:
                cell_rect_px_min = (params.focal_length_mm * cell_rect) / (min_dist * pixel_size_mm)
                if params.min_patch_fraction > 0.0 and cell_rect_px_min > 0.0:
                    required_patch = params.min_patch_fraction * cell_rect_px_min
                    if min_patch_pixels < required_patch:
                        allow_rect = False
            if allow_rect:
                rows.append(dict(
                    sigma=float(sigma), r=int(r), min_pixels=int(min_pix),
                    req_cell_mm=float(feat_mm),
                    grid_x=int(gx), grid_y=int(gy),
                    ids=int(ids_rect), geometry="rect",
                    actual_cell_mm=float(cell_rect),
                    marker_area_frac=float(frac),
                ))

    if not rows:
        msg = "No feasible configurations."
        if filtered_printability > 0:
            msg += f" {filtered_printability} configuration(s) skipped for printer limit "\
                   f"(feature_size_mm < {params.min_print_mm} mm)."
        raise ValueError(msg)

    # Rank by objective
    ids_key = "ids"
    if any(row.get("payload_ids", 0) > 0 for row in rows):
        ids_key = "payload_ids"

    # Compute global normalization bounds for balanced scoring
    eps = 1e-12
    ids_max = max((float(row.get(ids_key, 0)) for row in rows), default=0.0)
    px_max = max((float(row["min_pixels"]) for row in rows), default=0.0)
    ids_min = min((float(row.get(ids_key, 0)) for row in rows), default=0.0)
    px_min = min((float(row["min_pixels"]) for row in rows), default=0.0)

    for row in rows:
        row["ids_norm"] = (row["ids"] - ids_min) / max(ids_max - ids_min, eps)
        row["px_norm"] = (row["min_pixels"] - px_min) / max(px_max - px_min, eps)

    # Score all candidates
    for row in rows:
        row["score"] = score_candidate(
            int(row["ids"]), 
            int(row["min_pixels"]), 
            params.objective,
            ids_min=ids_min,
            ids_max=ids_max,
            px_min=px_min,
            px_max=px_max
        )


    rows.sort(key=lambda rw: rw["score"], reverse=True)
    best = rows[0]

    df = pd.DataFrame(rows).reset_index(drop=True) if _HAS_PANDAS else None

    return {
        "best": best,
        "rows": rows,
        "df": df,
        "skipped_printability": filtered_printability
    }

# ----------------------------
# Output helpers
# ----------------------------
def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if _HAS_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        import csv
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

# ----------------------------
# CLI / or just press Run
# ----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize 2D marker grid under imaging, printing, and image-area constraints")
    
    # ALIX MARKER
    # Marker / optics
    # on probe
    # p.add_argument("--marker-width", type=float, default=120.0) # 1st design
    # p.add_argument("--marker-height", type=float, default=130.0) # 1st design
    # p.add_argument("--marker-width", type=float, default=80.0) # in mm
    # p.add_argument("--marker-height", type=float, default=50.0) # in mm
    # on baby
    p.add_argument("--marker-width", type=float, default=60.0) # in mm
    p.add_argument("--marker-height", type=float, default=75.0) # in mm
    p.add_argument("--pixel-size-um", type=float, default=3.45)
    p.add_argument("--working-distance", type=float, default=300.0)
    p.add_argument("--working-distance-min", type=float, default=250.0, help="Closest working distance (mm) to cover; defaults to --working-distance.")
    p.add_argument("--working-distance-max", type=float, default=350.0, help="Farthest working distance (mm) to cover; defaults to --working-distance.")
    p.add_argument("--focal-length", type=float, default=4.0)
    # Image / sensor
    p.add_argument("--image-width-px", type=int, default=1920)
    p.add_argument("--image-height-px", type=int, default=1080)
    p.add_argument("--max-image-fraction", type=float, default=0.25)
    # Targets & ranges
    p.add_argument("--min-ids", type=int, default=50) # on probe
    # p.add_argument("--min-ids", type=int, default=10) # on baby
    p.add_argument("--sigma-min", type=float, default=4)
    p.add_argument("--sigma-max", type=float, default=10.0)
    p.add_argument("--sigma-step", type=float, default=0.1)
    p.add_argument("--r-min", type=int, default=4)
    p.add_argument("--r-max", type=int, default=12)
    # Strategy
    p.add_argument("--objective", choices=["max_ids", "max_robustness", "balanced"], default="max_robustness")
    p.add_argument("--geometry", choices=["square", "rect", "both"], default="both")
    p.add_argument("--min-patch-frac", type=float, default=0.15, help="Minimum fraction of the projected cell width that (2r+1) must span at the closest distance")
    
    # # EMILY-LIAM MARKER
    # # Marker / optics
    # p.add_argument("--marker-width", type=float, default=20.0) # in mm
    # p.add_argument("--marker-height", type=float, default=20.0) # in mm
    # p.add_argument("--pixel-size-um", type=float, default=2.75)
    # p.add_argument("--working-distance", type=float, default=1000.0)
    # p.add_argument("--working-distance-min", type=float, default=975.0, help="Closest working distance (mm) to cover; defaults to --working-distance.")
    # p.add_argument("--working-distance-max", type=float, default=1025.0, help="Farthest working distance (mm) to cover; defaults to --working-distance.")
    # p.add_argument("--focal-length", type=float, default=16.0)
    # # Image / sensor
    # p.add_argument("--image-width-px", type=int, default=2400)
    # p.add_argument("--image-height-px", type=int, default=2000)
    # p.add_argument("--max-image-fraction", type=float, default=0.20)
    # # Targets & ranges
    # p.add_argument("--min-ids", type=int, default=10)
    # p.add_argument("--sigma-min", type=float, default=4)
    # p.add_argument("--sigma-max", type=float, default=10.0)
    # p.add_argument("--sigma-step", type=float, default=0.1)
    # p.add_argument("--r-min", type=int, default=4)
    # p.add_argument("--r-max", type=int, default=12)
    # # Strategy
    # p.add_argument("--objective", choices=["max_ids", "max_robustness", "balanced"], default="balanced")
    # p.add_argument("--geometry", choices=["square", "rect", "both"], default="both")
    # p.add_argument("--min-patch-frac", type=float, default=0.15, help="Minimum fraction of the projected cell width that (2r+1) must span at the closest distance")


    # Fabrication
    p.add_argument("--min-print-mm", type=float, default=0.06)
    p.add_argument("--printer-dpi", type=int, default=1200, help="Target printer resolution (DPI) for cell_px computation")

    # Outputs
    p.add_argument("--out-dir", type=str, default="OptimizationResults")
    p.add_argument("--csv-out", type=str, default="marker_search_results.csv")
    p.add_argument("--png-out", type=str, default="plot.png")
    p.add_argument("--no-plot", action="store_true", help="Disable heatmap plot even if matplotlib is available")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 15})

    wd_min = args.working_distance_min if args.working_distance_min is not None else args.working_distance
    wd_max = args.working_distance_max if args.working_distance_max is not None else args.working_distance
    if wd_min <= 0.0 or wd_max <= 0.0:
        raise ValueError("Working distances must be positive.")
    if wd_min > wd_max:
        raise ValueError("--working-distance-min cannot exceed --working-distance-max.")

    params = SearchInputs(
        # Marker / optics
        marker_width_mm=args.marker_width,
        marker_height_mm=args.marker_height,
        pixel_size_um=args.pixel_size_um,
        working_distance_mm=args.working_distance,
        working_distance_min_mm=wd_min,
        working_distance_max_mm=wd_max,
        focal_length_mm=args.focal_length,
        # Image / sensor
        image_width_px=args.image_width_px,
        image_height_px=args.image_height_px,
        max_image_fraction=args.max_image_fraction,
        # Targets & ranges
        min_ids_required=args.min_ids,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max, sigma_step=args.sigma_step,
        r_min=args.r_min, r_max=args.r_max,
        # Strategy
        objective=args.objective,
        grid_geometry=args.geometry,
        min_patch_fraction=args.min_patch_frac,
        # Fabrication
        min_print_mm=args.min_print_mm
    )

    result = run_search(params)
    best = result["best"]
    rows = result["rows"]
    skipped_printability = result['skipped_printability']

    # Get the best score among all rows
    best_score = max(row['score'] for row in rows)

    # Get all candidates sharing the best score
    ties = [row for row in rows if abs(row['score'] - best_score) < 1e-10]
    print(f"\nNumber of candidates sharing the best score: {len(ties)}")
    # for i, row in enumerate(ties):
    #     print(f"Candidate {i+1}:")
    #     print(row)

    cell_px = ceil(best["actual_cell_mm"] * args.printer_dpi / 25.4)

    # Save outputs
    if args.csv_out:
        csv_path = out_dir / Path(args.csv_out).name
        save_csv(rows, str(csv_path))
        print(f"[+] Saved candidates to {csv_path}")

    best_sigma_values = [row['sigma'] for row in ties]
    best_r_values = [row['r'] for row in ties]

    print("=== BEST CONFIGURATIONS ===")
    print(f"Objective         : {params.objective}")
    print(f"Geometry          : {ties[0]['geometry']}")
    if abs(wd_min - wd_max) < 1e-6:
        print(f"Working distance  : {wd_min:.1f} mm")
    else:
        print(f"Working distance  : {wd_min:.1f} - {wd_max:.1f} mm")
    if min(best_sigma_values) == max(best_sigma_values):
        print(f"sigma value       : {min(best_sigma_values):.2f}")
    else:
        print(f"sigma range       : {min(best_sigma_values):.2f} to {max(best_sigma_values):.2f}")

    if min(best_r_values) == max(best_r_values):
        print(f"r value           : {min(best_r_values)}")
    else:
        print(f"r range           : {min(best_r_values)} to {max(best_r_values)}")
    print(f"min_pixels        : {ties[0]['min_pixels']}")
    print(f"required cell (mm): {ties[0]['req_cell_mm']:.4f}")
    print(f"grid_x x grid_y   : {ties[0]['grid_x']} x {ties[0]['grid_y']}")
    print(f"actual cell (mm)  : {ties[0]['actual_cell_mm']:.4f}")
    print(f"max IDs (n+1 rule): {ties[0]['ids']}")
    print(f"marker area frac  : {100*ties[0]['marker_area_frac']:.2f}% of image")
    print(f"cell_px           : {cell_px}")
    print("==========================")

    print(f"(info) Skipped {skipped_printability} configuration(s) for printer limit: "
      f"feature_size_mm < {params.min_print_mm} mm\n")

    if params.objective == 'balanced':
        base_png = out_dir / Path(args.png_out).name
        base_stem = base_png.stem
        base_suffix = base_png.suffix or ".png"

        x = [row["ids_norm"] for row in rows]
        y = [row["px_norm"] for row in rows]
        plt.figure(figsize=(8,8))
        plt.grid(True, color='gray', alpha=0.3, linewidth=0.7)
        plt.scatter(x, y, s=40, label="Candidates", color=(0.0, 0.4, 1.0, 0.03))
        plt.xlabel("Normalized IDs")
        plt.ylabel("Normalized Robustness")
        plt.scatter([1],[1], c='red', marker='*', s=200, label="Utopia")
        plt.scatter([best["ids_norm"]], [best["px_norm"]], c='red', s=120, label="Balanced Solution", zorder=5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{base_stem}_objective_space{base_suffix}", dpi=300, bbox_inches="tight")

        # For IDs:
        ids_counts = Counter(x)
        ids_values = np.array(list(ids_counts.keys()))
        ids_freq   = np.array(list(ids_counts.values()))

        plt.figure(figsize=(8,5))
        plt.grid(True, color='gray', alpha=0.5, linewidth=0.7)
        plt.bar(ids_values, ids_freq, width=0.005, color='dodgerblue', edgecolor='black')
        plt.xlabel('Normalized IDs')
        plt.ylabel('Number of points sharing value')
        plt.tight_layout()
        plt.savefig(out_dir / f"{base_stem}_ids_distribution{base_suffix}", dpi=300, bbox_inches="tight")

        # For Robustness:
        px_counts = Counter(y)
        px_values = np.array(list(px_counts.keys()))
        px_freq   = np.array(list(px_counts.values()))

        plt.figure(figsize=(8,5))
        plt.grid(True, color='gray', alpha=0.5, linewidth=0.7)
        plt.bar(px_values, px_freq, width=0.005, color='orange', edgecolor='black')
        plt.xlabel('Normalized Robustness')
        plt.ylabel('Number of points sharing value')
        plt.tight_layout()
        plt.savefig(out_dir / f"{base_stem}_robustness_distribution{base_suffix}", dpi=300, bbox_inches="tight")

        pairs = list(zip(x, y))
        counter = Counter(pairs)

        # Get sorted unique values for axes
        ids_bins = sorted(set(x))
        px_bins = sorted(set(y))

        # Create a grid for the bar locations
        X_idx, Y_idx = np.meshgrid(np.arange(len(ids_bins)), np.arange(len(px_bins)), indexing='ij')
        bar_x = X_idx.flatten()
        bar_y = Y_idx.flatten()
        bar_z = np.zeros_like(bar_x)

        # Fill heights (counts)
        bin_lookup = {(i,j):0 for i,j in zip(bar_x, bar_y)}
        for (id_val, px_val), cnt in counter.items():
            id_i = ids_bins.index(id_val)
            px_j = px_bins.index(px_val)
            bin_lookup[(id_i, px_j)] = cnt
        bar_height = np.array([bin_lookup[(i,j)] for i,j in zip(bar_x, bar_y)])

        # Bar sizes
        dx = dy = 0.05 * np.ones_like(bar_x)  # bar width/depth

        # Set up the plot
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=-110)

        nonzero = bar_height > 0

        X_centers = np.array([ids_bins[i] for i in bar_x])
        Y_centers = np.array([px_bins[j] for j in bar_y])

        # Offset base by half width/depth
        offset_x = X_centers - dx/2
        offset_y = Y_centers - dy/2

        best_id = best["ids_norm"]
        best_px = best["px_norm"]
        best_i = ids_bins.index(best_id) if best_id in ids_bins else None
        best_j = px_bins.index(best_px) if best_px in px_bins else None

        colors = np.tile(np.array([0.0, 0.4, 1.0, 0.3]), (bar_height.size, 1))
        if best_i is not None and best_j is not None:
            best_mask = (bar_x == best_i) & (bar_y == best_j)
            colors[best_mask] = (1.0, 0.2, 0.2, 1.0)

        ax.bar3d(
            offset_x[nonzero], offset_y[nonzero], bar_z[nonzero],
            dx[nonzero], dy[nonzero], bar_height[nonzero],
            shade=True, color=colors[nonzero]
        )
        ax.scatter([1.0], [1.0], [0.0], marker='*', s=200, c='red', zorder=5)

        ax.set_xlabel('Normalized IDs')
        ax.set_ylabel('Normalized Robustness')
        ax.set_zlabel('Number of configurations')
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10
        plt.tight_layout()
        plt.savefig(out_dir / f"{base_stem}_3d_histogram{base_suffix}", dpi=300, bbox_inches="tight")
        plt.show()

    return 0


if __name__ == "__main__":
    # Run without arguments to use defaults, or supply CLI flags for customization.
    raise SystemExit(main(sys.argv[1:]))
