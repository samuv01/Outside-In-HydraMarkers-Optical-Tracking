import sys
import argparse
import cv2
import numpy as np
from scipy.io import savemat
from pathlib import Path
from PIL import Image

from generator_hydra_marker_SV import HydraMarkerGenerator, Method


def perturb_unknown_cells(board: np.ndarray, fraction: float = 0.3, seed: int | None = None) -> None:
    """Randomly fix a subset of the unknown (value=2) cells to 0 or 1."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be within (0, 1]")

    unknown = np.argwhere(board == 2)
    if unknown.size == 0:
        return

    rng = np.random.default_rng(seed)
    count = max(1, int(len(unknown) * fraction))
    count = min(count, len(unknown))

    chosen_idx = rng.choice(len(unknown), size=count, replace=False)
    chosen_cells = unknown[chosen_idx]
    random_values = rng.integers(0, 2, size=count, dtype=np.uint8)

    board[chosen_cells[:, 0], chosen_cells[:, 1]] = random_values

# ALIX on probe and EMILY
#DEFAULT_CELL_SIZE = 378

# ALIX on baby
DEFAULT_CELL_SIZE = 473

#DEFAULT_CELL_SIZE = 394

DEFAULT_DOT_RADIUS = 0.2
DEFAULT_DOT_SUPERSAMPLE = 8
PERTURB_FRACTION = 0.3
PERTURB_SEED = None  # set to an integer for reproducible perturbation
PADDING_CELLS = 1  # whole-cell padding added before cropping
PARTIAL_PAD_PX = int(round(0.2 * DEFAULT_CELL_SIZE))  # keep almost full padding to preserve outer corners


def add_dots(
    board_img: np.ndarray,
    sta_mask: np.ndarray,
    cell_px: int = DEFAULT_CELL_SIZE,
    radius: float = DEFAULT_DOT_RADIUS,
    supersample: int = DEFAULT_DOT_SUPERSAMPLE,
) -> np.ndarray:
    """
    Overlay dot markers on the board image.

    supersample > 1 renders dots on an upscaled canvas and downsamples them
    afterwards, yielding smoother edges than drawing directly at base resolution.
    """

    if supersample < 1:
        raise ValueError("supersample must be >= 1")

    if supersample == 1:
        img = board_img.copy()
    else:
        high_shape = (board_img.shape[1] * supersample, board_img.shape[0] * supersample)
        img = cv2.resize(board_img, high_shape, interpolation=cv2.INTER_LINEAR)

    for y, x in zip(*np.nonzero(sta_mask)):
        cy = int((y + 0.5) * cell_px * supersample)
        cx = int((x + 0.5) * cell_px * supersample)
        base = img[cy, cx, 0]
        colour = 0 if base > 128 else 255
        cv2.circle(
            img,
            (cx, cy),
            int(cell_px * radius * supersample),
            (colour, colour, colour),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    if supersample == 1:
        return img

    return cv2.resize(
        img,
        (board_img.shape[1], board_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def make_checkerboard_with_white_padding(core: np.ndarray, pad_cells: int, cell_px: int) -> np.ndarray:
    core_h, core_w = core.shape
    total_h = core_h + 2 * pad_cells
    total_w = core_w + 2 * pad_cells
    pattern = np.fromfunction(
        lambda y, x: (y + x) % 2,
        (total_h, total_w),
        dtype=int,
    ).astype(np.uint8) * 255
    board = cv2.resize(
        pattern,
        (total_w * cell_px, total_h * cell_px),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def crop_partial_checkerboard(
    image: np.ndarray,
    core_shape: tuple[int, int],
    cell_px: int,
    pad_cells: int,
    pad_px: int,
) -> np.ndarray:
    """
    Crop an integer-cell padded checkerboard to show partial outer squares.

    When ``pad_cells`` >= 1 the function keeps the core plus ``pad_px`` pixels
    of the surrounding checker pattern (``pad_px`` < ``cell_px``).
    """

    if pad_cells <= 0 or pad_px <= 0:
        return image
    if pad_px >= pad_cells * cell_px:
        raise ValueError("pad_px must be smaller than the whole-cell padding")

    top = pad_cells * cell_px - pad_px
    left = pad_cells * cell_px - pad_px
    bottom = top + core_shape[0] * cell_px + 2 * pad_px
    right = left + core_shape[1] * cell_px + 2 * pad_px
    return image[top:bottom, left:right]


def render_field_image(field: np.ndarray, cell_px: int = DEFAULT_CELL_SIZE) -> np.ndarray:
    """
    Convert the raw marker field (values 0-3) into a coloured, scaled image.

    Mirrors the palette from ``HydraMarkerGenerator.show`` so the saved preview
    matches the on-screen diagnostics.
    """

    if field.ndim != 2:
        raise ValueError("marker field must be a 2D array")

    table_r = np.zeros((256,), dtype=np.uint8)
    table_g = np.zeros_like(table_r)
    table_b = np.zeros_like(table_r)
    table_r[1], table_r[2], table_r[3] = 255, 0, 128
    table_g[1], table_g[2], table_g[3] = 255, 0, 128
    table_b[1], table_b[2], table_b[3] = 255, 255, 128

    field = np.clip(field, 0, 255).astype(np.uint8, copy=False)
    state_r = cv2.LUT(field, table_r)
    state_g = cv2.LUT(field, table_g)
    state_b = cv2.LUT(field, table_b)

    coloured = cv2.merge((state_b, state_g, state_r))
    if cell_px <= 1:
        return coloured

    height, width = field.shape
    return cv2.resize(
        coloured,
        (width * cell_px, height * cell_px),
        interpolation=cv2.INTER_NEAREST,
    )

def make_checkerboard_with_locked_cells(
    field: np.ndarray,
    pad_cells: int,
    cell_px: int,
    locked_value: int = 3,
    locked_gray: int = 128,
) -> np.ndarray:
    """
    Create a checkerboard image with the same padding as before, but overwrite
    locked cells (field==locked_value) with solid gray squares.

    Returns a BGR image (OpenCV format).
    """
    if field.ndim != 2:
        raise ValueError("field must be 2D")

    core_h, core_w = field.shape
    total_h = core_h + 2 * pad_cells
    total_w = core_w + 2 * pad_cells

    # Base checkerboard at cell resolution (0/255)
    yy, xx = np.indices((total_h, total_w))
    cells = ((yy + xx) % 2).astype(np.uint8) * 255

    # Locked mask placed into padded grid
    locked_mask = np.zeros((total_h, total_w), dtype=bool)
    locked_mask[pad_cells:pad_cells + core_h, pad_cells:pad_cells + core_w] = (field == locked_value)

    # Overwrite locked cells with gray
    cells[locked_mask] = np.uint8(locked_gray)

    # Upscale to pixels
    board_gray = cv2.resize(
        cells,
        (total_w * cell_px, total_h * cell_px),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a Hydra marker and save its assets.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers"),
        help="Directory where generated files (images, numpy/mat files, logs) will be written.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the preview window (e.g. 0.5 halves the window size).",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=DEFAULT_CELL_SIZE,
        help="Pixel size of a checkerboard cell in the rendered outputs.",
    )
    parser.add_argument(
        "--dot-radius",
        type=float,
        default=DEFAULT_DOT_RADIUS,
        help="Radius of the dot relative to the cell size (0-0.5 is typical).",
    )
    parser.add_argument(
        "--dot-supersample",
        type=int,
        default=DEFAULT_DOT_SUPERSAMPLE,
        help="Supersampling factor for dot rendering; higher values yield smoother edges.",
    )
    args = parser.parse_args(argv)

    base_output_dir = args.output_dir.expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    cell_size = args.cell_size
    if cell_size <= 0:
        raise ValueError("cell-size must be positive.")

    dot_radius = args.dot_radius
    if dot_radius <= 0:
        raise ValueError("dot-radius must be positive.")
    if dot_radius >= 0.5:
        print("Warning: dot-radius >= 0.5 may cause overlapping dots.")

    dot_supersample = args.dot_supersample
    if dot_supersample < 1:
        raise ValueError("dot-supersample must be >= 1.")

    gen = HydraMarkerGenerator()
    # Define the field and tag pattern:
    # ALIX
    # on probe
    #field = np.full((6, 10), 2, dtype=np.uint8)
    #field = np.full((12, 13), 2, dtype=np.uint8)

    # on baby
    field = np.full((15, 17), 2, dtype=np.uint8)
    #field = np.full((6, 7), 2, dtype=np.uint8)

    # EMILY
    #field = np.full((4, 4), 2, dtype=np.uint8)
    # # for the probe and the 12x13 field
    # field[:3, :4] = 3
    # field[:3, -4:] = 3
    # field[-4:, :5] = 3
    # field[-4:, -5:] = 3

    # for the probe and the 15x17 field
    field[:4, :5] = 3           # top left
    field[:4, -5:] = 3          # top right
    field[-4:, :6] = 3          # bottom left
    field[-4:, -6:] = 3         # bottom right

    # C = np.zeros((5, 5), dtype=np.uint8)
    # # arms of the cross
    # C[:, 2] = 1      # vertical
    # C[2, :] = 1      # horizontal
    # C[2, 2] = 0      # middle of the cross black

    # patient marker innercells are locked
    # field[2:-2, 2:-2] = 3

    tag_shapes = [
            np.ones((4, 4), dtype=np.uint8),
            np.ones((2, 8), dtype=np.uint8),
    ]

    tag_suffix = "_".join(f"{shape.shape[0]}x{shape.shape[1]}tag" for shape in tag_shapes)
    results_dir_name = f"{field.shape[0]}x{field.shape[1]}_{tag_suffix}_try2"
    output_dir = base_output_dir / results_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # If you wanna make sure Marker Fields are actually different
    #perturb_unknown_cells(field, fraction=PERTURB_FRACTION, seed=PERTURB_SEED)

    gen.set_field(field)
    gen.set_tagShape(tag_shapes)

    log_path = output_dir / "generate.log"

    gen.generate(
        method=Method.FBWFC,
        max_ms=600_000,
        max_trial=500_000,
        show_process=False,
        log_path=str(log_path),
    )

    # Computethe hamming-distance histogram for the current field.
    #hist = gen.HDhist(order=4)

    sta = (gen._field == 1).astype(np.uint8)

    board = make_checkerboard_with_white_padding(sta, PADDING_CELLS, cell_size)
    marker = add_dots(board, np.pad(sta, PADDING_CELLS, mode="constant"), cell_px=cell_size,radius=dot_radius,supersample=dot_supersample)

    partial_pad_px = min(PARTIAL_PAD_PX, max(cell_size - 1, 0))
    marker = crop_partial_checkerboard(marker,sta.shape,cell_size,PADDING_CELLS,partial_pad_px)

    png_path = output_dir / "marker_board.png"
    pdf_path = output_dir / "marker_board.pdf"
    field_png_path = output_dir / "marker_field.png"
    npy_path = output_dir / "marker_sta.npy"
    mat_path = output_dir / "marker_sta.mat"

    field = gen._field
    if field is None:
        raise RuntimeError("generator did not produce a marker field")

    # board_locked = make_checkerboard_with_locked_cells(field, PADDING_CELLS, cell_size, locked_value=3, locked_gray=128)

    # marker_locked = add_dots(
    #     board_locked,
    #     np.pad(sta, PADDING_CELLS, mode="constant"),
    #     cell_px=cell_size,
    #     radius=dot_radius,
    #     supersample=dot_supersample,
    # )

    # marker_locked = crop_partial_checkerboard(
    #     marker_locked,
    #     sta.shape,
    #     cell_size,
    #     PADDING_CELLS,
    #     partial_pad_px,
    # )

    marker_image = Image.fromarray(marker)
    dpi = 1200
    marker_image.save(png_path, dpi=(dpi, dpi))
    
    # locked_png_path = output_dir / "marker_board_locked_gray.png"
    # Image.fromarray(marker_locked).save(locked_png_path, dpi=(dpi, dpi))

    letter_width_px = int(round(8.5 * dpi))
    letter_height_px = int(round(11.0 * dpi))
    if marker_image.width > letter_width_px or marker_image.height > letter_height_px:
        raise ValueError(
            "Marker image exceeds letter-size dimensions at the chosen DPI; "
            "reduce cell-size or DPI to fit."
        )
    letter_canvas = Image.new("RGB", (letter_width_px, letter_height_px), "white")
    offset = (
        (letter_width_px - marker_image.width) // 2,
        (letter_height_px - marker_image.height) // 2,
    )
    letter_canvas.paste(marker_image, offset)
    letter_canvas.save(pdf_path, "PDF", resolution=float(dpi))

    cv2.imwrite(str(field_png_path), render_field_image(field, cell_size))
    np.save(str(npy_path), sta)
    savemat(str(mat_path), {"sta": sta})

    np.set_printoptions(linewidth=120)

    scale = 0.5
    if scale <= 0:
        raise ValueError("display-scale must be positive.")

    preview = marker
    if not np.isclose(scale, 1.0):
        new_size = (int(marker.shape[1] * scale), int(marker.shape[0] * scale))
        if new_size[0] <= 0 or new_size[1] <= 0:
            raise ValueError("display-scale results in zero-sized preview window.")
        preview = cv2.resize(marker, new_size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

    cv2.imshow("Generated Hydra Marker", preview)
    print(f"Marker image saved to {png_path}")
    print(f"Marker PDF saved to {pdf_path}")
    print(f"Marker field image saved to {field_png_path}")
    print(f"Binary mask saved to {npy_path} and {mat_path}")
    print("Press any key to close the preview window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # go into main and change parameters as you prefer. Consider feasibility check when changing field/tag sizes.
