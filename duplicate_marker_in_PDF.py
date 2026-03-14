from PIL import Image

# # ALIX on Probe
# png_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\marker_board.png"
# #pdf_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\marker_board_repetitive.pdf"

# # ALIX on Baby
# png_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\3x3_2x3tag\marker_board.png"
# #pdf_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\3x3_2x3tag\marker_board_repetitive.pdf"

# # EMILY
# png_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag\marker_board.png"
# #pdf_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag\marker_board_repetitive.pdf"

# marker_image = Image.open(png_path)

# dpi = 1200

# # Ensure PNG carries correct DPI metadata
# marker_image.save(png_path, dpi=(dpi, dpi))

# # Letter page size in pixels at 1200 dpi
# letter_width_px  = int(round(8.5 * dpi))
# letter_height_px = int(round(11.0 * dpi))

# # ---- layout parameters ----
# num_markers = 5
# # vertical gap between markers
# gap_px = int(0.20 * dpi)

# total_markers_height = num_markers * marker_image.height + (num_markers - 1) * gap_px

# if marker_image.width > letter_width_px or total_markers_height > letter_height_px:
#     raise ValueError(
#         "Three markers (plus gaps) do not fit on a letter-size page at this DPI."
#     )

# # Create the letter-sized white canvas
# letter_canvas = Image.new("RGB", (letter_width_px, letter_height_px), "white")

# # Horizontal centering
# x_offset = (letter_width_px - marker_image.width) // 2

# # Vertical centering of the whole 3-marker block
# top_margin = (letter_height_px - total_markers_height) // 2

# for i in range(num_markers):
#     y_offset = top_margin + i * (marker_image.height + gap_px)
#     letter_canvas.paste(marker_image, (x_offset, y_offset))

# # Save as high-resolution PDF, preserving the 1200 dpi mapping
# letter_canvas.save(pdf_path, "PDF", resolution=float(dpi))


# from PIL import Image, ImageDraw

# def draw_dotted_rect(draw, x0, y0, x1, y1, *, color=(255, 0, 0), width=1, dash=12, gap=12):
#     """Draw a dotted (dash-gap) rectangle. Assumes axis-aligned rectangle."""
#     def seg_line(p0, p1):
#         if p0[0] == p1[0]:  # vertical
#             x = p0[0]
#             y_start, y_end = sorted([p0[1], p1[1]])
#             y = y_start
#             while y < y_end:
#                 y2 = min(y + dash, y_end)
#                 draw.line([(x, y), (x, y2)], fill=color, width=width)
#                 y += dash + gap
#         else:  # horizontal
#             y = p0[1]
#             x_start, x_end = sorted([p0[0], p1[0]])
#             x = x_start
#             while x < x_end:
#                 x2 = min(x + dash, x_end)
#                 draw.line([(x, y), (x2, y)], fill=color, width=width)
#                 x += dash + gap

#     seg_line((x0, y0), (x1, y0))  # top
#     seg_line((x1, y0), (x1, y1))  # right
#     seg_line((x1, y1), (x0, y1))  # bottom
#     seg_line((x0, y1), (x0, y0))  # left

# # Define your output PDF path
# pdf_path = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\combined_markers.pdf"

# # Define the list of images in the exact order you want them printed
# # You can repeat paths if you want the same image multiple times
# image_paths = [
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\4x4_3x3tag_2x4tag_a\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\4x4_3x3tag_2x4tag_b\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\4x4_3x3tag_2x4tag_c\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\4x4_3x3tag_2x4tag_d\marker_board.png",

#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x6_3x3tag_1x6tag_a\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x6_3x3tag_1x6tag_b\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x6_3x3tag_1x6tag_c\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x6_3x3tag_1x6tag_d\marker_board.png",

#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_e\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_f\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_g\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_h\marker_board.png",

#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag_a\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag_b\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag_c\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\9x9_3x3tag_1x9tag_d\marker_board.png",

#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_a\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_b\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_c\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\5x5_3x3tag_1x5tag_d\marker_board.png",

#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\marker_board.png",
#     r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\marker_board.png",
# ]

# path_to_rotate = r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers\6x10_3x3tag_1x10tag_6x2tag\marker_board.png"

# dpi = 1200
# margin_inches = 0.3
# gap_inches = 0.1  # Gap between images

# letter_width_px  = int(round(8.5 * dpi))
# letter_height_px = int(round(11.0 * dpi))
# margin_px = int(round(margin_inches * dpi))
# gap_px = int(round(gap_inches * dpi))

# # Create the canvas
# letter_canvas = Image.new("RGB", (letter_width_px, letter_height_px), "white")
# draw = ImageDraw.Draw(letter_canvas)

# # Initialize cursor positions at the top-left margin
# current_x = margin_px
# current_y = margin_px

# # Track the widest image in the current column so we know how far to step right later
# current_column_width = 0

# for path in image_paths:
#     try:
#         img = Image.open(path)
#     except FileNotFoundError:
#         print(f"Warning: Could not find file {path}. Skipping.")
#         continue

#     img.save(path, dpi=(dpi, dpi))

#     if path == path_to_rotate:
#             img = img.rotate(90, expand=True) 

#     w, h = img.size




#     # 1. Check if we fit vertically in the current column
#     # If adding this image exceeds the bottom margin...
#     if current_y + h > letter_height_px - margin_px:
#         # Move to the next column:
#         # Shift X by the width of the previous column + gap
#         current_x += current_column_width + gap_px
#         # Reset Y to the top
#         current_y = margin_px
#         # Reset column width for the new column
#         current_column_width = 0

#     # 2. Check if we fit horizontally (did we run off the right side?)
#     if current_x + w > letter_width_px - margin_px:
#         print("Warning: Page is full! Stopping execution to prevent cutoff.")
#         break

#     # 3. Paste the image
#     letter_canvas.paste(img, (current_x, current_y))

#     pad = 0  # set to 1 if you want the line just OUTSIDE the image edge
#     x0, y0 = current_x - pad, current_y - pad
#     x1, y1 = current_x + w - 1 + pad, current_y + h - 1 + pad
#     draw_dotted_rect(draw, x0, y0, x1, y1, color=(255, 0, 0), width=1, dash=12, gap=12)

#     # 4. Update cursor for next image
#     # Move Y down by height of current image + gap
#     current_y += h + gap_px
    
#     # Update the max width of the current column
#     if w > current_column_width:
#         current_column_width = w

# letter_canvas.save(pdf_path, "PDF", resolution=float(dpi))
# print(f"PDF saved successfully at: {pdf_path}")


from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors

DPI = 1200
PAD_CELLS = 1
DOT_RADIUS = 0.2

def infer_cell_and_pad_px(png_path: Path, sta_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Infer cell_px and partial_pad_px from marker_board.png size and sta shape.

    marker_board.png size (after your crop) is:
        width_px  = W*cell_px + 2*partial_pad_px
        height_px = H*cell_px + 2*partial_pad_px

    Your generator uses partial_pad_px ≈ round(0.5*cell_px), so:
        width_px ≈ (W + 1)*cell_px  -> cell_px ≈ width_px/(W+1)
    """
    img = Image.open(png_path)
    width_px, height_px = img.size
    H, W = sta_shape

    cell_x = int(round(width_px / (W + 1)))
    cell_y = int(round(height_px / (H + 1)))
    cell_px = int(round((cell_x + cell_y) / 2))

    partial_pad_px = int(round((width_px - W * cell_px) / 2))
    # sanity fallback
    if partial_pad_px <= 0:
        partial_pad_px = int(round(0.5 * cell_px))

    return cell_px, partial_pad_px

def marker_size_pt(sta: np.ndarray, cell_px: int, partial_pad_px: int) -> tuple[float, float]:
    H, W = sta.shape
    w_px = W * cell_px + 2 * partial_pad_px
    h_px = H * cell_px + 2 * partial_pad_px
    w_pt = (w_px / DPI) * inch
    h_pt = (h_px / DPI) * inch
    return w_pt, h_pt

def draw_marker_vector(c: canvas.Canvas, sta: np.ndarray, cell_px: int, partial_pad_px: int):
    """
    Draw the cropped marker as vector art at the current origin (0,0),
    assuming the page coordinate system is bottom-left origin.
    """
    H, W = sta.shape

    cell_pt = (cell_px / DPI) * inch
    pad_pt  = (partial_pad_px / DPI) * inch
    dot_r_pt = DOT_RADIUS * cell_pt

    total_h = H + 2 * PAD_CELLS
    total_w = W + 2 * PAD_CELLS

    crop_w = W * cell_pt + 2 * pad_pt
    crop_h = H * cell_pt + 2 * pad_pt

    # Crop window in padded-board coordinates:
    left_crop = PAD_CELLS * cell_pt - pad_pt
    bot_crop  = PAD_CELLS * cell_pt - pad_pt

    # Clip to crop box and shift so crop box aligns with (0,0)
    c.saveState()
    p = c.beginPath()
    p.rect(0, 0, crop_w, crop_h)
    c.clipPath(p, stroke=0, fill=0)
    c.translate(-left_crop, -bot_crop)

    # Checkerboard squares
    for row in range(total_h):
        for col in range(total_w):
            is_white = ((row + col) % 2 == 1)  # matches your pattern
            c.setFillColor(colors.white if is_white else colors.black)

            x = col * cell_pt
            y = (total_h - row - 1) * cell_pt  # map top-row to top
            c.rect(x, y, cell_pt, cell_pt, stroke=0, fill=1)

    # Dots (same logic: black on white square, white on black square)
    ys, xs = np.nonzero(sta)
    for y_cell, x_cell in zip(ys, xs):
        row = y_cell + PAD_CELLS
        col = x_cell + PAD_CELLS
        cell_is_white = ((row + col) % 2 == 1)

        c.setFillColor(colors.black if cell_is_white else colors.white)
        cx = (col + 0.5) * cell_pt
        cy = (total_h - row - 0.5) * cell_pt
        c.circle(cx, cy, dot_r_pt, stroke=0, fill=1)

    c.restoreState()

def build_combined_vector_pdfs(
    marker_dirs: list[Path],
    out_art_pdf: Path,
    out_die_pdf: Path,
    margin_inches=0.3,
    gap_inches=0.1,
    rotate_dir: Path | None = None,
):
    page_w, page_h = letter
    margin = margin_inches * inch
    gap = gap_inches * inch

    # Spot color for cutline (ask vendor if they want a specific name)
    cut = colors.CMYKColorSep(0, 1, 0, 0, spotName="CutContour")
    cut_line_width = 0.25  # pt

    c_art = canvas.Canvas(str(out_art_pdf), pagesize=letter)
    c_die = canvas.Canvas(str(out_die_pdf), pagesize=letter)

    current_x = margin
    current_y_top = page_h - margin
    current_column_width = 0

    for d in marker_dirs:
        sta_path = d / "marker_sta.npy"
        png_path = d / "marker_board.png"
        if not sta_path.exists() or not png_path.exists():
            print("Skipping (missing files):", d)
            continue

        sta = np.load(sta_path).astype(np.uint8)
        cell_px, partial_pad_px = infer_cell_and_pad_px(png_path, sta.shape)

        w_pt, h_pt = marker_size_pt(sta, cell_px, partial_pad_px)

        rotate = (rotate_dir is not None and d.resolve() == rotate_dir.resolve())
        if rotate:
            w_pt, h_pt = h_pt, w_pt  # swapped footprint on page

        # column wrap (vertical)
        if (current_y_top - h_pt) < margin:
            current_x += current_column_width + gap
            current_y_top = page_h - margin
            current_column_width = 0

        # stop if no horizontal room
        if (current_x + w_pt) > (page_w - margin):
            print("Page full, stopping before cutoff.")
            break

        y_bottom = current_y_top - h_pt

        # --- ART placement ---
        c_art.saveState()
        c_art.translate(current_x, y_bottom)
        if rotate:
            # rotate around bottom-left of the placement box
            c_art.translate(0, h_pt)
            c_art.rotate(-90)
        draw_marker_vector(c_art, sta, cell_px, partial_pad_px)
        c_art.restoreState()

        # --- DIELINE placement (vector rectangle only) ---
        c_die.setStrokeColor(cut)
        c_die.setLineWidth(cut_line_width)
        c_die.rect(current_x, y_bottom, w_pt, h_pt, stroke=1, fill=0)

        # advance cursor
        current_y_top = y_bottom - gap
        current_column_width = max(current_column_width, w_pt)

    c_art.showPage(); c_art.save()
    c_die.showPage(); c_die.save()
    print("Wrote:", out_art_pdf)
    print("Wrote:", out_die_pdf)

if __name__ == "__main__":
    base = Path(r"C:\Users\samue\Desktop\GIT_STL\US_Trackers\Sam_HydraMarkers")

    # Put your marker folders in the exact order you want.
    marker_dirs = [
        base / r"4x4_3x3tag_2x4tag_a",
        base / r"4x4_3x3tag_2x4tag_b",
        base / r"4x4_3x3tag_2x4tag_c",
        base / r"4x4_3x3tag_2x4tag_d",

        base / r"6x6_3x3tag_1x6tag_a",
        base / r"6x6_3x3tag_1x6tag_b",
        base / r"6x6_3x3tag_1x6tag_c",
        base / r"6x6_3x3tag_1x6tag_d",

        base / r"5x5_3x3tag_1x5tag_e",
        base / r"5x5_3x3tag_1x5tag_f",
        base / r"5x5_3x3tag_1x5tag_g",
        base / r"5x5_3x3tag_1x5tag_h",

        base / r"9x9_3x3tag_1x9tag_a",
        base / r"9x9_3x3tag_1x9tag_b",
        base / r"9x9_3x3tag_1x9tag_c",
        base / r"9x9_3x3tag_1x9tag_d",

        base / r"5x5_3x3tag_1x5tag_a",
        base / r"5x5_3x3tag_1x5tag_b",
        base / r"5x5_3x3tag_1x5tag_c",
        base / r"5x5_3x3tag_1x5tag_d",

        base / r"6x10_3x3tag_1x10tag_6x2tag",
        base / r"6x10_3x3tag_1x10tag_6x2tag",  # repeated on purpose
    ]

    rotate_dir = base / r"6x10_3x3tag_1x10tag_6x2tag"

    build_combined_vector_pdfs(
        marker_dirs=marker_dirs,
        out_art_pdf=base / "combined_markers_ART_vector.pdf",
        out_die_pdf=base / "combined_markers_DIELINE_vector.pdf",
        margin_inches=0.3,
        gap_inches=0.1,
        rotate_dir=rotate_dir,
    )
