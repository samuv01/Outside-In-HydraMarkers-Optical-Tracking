"""
Python port of the `generator_HydraMarker` interface.

The original C++ implementation ships several generation strategies.  This
module currently provides a pure-Python translation of the fast bWFC approach
(`METHOD::FBWFC`) together with the utility routines for loading, saving, and
visualising marker fields.  Remaining strategies can be added incrementally
using the same structure.
"""

from __future__ import annotations

import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, TextIO, Tuple

import cv2
import numpy as np

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil is optional at runtime
    psutil = None


class Method(str, Enum):
    """Enumeration matching the C++ `METHOD` enum."""

    FBWFC = "FBWFC"
    BWFC = "BWFC"
    DOF = "DOF"
    GENETIC = "GENETIC"


class HydraMarkerGenerator:
    """
    Python equivalent of the C++ `generator_HydraMarker` class.

    The implementation mirrors the public API so existing examples can be ported
    gradually.  The fast-bWFC generation logic is available; the remaining
    strategies raise ``NotImplementedError`` placeholders until ported.
    """

    def __init__(self) -> None:
        self._field: np.ndarray | None = None
        self._tag_shapes: List[np.ndarray] = []
        self._time_start: float = 0.0
        self._max_ms: float = 3_600_000_000.0
        self._max_trial: int = np.iinfo(np.int32).max

        self._tree: List[List[Tuple[int, int, int]]] = []
        self._depth: int = 0
        self._rot_prop: List[int] = []
        self._tag2x: List[np.ndarray] = []
        self._tag2y: List[np.ndarray] = []
        self._tag_map: List[Tuple[np.ndarray, np.ndarray]] = []
        self._xy2tag: Dict[Tuple[int, int], List[List[np.ndarray]]] = {}
        self._log_handle: TextIO | None = None

    # ------------------------------------------------------------------
    # Marker-field configuration
    # ------------------------------------------------------------------
    def set_field(self, field: np.ndarray | str | Path) -> None:
        """
        Set the marker field.

        Parameters
        ----------
        field:
            Either a `numpy.ndarray`/OpenCV image (dtype=uint8, values
            0,1,2,3) or a path to a `.field` file created by `save`.
        """

        if isinstance(field, (str, Path)):
            loaded_field, tag_shapes = self._load_field(Path(field))
            self._field = loaded_field
            self._tag_shapes = tag_shapes
        else:
            arr = np.asarray(field, dtype=np.uint8)
            if arr.ndim != 2:
                raise ValueError("marker field must be a 2D array")
            self._field = arr.copy()
        self._reset_cache()

    def set_field_from_file(self, path: str | Path) -> None:
        """Explicit helper mirroring the C++ overload that accepts a file path."""

        self.set_field(path)

    # ------------------------------------------------------------------
    # Tag-shape configuration
    # ------------------------------------------------------------------
    def set_tagShape(self, shapes: Sequence[np.ndarray] | np.ndarray) -> None:
        """
        Set the tag shapes used during generation.

        Accepts either a single `numpy.ndarray` (matching the C++ overload that
        takes one `Mat1b`) or a sequence of arrays.  Each array must be uint8 and
        contain only 0/1 values, where 0 denotes hollow and 1 denotes solid.
        """

        if isinstance(shapes, np.ndarray):
            shapes_to_process: Sequence[np.ndarray] = [shapes]
        else:
            shapes_to_process = shapes

        processed: List[np.ndarray] = []
        for shape in shapes_to_process:
            arr = np.asarray(shape, dtype=np.uint8)
            if arr.ndim != 2:
                raise ValueError("tag shapes must be 2D arrays")
            if not np.isin(arr, (0, 1)).all():
                raise ValueError("tag shapes must contain only 0 and 1 values")
            processed.append(arr.copy())
        self._tag_shapes = processed
        self._reset_cache()

    # Provide the camelCase alias expected by existing code.
    set_tag_shape = set_tagShape

    # ------------------------------------------------------------------
    # Generation pipeline
    # ------------------------------------------------------------------
    def generate(
        self,
        method: Method = Method.FBWFC,
        max_ms: float = 3_600_000_000,
        max_trial: int = np.iinfo(np.int32).max,
        show_process: bool = True,
        log_path: str | Path = "process.log",
    ) -> None:
        """
        Generate the marker field by filling unknown (value=2) cells.
        """

        if self._field is None:
            raise RuntimeError("marker field is not initialised")
        if not self._tag_shapes:
            raise RuntimeError("no tag shapes configured")

        self._time_start = cv2.getTickCount()
        self._max_ms = max_ms
        self._max_trial = max_trial

        log_path = Path(log_path)
        if log_path.suffix.lower() != ".log":
            raise ValueError("log file must use the .log suffix")
        print(f"log file will be refreshed, path: {log_path}")

        self._log_handle = log_path.open("w", encoding="utf-8")
        try:
            if method is Method.FBWFC:
                print("generate marker field by fast-bWFC")
                print(f"in {self._max_ms} ms")
                self._generate_fbwfc(show_process)
                return
            if method is Method.DOF:
                raise NotImplementedError("DOF generation has not been ported yet")
            if method is Method.BWFC:
                raise NotImplementedError("bWFC generation has not been ported yet")
            if method is Method.GENETIC:
                raise NotImplementedError("Genetic generation has not been ported yet")
        finally:
            self._log_handle.close()
            self._log_handle = None

        raise ValueError(f"unsupported generation method: {method!s}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def show(self) -> np.ndarray:
        """
        Render the current marker field as an RGB image.

        Returns
        -------
        np.ndarray
            8-bit BGR image where:
            - 0 -> black
            - 1 -> white
            - 2 -> red
            - 3 -> grey
        """

        if self._field is None:
            raise RuntimeError("marker field is not initialised")

        table_r = np.zeros((256,), dtype=np.uint8)
        table_g = np.zeros_like(table_r)
        table_b = np.zeros_like(table_r)
        table_r[1], table_r[2], table_r[3] = 255, 0, 128
        table_g[1], table_g[2], table_g[3] = 255, 0, 128
        table_b[1], table_b[2], table_b[3] = 255, 255, 128

        state_r = cv2.LUT(self._field, table_r)
        state_g = cv2.LUT(self._field, table_g)
        state_b = cv2.LUT(self._field, table_b)

        state_show = cv2.merge((state_b, state_g, state_r))
        cv2.namedWindow("state", cv2.WINDOW_NORMAL)
        cv2.imshow("state", state_show)
        return state_show

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        Persist the current marker field and tag shapes to a `.field` file.
        """

        if self._field is None:
            raise RuntimeError("marker field is not initialised")

        path = Path(path)
        if path.suffix.lower() != ".field":
            raise ValueError("output file must use the .field suffix")

        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"{self._field.shape[0]} {self._field.shape[1]}\n")
            fh.write(" ".join(str(int(v)) for v in self._field.flat))
            fh.write("\n")

            fh.write(f"{len(self._tag_shapes)}\n")
            for shape in self._tag_shapes:
                fh.write(f"{shape.shape[0]} {shape.shape[1]}\n")
                fh.write(" ".join(str(int(v)) for v in shape.flat))
                fh.write("\n")

    def HDhist(self, order: int = 3) -> List[int]:
        """
        Compute the hamming-distance histogram for the current field.

        Parameters
        ----------
        order:
            Size of the square tag (3, 4, or 5) for which the histogram should
            be evaluated. Matches the capabilities of the original C++
            implementation.
        """

        if self._field is None:
            raise RuntimeError("marker field is not initialised")
        if order not in (3, 4, 5):
            raise ValueError("HD histogram currently supports orders 3, 4, and 5")

        field = self._field
        field_h, field_w = field.shape
        order_sq = order * order

        # No valid placements if the tag is larger than the field.
        if field_h < order or field_w < order:
            return [0] * (order_sq + 1)

        # Prepare coordinate templates for the four tag orientations.
        grid_x, grid_y = np.meshgrid(
            np.arange(order, dtype=np.int32),
            np.arange(order, dtype=np.int32),
        )
        coords_x = grid_x.reshape(-1)
        coords_y = grid_y.reshape(-1)

        placements: List[np.ndarray] = []

        for rot in range(4):
            if rot > 0:
                # Rotate the coordinate template by 90 degrees clockwise.
                coords_x, coords_y = coords_y.copy(), coords_x.copy()
                min_x = int(coords_x.min())
                max_x = int(coords_x.max())
                min_y = int(coords_y.min())
                max_y = int(coords_y.max())
                coords_x = -coords_x + (min_x + max_x)
                min_x = int(coords_x.min())
                max_x = int(coords_x.max())
            else:
                min_x = int(coords_x.min())
                max_x = int(coords_x.max())
                min_y = int(coords_y.min())
                max_y = int(coords_y.max())

            for y_offset in range(-min_y, field_h - max_y):
                for x_offset in range(-min_x, field_w - max_x):
                    sample_x = coords_x + x_offset
                    sample_y = coords_y + y_offset
                    patch = field[sample_y, sample_x]

                    if np.any(patch == 3):
                        continue

                    placements.append(patch.astype(np.uint8))

        if not placements:
            return [0] * (order_sq + 1)

        state_by_tag = np.vstack(placements)
        hist = np.zeros(order_sq + 1, dtype=np.int64)

        # Accumulate hamming distances for every unique tag pair.
        for idx in range(state_by_tag.shape[0] - 1):
            diffs = np.count_nonzero(state_by_tag[idx + 1 :] != state_by_tag[idx], axis=1)
            counts = np.bincount(diffs, minlength=hist.size)
            hist[: counts.size] += counts

        print("\nHamming distance histogram:")
        print(" ".join(str(int(v)) for v in hist))

        return hist.astype(int).tolist()

    # snake_case alias for Python users.
    hd_hist = HDhist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_cache(self) -> None:
        """Invalidate cached tables derived from the field/tag state."""

        self._tree = []
        self._depth = 0
        self._rot_prop = []
        self._tag2x = []
        self._tag2y = []
        self._tag_map = []
        self._xy2tag = {}

    def _generate_fbwfc(self, show_process: bool) -> None:
        assert self._field is not None

        self._depth = 0
        self._tree = [[] for _ in range(self._field.size)]
        self._build_table()

        for shape_idx, shape in enumerate(self._tag_shapes):
            ones = int(np.count_nonzero(shape == 1))
            tag_num = math.pow(2.0, ones - math.sqrt(float(self._rot_prop[shape_idx])))
            kernel = (shape == 1).astype(np.uint8)
            need_fill = cv2.erode(
                (self._field == 2).astype(np.uint8),
                kernel,
                anchor=(-1, -1),
                iterations=1,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            need_num = int(np.count_nonzero(need_fill))
            if tag_num < need_num:
                print("\nthe request marker field is larger than possible!")
                break

        num_tag = sum(table.shape[0] for table in self._tag2x)
        step_count = 1
        progress = 0.0
        time_first_stuck = cv2.getTickCount()

        while True:
            state_by_tag = self._read_state()
            complete = self._get_complete(state_by_tag)
            incomplete, incomplete_info = self._get_incomplete(state_by_tag)

            ticks_now = cv2.getTickCount()
            time_step = 1000.0 * (ticks_now - self._time_start) / cv2.getTickFrequency()
            mem_bytes = 0
            if psutil is not None:
                mem_bytes = psutil.Process(os.getpid()).memory_info().rss

            num_complete = sum(table.shape[0] for table in complete)
            cur_progress = 100.0 * num_complete / max(num_tag, 1)
            if cur_progress <= progress:
                time_stuck = (
                    1000.0 * (ticks_now - time_first_stuck) / cv2.getTickFrequency()
                )
                if cur_progress > 110.0 and time_stuck > 60_000 and time_stuck > 0.5 * time_step:
                    print("\nget stuck!")
                    return
            else:
                time_first_stuck = ticks_now
            progress = max(progress, cur_progress)

            log_line = (
                f"step: {step_count}, time: {time_step:.0f}ms, "
                f"mem: {mem_bytes // 1_048_576}MB, progress: {progress:.2f}%"
            )
            if self._log_handle is not None:
                self._log_handle.write(
                    f"{step_count} {time_step} {progress} {mem_bytes}\n"
                )
                self._log_handle.flush()
            print(log_line, end="\r", flush=True)

            if time_step > self._max_ms or step_count >= self._max_trial:
                print("\ntime/step out!")
                return

            if self._has_conflict(complete):
                while True:
                    if self._depth == 0:
                        raise RuntimeError(
                            "fast-bWFC: conflict inevitable; cannot generate marker field"
                        )
                    self._depth -= 1
                    x, y, _ = self._tree[self._depth][0]
                    self._field[y, x] = np.uint8(2)
                    self._tree[self._depth].pop(0)
                    if self._tree[self._depth]:
                        break
            else:
                if int(np.count_nonzero(self._field == 2)) == 0:
                    print("\ncomplete!")
                    return
                focus = self._dof(state_by_tag, incomplete, incomplete_info)
                if focus == (-1, -1):
                    print("\ncomplete!")
                    return
                value = int(self._risk(state_by_tag, complete, focus))
                self._tree[self._depth] = [
                    (focus[0], focus[1], value),
                    (focus[0], focus[1], 1 - value),
                ]

            x, y, val = self._tree[self._depth][0]
            self._field[y, x] = np.uint8(val)
            self._depth += 1
            step_count += 1

            if show_process:
                self.show()
                cv2.waitKey(1)

    def _build_table(self) -> None:
        assert self._field is not None

        self._rot_prop = []
        self._tag2x = []
        self._tag2y = []
        self._tag_map = []

        for shape in self._tag_shapes:
            rot90 = cv2.rotate(shape, cv2.ROTATE_90_CLOCKWISE)
            if shape.shape == rot90.shape and np.array_equal(shape, rot90):
                self._rot_prop.append(4)
                continue

            rot180 = cv2.rotate(rot90, cv2.ROTATE_90_CLOCKWISE)
            if shape.shape == rot180.shape and np.array_equal(shape, rot180):
                self._rot_prop.append(2)
                continue

            self._rot_prop.append(1)

        field_h, field_w = self._field.shape
        xy_template = []
        for rot_count in self._rot_prop:
            xy_template.append([[] for _ in range(rot_count)])

        self._xy2tag = {}
        for y in range(field_h):
            for x in range(field_w):
                per_shape = [list(map(list, template)) for template in xy_template]
                self._xy2tag[(x, y)] = per_shape

        for shape_idx, shape in enumerate(self._tag_shapes):
            x_coords = np.tile(
                np.arange(shape.shape[1], dtype=np.int32), (shape.shape[0], 1)
            )
            y_coords = np.tile(
                np.arange(shape.shape[0], dtype=np.int32).reshape(-1, 1),
                (1, shape.shape[1]),
            )

            mask = shape == 1
            X = x_coords[mask].astype(np.int32)
            Y = y_coords[mask].astype(np.int32)

            if X.size == 0:
                self._tag2x.append(np.empty((0, 0), dtype=np.int32))
                self._tag2y.append(np.empty((0, 0), dtype=np.int32))
                self._tag_map.append(
                    (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32))
                )
                continue

            X_rot = X.copy()
            Y_rot = Y.copy()
            rows_x: List[np.ndarray] = []
            rows_y: List[np.ndarray] = []

            for rot in range(4):
                if rot > 0:
                    X_rot, Y_rot = Y_rot.copy(), X_rot.copy()
                    min_x, max_x = X_rot.min(), X_rot.max()
                    min_y, max_y = Y_rot.min(), Y_rot.max()
                    X_rot = -X_rot + (min_x + max_x)
                else:
                    min_x, max_x = X_rot.min(), X_rot.max()
                    min_y, max_y = Y_rot.min(), Y_rot.max()

                for dy in range(-min_y, field_h - max_y):
                    for dx in range(-min_x, field_w - max_x):
                        xs = X_rot + dx
                        ys = Y_rot + dy
                        values = self._field[ys, xs]
                        if np.any(values == 3):
                            continue
                        rows_x.append(xs.copy())
                        rows_y.append(ys.copy())

            tag2x = np.array(rows_x, dtype=np.int32)
            tag2y = np.array(rows_y, dtype=np.int32)
            self._tag2x.append(tag2x)
            self._tag2y.append(tag2y)
            self._tag_map.append(
                (tag2x.astype(np.float32), tag2y.astype(np.float32))
            )

            if tag2x.size == 0:
                continue

            total_rows = tag2x.shape[0]
            rot_prop = self._rot_prop[shape_idx]
            block_size = total_rows // max(rot_prop, 1)
            for rot in range(rot_prop):
                start = rot * block_size
                end = (rot + 1) * block_size if rot < rot_prop - 1 else total_rows
                for row_idx in range(start, end):
                    xs = tag2x[row_idx]
                    ys = tag2y[row_idx]
                    for col_idx in range(xs.size):
                        x_val = int(xs[col_idx])
                        y_val = int(ys[col_idx])
                        self._xy2tag[(x_val, y_val)][shape_idx][rot].append(
                            (row_idx, col_idx)
                        )

        for key, per_shape in self._xy2tag.items():
            for shape_idx, orient_lists in enumerate(per_shape):
                for rot_idx, entries in enumerate(orient_lists):
                    if entries:
                        orient_lists[rot_idx] = np.array(entries, dtype=np.int32)
                    else:
                        orient_lists[rot_idx] = np.empty((0, 2), dtype=np.int32)
                per_shape[shape_idx] = orient_lists
            self._xy2tag[key] = per_shape

    def _read_state(self) -> List[np.ndarray]:
        assert self._field is not None

        maxnum = 32766
        state_by_tag: List[np.ndarray] = []
        for map_x, map_y in self._tag_map:
            rows = map_x.shape[0]
            if rows == 0:
                state_by_tag.append(np.empty((0, map_x.shape[1] if map_x.ndim == 2 else 0), dtype=np.uint8))
                continue
            blocks: List[np.ndarray] = []
            for start in range(0, rows, maxnum):
                end = min(start + maxnum, rows)
                sub_x = map_x[start:end]
                sub_y = map_y[start:end]
                remapped = cv2.remap(
                    self._field,
                    sub_x,
                    sub_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=2,
                )
                blocks.append(remapped)
            state_by_tag.append(np.vstack(blocks))
        return state_by_tag

    @staticmethod
    def _get_complete(state_by_tag: List[np.ndarray]) -> List[np.ndarray]:
        complete: List[np.ndarray] = []
        for table in state_by_tag:
            if table.size == 0:
                complete.append(np.empty((0, table.shape[1] if table.ndim == 2 else 0), dtype=np.uint8))
                continue
            unknowns = np.count_nonzero(table == 2, axis=1)
            complete.append(table[unknowns == 0].copy())
        return complete

    @staticmethod
    def _get_incomplete(
        state_by_tag: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        incomplete: List[np.ndarray] = []
        info: List[np.ndarray] = []
        for table in state_by_tag:
            if table.size == 0:
                incomplete.append(np.empty((0, table.shape[1] if table.ndim == 2 else 0), dtype=np.uint8))
                info.append(np.empty((0, 2), dtype=np.int32))
                continue
            unknowns = np.count_nonzero(table == 2, axis=1)
            mask = unknowns > 0
            incomplete.append(table[mask].copy())
            indices = np.flatnonzero(mask)
            finfo = np.column_stack((indices, unknowns[mask])).astype(np.int32)
            info.append(finfo)
        return incomplete, info

    @staticmethod
    def _has_conflict(complete: List[np.ndarray]) -> bool:
        for table in complete:
            if table.size == 0:
                continue
            cols = table.shape[1]
            powers = np.power(2.0, np.arange(cols, dtype=np.float32))
            key = table.astype(np.float32) @ powers
            if np.unique(key).size != key.size:
                return True
        return False

    def _dof(
        self,
        state_by_tag: List[np.ndarray],
        incomplete: List[np.ndarray],
        freedom: List[np.ndarray],
    ) -> Tuple[int, int]:
        min_freedom = None
        min_tag: Tuple[int, int] | None = None
        for shape_idx, info in enumerate(freedom):
            if info.size == 0:
                continue
            idx = int(np.argmin(info[:, 1]))
            value = int(info[idx, 1])
            if min_freedom is None or value < min_freedom:
                min_freedom = value
                min_tag = (shape_idx, int(info[idx, 0]))
        if min_tag is None:
            return (-1, -1)

        tag_row = state_by_tag[min_tag[0]][min_tag[1]]
        unknown_positions = np.flatnonzero(tag_row == 2)
        if unknown_positions.size == 0:
            return (-1, -1)
        pos = int(unknown_positions[0])
        x = int(self._tag2x[min_tag[0]][min_tag[1], pos])
        y = int(self._tag2y[min_tag[0]][min_tag[1], pos])
        return (x, y)

    def _risk(
        self,
        state_by_tag: List[np.ndarray],
        complete: List[np.ndarray],
        focus: Tuple[int, int],
    ) -> bool:
        fx, fy = focus
        relate_info = self._xy2tag.get((fx, fy))
        if relate_info is None:
            return False

        risk0 = 0
        risk1 = 0
        for shape_idx, orient_lists in enumerate(relate_info):
            state_table = state_by_tag[shape_idx]
            if state_table.size == 0:
                continue

            relate_tables: List[np.ndarray] = []
            relate_entries: List[np.ndarray] = []
            for entries in orient_lists:
                if entries.size == 0:
                    continue
                relate_entries.append(entries)
                relate_tables.append(state_table[entries[:, 0]])

            compare = (
                np.vstack(relate_tables) if relate_tables else np.empty((0, state_table.shape[1]), dtype=np.uint8)
            )
            if complete[shape_idx].size:
                compare = (
                    np.vstack((compare, complete[shape_idx]))
                    if compare.size
                    else complete[shape_idx].copy()
                )

            if compare.size == 0:
                continue

            if relate_entries:
                entries_all = np.vstack(relate_entries)
            else:
                entries_all = np.empty((0, 2), dtype=np.int32)

            for tag_idx, f_ind in entries_all:
                f_tag = state_table[tag_idx]
                f_tag_matrix = np.repeat(f_tag[np.newaxis, :], compare.shape[0], axis=0)
                match = (f_tag_matrix == 2) | (compare == 2) | (f_tag_matrix == compare)
                full_match = np.all(match, axis=1)
                if not np.any(full_match):
                    continue
                compare_col = compare[:, f_ind]
                risk0 += int(np.count_nonzero(full_match & (compare_col == 0)))
                risk1 += int(np.count_nonzero(full_match & (compare_col == 1)))

        return risk0 > risk1

    @staticmethod
    def _load_field(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
        with path.open("r", encoding="utf-8") as fh:
            dims = fh.readline().strip().split()
            if len(dims) != 2:
                raise ValueError("invalid field file header")
            rows, cols = map(int, dims)

            values = np.fromstring(fh.readline(), dtype=np.uint8, sep=" ")
            if values.size != rows * cols:
                raise ValueError("field payload size mismatch")
            field = values.reshape((rows, cols))

            shape_count_line = fh.readline()
            if not shape_count_line:
                return field.copy(), []

            shape_count = int(shape_count_line.strip())
            shapes: List[np.ndarray] = []
            for _ in range(shape_count):
                meta = fh.readline()
                if not meta:
                    raise ValueError("unexpected EOF while reading tag shapes")
                shape_rows, shape_cols = map(int, meta.strip().split())
                payload = fh.readline()
                if payload is None:
                    raise ValueError("unexpected EOF while reading tag shapes")
                shape_values = np.fromstring(payload, dtype=np.uint8, sep=" ")
                if shape_values.size != shape_rows * shape_cols:
                    raise ValueError("tag shape payload size mismatch")
                shapes.append(shape_values.reshape((shape_rows, shape_cols)))

        return field.copy(), [shape.copy() for shape in shapes]
