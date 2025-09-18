# from __future__ import annotations
# import numpy as np
# from typing import Dict, Tuple


# # ---- Basic grid ops ----
# def rotate(grid: np.ndarray, k: int) -> np.ndarray:
#     return np.rot90(grid, k % 4)


# def flip_h(grid: np.ndarray) -> np.ndarray:
#     return grid[:, ::-1]


# def flip_v(grid: np.ndarray) -> np.ndarray:
#     return grid[::-1, :]


# def transpose(grid: np.ndarray) -> np.ndarray:
#     return grid.T


# def apply_dihedral(grid: np.ndarray, code: str) -> np.ndarray:
#     """code in {'I','R90','R180','R270','FH','FV','T','FH_R90'} covering D4."""
#     g = grid
#     if code == "I":
#         return g
#     if code == "R90":
#         return rotate(g, 1)
#     if code == "R180":
#         return rotate(g, 2)
#     if code == "R270":
#         return rotate(g, 3)
#     if code == "FH":
#         return flip_h(g)
#     if code == "FV":
#         return flip_v(g)
#     if code == "T":
#         return transpose(g)
#     if code == "FH_R90":
#         return flip_h(rotate(g, 1))
#     return g


# DIHEDRAL_CODES = ["I", "R90", "R180", "R270", "FH", "FV", "T", "FH_R90"]


# # ---- Color ops ----
# def apply_color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
#     if not mapping:
#         return grid.copy()
#     out = grid.copy()
#     maxc = int(out.max(initial=0))
#     lut = np.arange(maxc + 1, dtype=out.dtype)
#     for k, v in mapping.items():
#         if k <= maxc:
#             lut[k] = v
#     return lut[out]


# # ---- Translation (same size, clipped) ----
# def translate(grid: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
#     H, W = grid.shape
#     out = np.full((H, W), fill, dtype=grid.dtype)
#     ys = slice(max(0, dy), min(H, H + dy))
#     xs = slice(max(0, dx), min(W, W + dx))
#     ys_src = slice(max(0, -dy), min(H, H - dy))
#     xs_src = slice(max(0, -dx), min(W, W - dx))
#     out[ys, xs] = grid[ys_src, xs_src]
#     return out


# # ---- Border add/remove ----
# def crop_to_bbox(
#     grid: np.ndarray, background: int
# ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
#     mask = grid != background
#     if not mask.any():
#         return grid.copy(), (0, 0, grid.shape[0], grid.shape[1])
#     rows, cols = np.where(mask)
#     r0, r1 = rows.min(), rows.max() + 1
#     c0, c1 = cols.min(), cols.max() + 1
#     return grid[r0:r1, c0:c1], (r0, c0, r1, c1)


# def add_border(
#     grid: np.ndarray, top: int, bottom: int, left: int, right: int, color: int
# ) -> np.ndarray:
#     H, W = grid.shape
#     out = np.full((H + top + bottom, W + left + right), color, dtype=grid.dtype)
#     out[top : top + H, left : left + W] = grid
#     return out


# # ---- Tiling ----
# def tile_to_shape(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
#     h, w = grid.shape
#     ry = (out_h + h - 1) // h
#     rx = (out_w + w - 1) // w
#     big = np.tile(grid, (ry, rx))
#     return big[:out_h, :out_w]
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


# ---- Basic grid ops ----
def rotate(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k % 4)


def flip_h(grid: np.ndarray) -> np.ndarray:
    return grid[:, ::-1]


def flip_v(grid: np.ndarray) -> np.ndarray:
    return grid[::-1, :]


def transpose(grid: np.ndarray) -> np.ndarray:
    return grid.T


def apply_dihedral(grid: np.ndarray, code: str) -> np.ndarray:
    """code in {'I','R90','R180','R270','FH','FV','T','FH_R90'} covering D4."""
    g = grid
    if code == "I":
        return g
    if code == "R90":
        return rotate(g, 1)
    if code == "R180":
        return rotate(g, 2)
    if code == "R270":
        return rotate(g, 3)
    if code == "FH":
        return flip_h(g)
    if code == "FV":
        return flip_v(g)
    if code == "T":
        return transpose(g)
    if code == "FH_R90":
        return flip_h(rotate(g, 1))
    return g


DIHEDRAL_CODES = ["I", "R90", "R180", "R270", "FH", "FV", "T", "FH_R90"]


# ---- Color ops ----
def apply_color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    if not mapping:
        return grid.copy()
    out = grid.copy()
    maxc = int(out.max(initial=0))
    lut = np.arange(maxc + 1, dtype=out.dtype)
    for k, v in mapping.items():
        if 0 <= k <= maxc:
            lut[k] = v
    return lut[out]


# ---- Translation (same size, clipped) ----
def translate(grid: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    H, W = grid.shape
    out = np.full((H, W), fill, dtype=grid.dtype)
    ys = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, dx), min(W, W + dx))
    ys_src = slice(max(0, -dy), min(H, H - dy))
    xs_src = slice(max(0, -dx), min(W, W - dx))
    out[ys, xs] = grid[ys_src, xs_src]
    return out


# ---- Border add/remove ----
def crop_to_bbox(
    grid: np.ndarray, background: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    mask = grid != background
    if not mask.any():
        return grid.copy(), (0, 0, grid.shape[0], grid.shape[1])
    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return grid[r0:r1, c0:c1], (r0, c0, r1, c1)


def add_border(
    grid: np.ndarray, top: int, bottom: int, left: int, right: int, color: int
) -> np.ndarray:
    """Pad with a uniform border. All margins are clipped to be non-negative to avoid shape errors."""
    H, W = grid.shape
    top = max(0, int(top))
    bottom = max(0, int(bottom))
    left = max(0, int(left))
    right = max(0, int(right))
    out = np.full((H + top + bottom, W + left + right), color, dtype=grid.dtype)
    out[top : top + H, left : left + W] = grid
    return out


# ---- Tiling ----
def tile_to_shape(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = grid.shape
    ry = (out_h + h - 1) // h
    rx = (out_w + w - 1) // w
    big = np.tile(grid, (ry, rx))
    return big[:out_h, :out_w]
