# from __future__ import annotations
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from collections import Counter
# from arc_transforms import (
#     apply_dihedral,
#     DIHEDRAL_CODES,
#     apply_color_map,
#     translate,
#     crop_to_bbox,
#     add_border,
#     tile_to_shape,
# )


# # ---------- Utility ----------
# def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
#     return a.shape == b.shape and np.array_equal(a, b)


# def infer_background_color(grid: np.ndarray) -> int:
#     vals, counts = np.unique(grid, return_counts=True)
#     return int(vals[counts.argmax()])


# # ---------- Color map inference ----------
# def infer_color_mapping(inp: np.ndarray, out: np.ndarray) -> Dict[int, int]:
#     """Cheap global color mapping by frequency ranking."""
#     mapping: Dict[int, int] = {}
#     vi, ci = np.unique(inp, return_counts=True)
#     vo, co = np.unique(out, return_counts=True)
#     order_i = [v for _, v in sorted(zip(ci, vi), reverse=True)]
#     order_o = [v for _, v in sorted(zip(co, vo), reverse=True)]
#     for a, b in zip(order_i, order_o):
#         mapping[int(a)] = int(b)
#     return mapping


# # ---------- Geometric inference ----------
# def best_dihedral(inp: np.ndarray, out: np.ndarray) -> str:
#     best, bestScore = "I", -1
#     for code in DIHEDRAL_CODES:
#         g = apply_dihedral(inp, code)
#         if g.shape != out.shape:
#             continue
#         score = (g == out).sum()
#         if score > bestScore:
#             bestScore, best = score, code
#     return best


# # ---------- Translation inference ----------
# def best_translation(
#     inp: np.ndarray, out: np.ndarray, background: int
# ) -> Tuple[int, int]:
#     H, W = inp.shape
#     best = (0, 0)
#     bestScore = -1
#     for dy in range(-H + 1, H):
#         g_row = translate(inp, dy, 0, background)
#         for dx in range(-W + 1, W):
#             g = translate(g_row, 0, dx, background)
#             score = (g == out).sum()
#             if score > bestScore:
#                 bestScore, best = score, (dy, dx)
#     return best


# # ---------- Border/crop inference ----------
# def detect_crop_or_border(
#     inp: np.ndarray, out: np.ndarray, background: int
# ) -> Tuple[str, Tuple[int, int, int, int], int]:
#     """Return ('none', _, _) or ('crop', bbox, _) or ('border', (top,left,bottom,right), color)."""
#     # crop
#     cropped, bbox = crop_to_bbox(inp, background)
#     if cropped.shape == out.shape and np.array_equal(cropped, out):
#         r0, c0, r1, c1 = bbox
#         return "crop", (r0, c0, r1, c1), 0
#     # border
#     color = infer_background_color(out)
#     H, W = inp.shape
#     Ho, Wo = out.shape
#     for t in range(0, max(1, Ho - H) + 1):
#         for l in range(0, max(1, Wo - W) + 1):
#             b = add_border(inp, t, Ho - H - t, l, Wo - W - l, color)
#             if b.shape == out.shape and np.array_equal(b, out):
#                 return "border", (t, l, Ho - H - t, Wo - W - l), color
#     return "none", (0, 0, 0, 0), 0


# # ---------- Tiling ----------
# def detect_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
#     h, w = inp.shape
#     H, W = out.shape
#     if H % h != 0 or W % w != 0:
#         return False
#     tiled = tile_to_shape(inp, H, W)
#     return np.array_equal(tiled, out)


# # ---------- Rule application ----------
# class Rule:
#     def __init__(
#         self,
#         background: int,
#         dihedral: str,
#         colormap: Dict[int, int],
#         translation: Tuple[int, int],
#         post: str = "none",
#         post_params: Tuple = (),
#         post_color: int = 0,
#     ):
#         self.background = int(background)
#         self.dihedral = dihedral
#         self.colormap = dict(colormap) if colormap else {}
#         self.translation = translation  # (dy, dx)
#         self.post = post  # 'none' | 'crop' | 'border' | 'tile'
#         self.post_params = post_params  # bbox or border tuple or (H,W)
#         self.post_color = int(post_color)

#     def apply(self, grid: np.ndarray) -> np.ndarray:
#         g = grid
#         g = apply_dihedral(g, self.dihedral)
#         if self.colormap:
#             g = apply_color_map(g, self.colormap)
#         if self.translation != (0, 0):
#             g = translate(g, self.translation[0], self.translation[1], self.background)
#         if self.post == "crop":
#             r0, c0, r1, c1 = self.post_params
#             g = g[r0:r1, c0:c1]
#         elif self.post == "border":
#             t, l, b, r = self.post_params
#             g = add_border(g, t, b, l, r, self.post_color)
#         elif self.post == "tile":
#             H, W = self.post_params
#             g = tile_to_shape(g, H, W)
#         return g


# # ---------- Rule inference ----------
# def infer_rule(
#     train_pairs: List[Tuple[np.ndarray, np.ndarray]], background: Optional[int] = None
# ) -> Tuple[Rule, Rule, float]:
#     """Return (best_rule, second_rule, best_score) where score = #train pairs matched exactly."""
#     if background is None:
#         background = int(
#             np.bincount(np.concatenate([p[0].ravel() for p in train_pairs])).argmax()
#         )
#     x0, y0 = train_pairs[0]
#     bg0 = background
#     cand_colormap = infer_color_mapping(x0, y0)
#     colormaps = [cand_colormap, {}]
#     dihedrals = DIHEDRAL_CODES
#     candidates: List[Tuple[Rule, int]] = []
#     for cm in colormaps:
#         for d in dihedrals:
#             g0 = apply_dihedral(x0, d)
#             if cm:
#                 g0 = apply_color_map(g0, cm)
#             if g0.shape == y0.shape:
#                 dy, dx = best_translation(g0, y0, bg0)
#                 rule = Rule(bg0, d, cm, (dy, dx))
#                 rtype, params, color = detect_crop_or_border(rule.apply(x0), y0, bg0)
#                 if rtype != "none":
#                     rule.post = rtype
#                     rule.post_params = params
#                     rule.post_color = color
#                 elif detect_tiling(rule.apply(x0), y0):
#                     rule.post = "tile"
#                     rule.post_params = y0.shape
#             else:
#                 rule = Rule(bg0, d, cm, (0, 0))
#                 if detect_tiling(apply_dihedral(x0, d), y0):
#                     rule.post = "tile"
#                     rule.post_params = y0.shape
#                 else:
#                     rtype, params, color = detect_crop_or_border(
#                         apply_dihedral(x0, d), y0, bg0
#                     )
#                     if rtype != "none":
#                         rule.post = rtype
#                         rule.post_params = params
#                         rule.post_color = color
#             score = 0
#             for xi, yi in train_pairs:
#                 pred = rule.apply(xi)
#                 if pred.shape == yi.shape and np.array_equal(pred, yi):
#                     score += 1
#             candidates.append((rule, score))
#     candidates.sort(key=lambda rs: (-rs[1], rs[0].dihedral != "I", len(rs[0].colormap)))
#     best_rule, best_score = candidates[0]
#     second_rule = candidates[1][0] if len(candidates) > 1 else best_rule
#     return best_rule, second_rule, float(best_score)
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from arc_transforms import (
    apply_dihedral,
    DIHEDRAL_CODES,
    apply_color_map,
    translate,
    crop_to_bbox,
    add_border,
    tile_to_shape,
)


# ---------- Utility ----------
def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def infer_background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


# ---------- Color map inference ----------
def infer_color_mapping(inp: np.ndarray, out: np.ndarray) -> Dict[int, int]:
    """Cheap global color mapping by frequency ranking."""
    mapping: Dict[int, int] = {}
    vi, ci = np.unique(inp, return_counts=True)
    vo, co = np.unique(out, return_counts=True)
    order_i = [int(v) for _, v in sorted(zip(ci, vi), reverse=True)]
    order_o = [int(v) for _, v in sorted(zip(co, vo), reverse=True)]
    for a, b in zip(order_i, order_o):
        mapping[a] = b
    return mapping


# ---------- Geometric inference ----------
def best_dihedral(inp: np.ndarray, out: np.ndarray) -> str:
    best, bestScore = "I", -1
    for code in DIHEDRAL_CODES:
        g = apply_dihedral(inp, code)
        if g.shape != out.shape:
            continue
        score = (g == out).sum()
        if score > bestScore:
            bestScore, best = score, code
    return best


# ---------- Translation inference ----------
def best_translation(
    inp: np.ndarray, out: np.ndarray, background: int
) -> Tuple[int, int]:
    H, W = inp.shape
    best = (0, 0)
    bestScore = -1
    for dy in range(-H + 1, H):
        g_row = translate(inp, dy, 0, background)
        for dx in range(-W + 1, W):
            g = translate(g_row, 0, dx, background)
            score = (g == out).sum()
            if score > bestScore:
                bestScore, best = score, (dy, dx)
    return best


# ---------- Border/crop inference (FIXED) ----------
def detect_crop_or_border(
    inp: np.ndarray, out: np.ndarray, background: int
) -> Tuple[str, Tuple[int, int, int, int], int]:
    """
    Return ('none', _, _) or ('crop', bbox, _) or ('border', (top,left,bottom,right), color).
    Only tries borders when Ho>=H and Wo>=W; loops are exact 0..delta (inclusive) — no negative margins.
    """
    # crop-to-bbox
    cropped, bbox = crop_to_bbox(inp, background)
    if cropped.shape == out.shape and np.array_equal(cropped, out):
        r0, c0, r1, c1 = bbox
        return "crop", (r0, c0, r1, c1), 0

    H, W = inp.shape
    Ho, Wo = out.shape
    # border padding only makes sense if output is not smaller on either axis
    if Ho >= H and Wo >= W:
        color = infer_background_color(out)
        dH = Ho - H
        dW = Wo - W
        # exact inclusive ranges: 0..dH and 0..dW
        for t in range(0, dH + 1):
            for l in range(0, dW + 1):
                btm = dH - t
                rgt = dW - l
                b = add_border(inp, t, btm, l, rgt, color)
                # shapes now guaranteed to match if our arithmetic is right
                if b.shape == out.shape and np.array_equal(b, out):
                    return "border", (t, l, btm, rgt), color

    return "none", (0, 0, 0, 0), 0


# ---------- Tiling ----------
def detect_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
    h, w = inp.shape
    H, W = out.shape
    if H % h != 0 or W % w != 0:
        return False
    tiled = tile_to_shape(inp, H, W)
    return np.array_equal(tiled, out)


# ---------- Rule + inference (unchanged except they call detect_crop_or_border) ----------
class Rule:
    def __init__(
        self,
        background: int,
        dihedral: str,
        colormap: Dict[int, int],
        translation: Tuple[int, int],
        post: str = "none",
        post_params: Tuple = (),
        post_color: int = 0,
    ):
        self.background = int(background)
        self.dihedral = dihedral
        self.colormap = dict(colormap) if colormap else {}
        self.translation = translation  # (dy, dx)
        self.post = post  # 'none' | 'crop' | 'border' | 'tile'
        self.post_params = post_params  # bbox or border tuple or (H,W)
        self.post_color = int(post_color)

    def apply(self, grid: np.ndarray) -> np.ndarray:
        g = grid
        g = apply_dihedral(g, self.dihedral)
        if self.colormap:
            g = apply_color_map(g, self.colormap)
        if self.translation != (0, 0):
            g = translate(g, self.translation[0], self.translation[1], self.background)
        if self.post == "crop":
            r0, c0, r1, c1 = self.post_params
            g = g[r0:r1, c0:c1]
        elif self.post == "border":
            t, l, b, r = self.post_params
            g = add_border(g, t, b, l, r, self.post_color)
        elif self.post == "tile":
            H, W = self.post_params
            g = tile_to_shape(g, H, W)
        return g


def infer_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]], background: Optional[int] = None
):
    if background is None:
        background = int(
            np.bincount(np.concatenate([p[0].ravel() for p in train_pairs])).argmax()
        )
    x0, y0 = train_pairs[0]
    bg0 = background
    cand_colormap = infer_color_mapping(x0, y0)
    colormaps = [cand_colormap, {}]
    dihedrals = DIHEDRAL_CODES
    candidates: List[Tuple[Rule, int]] = []
    for cm in colormaps:
        for d in dihedrals:
            g0 = apply_dihedral(x0, d)
            if cm:
                g0 = apply_color_map(g0, cm)
            if g0.shape == y0.shape:
                dy, dx = best_translation(g0, y0, bg0)
                rule = Rule(bg0, d, cm, (dy, dx))
                rtype, params, color = detect_crop_or_border(rule.apply(x0), y0, bg0)
                if rtype != "none":
                    rule.post = rtype
                    rule.post_params = params
                    rule.post_color = color
                elif detect_tiling(rule.apply(x0), y0):
                    rule.post = "tile"
                    rule.post_params = y0.shape
            else:
                rule = Rule(bg0, d, cm, (0, 0))
                if detect_tiling(apply_dihedral(x0, d), y0):
                    rule.post = "tile"
                    rule.post_params = y0.shape
                else:
                    rtype, params, color = detect_crop_or_border(
                        apply_dihedral(x0, d), y0, bg0
                    )
                    if rtype != "none":
                        rule.post = rtype
                        rule.post_params = params
                        rule.post_color = color
            score = 0
            for xi, yi in train_pairs:
                pred = rule.apply(xi)
                if pred.shape == yi.shape and np.array_equal(pred, yi):
                    score += 1
            candidates.append((rule, score))
    candidates.sort(key=lambda rs: (-rs[1], rs[0].dihedral != "I", len(rs[0].colormap)))
    best_rule, best_score = candidates[0]
    second_rule = candidates[1][0] if len(candidates) > 1 else best_rule
    return best_rule, second_rule, float(best_score)
