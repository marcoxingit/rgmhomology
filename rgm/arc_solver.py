
# arc_solver.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
from collections import Counter

from arc_tda_features import (
    ARCObject,
    grid_to_objects_with_tda,
    wasserstein_matching,
    extract_objects_from_grid,
)

Grid = np.ndarray

@dataclass
class MatchRule:
    translation: Tuple[int, int] = (0, 0)
    rotation_k: int = 0
    flip_h: bool = False
    flip_v: bool = False
    recolor_to: Optional[int] = None

@dataclass
class GlobalRule:
    translation: Tuple[int, int] = (0, 0)
    rotation_k: int = 0
    flip_h: bool = False
    flip_v: bool = False
    recolor_map: Dict[int, int] = None
    add_border: Optional[Tuple[int, int, int, int, int]] = None
    output_resize: Optional[Tuple[int, int]] = None

def load_arc_task_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def grid_from_list(lst: List[List[int]]) -> Grid:
    return np.array(lst, dtype=int)

def grid_to_list(grid: Grid) -> List[List[int]]:
    return grid.astype(int).tolist()

def detect_border_change(inp: Grid, out: Grid) -> Optional[Tuple[int, int, int, int, int]]:
    Hi, Wi = inp.shape
    Ho, Wo = out.shape
    if Ho < Hi or Wo < Wi:
        return None
    top = (Ho - Hi) // 2
    left = (Wo - Wi) // 2
    bottom = Ho - Hi - top
    right = Wo - Wi - left
    color = int(np.bincount(out.ravel()).argmax())
    ok = True
    if top>0: ok &= np.all(out[:top, :] == color)
    if bottom>0: ok &= np.all(out[Ho-bottom:, :] == color)
    if left>0: ok &= np.all(out[:, :left] == color)
    if right>0: ok &= np.all(out[:, Wo-right:] == color)
    if ok:
        inner = out[top:Ho-bottom if bottom>0 else Ho, left:Wo-right if right>0 else Wo]
        if inner.shape == inp.shape and np.array_equal(inner, inp):
            return (top, bottom, left, right, color)
    return None

def apply_border(grid: Grid, border: Tuple[int,int,int,int,int]) -> Grid:
    top,bottom,left,right,color = border
    H,W = grid.shape
    out = np.full((H+top+bottom, W+left+right), color, dtype=int)
    out[top:top+H, left:left+W] = grid
    return out

def crop_bbox(mask: np.ndarray) -> Tuple[int,int,int,int]:
    rows, cols = np.where(mask)
    return rows.min(), cols.min(), rows.max()+1, cols.max()+1

def transform_mask(mask: np.ndarray, rotation_k:int=0, flip_h=False, flip_v=False) -> np.ndarray:
    m = mask.copy()
    if flip_h: m = np.fliplr(m)
    if flip_v: m = np.flipud(m)
    if rotation_k % 4 != 0: m = np.rot90(m, k=rotation_k)
    return m

def place_mask(canvas: Grid, mask: np.ndarray, color: int, top_left: Tuple[int,int]) -> None:
    H,W = canvas.shape
    mh, mw = mask.shape
    r0,c0 = top_left
    r1,c1 = r0+mh, c0+mw
    if r0<0 or c0<0 or r1>H or c1>W:
        rr0 = max(r0,0); cc0=max(c0,0); rr1=min(r1,H); cc1=min(c1,W)
        mr0 = rr0 - r0; mc0 = cc0 - c0; mr1 = mr0 + (rr1-rr0); mc1 = mc0 + (cc1-cc0)
        if rr1>rr0 and cc1>cc0:
            canvas[rr0:rr1, cc0:cc1] = np.where(mask[mr0:mr1, mc0:mc1], color, canvas[rr0:rr1, cc0:cc1])
    else:
        canvas[r0:r1, c0:c1] = np.where(mask, color, canvas[r0:r1, c0:c1])

def _mode_seq(seq, default):
    if not seq:
        return default
    from collections import Counter
    return Counter(seq).most_common(1)[0][0]

def infer_per_pair_rule(inp: Grid, out: Grid, background:int=0):
    border = detect_border_change(inp, out)
    output_resize = out.shape if inp.shape != out.shape and border is None else None

    objs_in = grid_to_objects_with_tda(inp, background=background)
    objs_out = grid_to_objects_with_tda(out, background=background)
    if len(objs_in)==0 or len(objs_out)==0:
        return GlobalRule(recolor_map={}, add_border=border, output_resize=output_resize), []

    matches, D = wasserstein_matching(objs_in, objs_out, return_matrix=True)
    if not matches:
        return GlobalRule(recolor_map={}, add_border=border, output_resize=output_resize), []

    results = []
    translations = []
    rotations = []
    flips_h, flips_v = [], []
    recolor_map: Dict[int,int] = {}

    for i,j in matches:
        oi, oj = objs_in[i], objs_out[j]
        dy = int(round(oj.centroid_rc[0] - oi.centroid_rc[0]))
        dx = int(round(oj.centroid_rc[1] - oi.centroid_rc[1]))
        translations.append((dy, dx))

        ri0, ci0, ri1, ci1 = crop_bbox(oi.mask)
        rj0, cj0, rj1, cj1 = crop_bbox(oj.mask)
        mi = oi.mask[ri0:ri1, ci0:ci1]
        mj = oj.mask[rj0:rj1, cj0:cj1]

        best = (0, False, False, 10**9)
        for fh in (False, True):
            for fv in (False, True):
                for rk in (0,1,2,3):
                    mti = transform_mask(mi, rotation_k=rk, flip_h=fh, flip_v=fv)
                    if mti.shape != mj.shape: 
                        continue
                    err = np.sum(mti ^ mj)
                    if err < best[3]:
                        best = (rk, fh, fv, err)
        rk, fh, fv, _ = best
        rotations.append(rk); flips_h.append(fh); flips_v.append(fv)

        if oi.color != oj.color:
            recolor_map[oi.color] = oj.color

        results.append((oi, oj, MatchRule(translation=(dy,dx), rotation_k=rk, flip_h=fh, flip_v=fv, recolor_to=recolor_map.get(oi.color))))

    glob = GlobalRule(
        translation=_mode_seq(translations, (0,0)),
        rotation_k=_mode_seq(rotations, 0),
        flip_h=_mode_seq(flips_h, False),
        flip_v=_mode_seq(flips_v, False),
        recolor_map=recolor_map,
        add_border=border,
        output_resize=output_resize,
    )
    return glob, results

def combine_rules(rules: List[GlobalRule]) -> GlobalRule:
    if not rules:
        return GlobalRule(recolor_map={})
    trans = [r.translation for r in rules if r.translation is not None]
    rots  = [r.rotation_k for r in rules if r.rotation_k is not None]
    fhs   = [r.flip_h for r in rules]
    fvs   = [r.flip_v for r in rules]
    borders = [r.add_border for r in rules if r.add_border is not None]
    resizes = [r.output_resize for r in rules if r.output_resize is not None]
    recolor_map: Dict[int,int] = {}
    for r in rules:
        if r.recolor_map:
            recolor_map.update(r.recolor_map)
    return GlobalRule(
        translation=_mode_seq(trans, (0,0)),
        rotation_k=_mode_seq(rots, 0),
        flip_h=_mode_seq(fhs, False),
        flip_v=_mode_seq(fvs, False),
        recolor_map=recolor_map,
        add_border=_mode_seq(borders, None) if borders else None,
        output_resize=_mode_seq(resizes, None) if resizes else None
    )

def apply_rule_to_grid(inp: Grid, rule: GlobalRule, background:int=0) -> Grid:
    grid = inp.copy()
    if rule.add_border is not None:
        grid = apply_border(grid, rule.add_border)
    objs = extract_objects_from_grid(inp, background=background)
    if rule.output_resize is not None and rule.add_border is None:
        outH, outW = rule.output_resize
    else:
        outH, outW = grid.shape
    out = np.full((outH, outW), background, dtype=int)
    dy, dx = rule.translation
    for o in objs:
        color = rule.recolor_map.get(o.color, o.color) if rule.recolor_map else o.color
        r0,c0,r1,c1 = crop_bbox(o.mask)
        m = o.mask[r0:r1, c0:c1]
        m = transform_mask(m, rotation_k=rule.rotation_k, flip_h=rule.flip_h, flip_v=rule.flip_v)
        top_left = (r0 + dy, c0 + dx)
        place_mask(out, m, color, top_left)
    return out

class ARCSolver:
    def __init__(self, background:int=0):
        self.background = background

    def solve_task(self, task: Dict[str, Any]) -> List[Grid]:
        train_pairs = [(grid_from_list(p["input"]), grid_from_list(p["output"])) for p in task["train"]]
        test_inputs = [grid_from_list(t["input"]) for t in task["test"]]
        per_pair_rules = []
        for inp, out in train_pairs:
            rule, _ = infer_per_pair_rule(inp, out, background=self.background)
            per_pair_rules.append(rule)
        global_rule = combine_rules(per_pair_rules)
        preds = [apply_rule_to_grid(ti, global_rule, background=self.background) for ti in test_inputs]
        return preds
