# arc_actions.py
# Discrete "actions" implemented as simple transforms on binary masks.
# Start minimal: identity + translations in {-1,0,1} x {-1,0,1} (9 actions).

from __future__ import annotations
from typing import Tuple, List
import numpy as np

# Action IDs and their (dy, dx) effect on a mask
# Index 0 is identity; 1..8 are the 8 neighbors
TRANSLATION_3x3: List[Tuple[int, int]] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def action_count() -> int:
    return len(TRANSLATION_3x3)


def apply_action_mask(mask: np.ndarray, action_id: int, H: int, W: int) -> np.ndarray:
    """Return a new HxW mask after applying the discrete action to 'mask' with clipping."""
    dy, dx = TRANSLATION_3x3[action_id]
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return out
    y2 = ys + dy
    x2 = xs + dx
    keep = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
    out[y2[keep], x2[keep]] = True
    return out


def best_action_by_iou(src_mask: np.ndarray, tgt_mask: np.ndarray) -> int:
    """Pick the action with max IoU between transformed src_mask and tgt_mask."""
    H, W = tgt_mask.shape
    best_a, best_iou = 0, -1.0
    tgt = tgt_mask.astype(bool)
    for a in range(action_count()):
        pred = apply_action_mask(src_mask, a, H, W)
        inter = np.logical_and(pred, tgt).sum()
        union = np.logical_or(pred, tgt).sum()
        iou = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
        if iou > best_iou:
            best_iou, best_a = iou, a
    return best_a
