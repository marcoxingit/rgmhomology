# # arc_actions.py
# # Discrete "actions" implemented as simple transforms on binary masks.
# # Start minimal: identity + translations in {-1,0,1} x {-1,0,1} (9 actions).

# from __future__ import annotations
# from typing import Tuple, List
# import numpy as np

# # Action IDs and their (dy, dx) effect on a mask
# # Index 0 is identity; 1..8 are the 8 neighbors
# TRANSLATION_3x3: List[Tuple[int, int]] = [
#     (0, 0),
#     (-1, 0),
#     (1, 0),
#     (0, -1),
#     (0, 1),
#     (-1, -1),
#     (-1, 1),
#     (1, -1),
#     (1, 1),
# ]


# def action_count() -> int:
#     return len(TRANSLATION_3x3)


# def apply_action_mask(mask: np.ndarray, action_id: int, H: int, W: int) -> np.ndarray:
#     """Return a new HxW mask after applying the discrete action to 'mask' with clipping."""
#     dy, dx = TRANSLATION_3x3[action_id]
#     out = np.zeros((H, W), dtype=bool)
#     ys, xs = np.where(mask)
#     if ys.size == 0:
#         return out
#     y2 = ys + dy
#     x2 = xs + dx
#     keep = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
#     out[y2[keep], x2[keep]] = True
#     return out


# def best_action_by_iou(src_mask: np.ndarray, tgt_mask: np.ndarray) -> int:
#     """Pick the action with max IoU between transformed src_mask and tgt_mask."""
#     H, W = tgt_mask.shape
#     best_a, best_iou = 0, -1.0
#     tgt = tgt_mask.astype(bool)
#     for a in range(action_count()):
#         pred = apply_action_mask(src_mask, a, H, W)
#         inter = np.logical_and(pred, tgt).sum()
#         union = np.logical_or(pred, tgt).sum()
#         iou = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
#         if iou > best_iou:
#             best_iou, best_a = iou, a
#     return best_a
# rgm/arc_actions.py
# Discrete "actions" as ARC-friendly transforms on binary masks.
# Default library: D4 dihedrals (8) × translations in {-r..r}^2 (e.g., r=1 → 9) = 72 actions.

from __future__ import annotations
from typing import Tuple, List
import numpy as np

# ----- dihedral helpers -----


# d in [0..7]: 0=I, 1=R90, 2=R180, 3=R270, 4=FlipH, 5=FlipV, 6=FlipMainDiag, 7=FlipAntiDiag
def apply_dihedral_mask(mask: np.ndarray, d: int) -> np.ndarray:
    m = mask
    if d == 0:  # I
        return m
    elif d == 1:  # R90
        return np.rot90(m, k=1)
    elif d == 2:  # R180
        return np.rot90(m, k=2)
    elif d == 3:  # R270
        return np.rot90(m, k=3)
    elif d == 4:  # FlipH
        return np.flip(m, axis=1)
    elif d == 5:  # FlipV
        return np.flip(m, axis=0)
    elif d == 6:  # Flip main diagonal
        return np.transpose(m)
    elif d == 7:  # Flip anti-diagonal (transpose + rot180 is equivalent)
        return np.flip(np.transpose(m), axis=0)
    else:
        raise ValueError(f"bad dihedral code {d}")


def translation_grid(radius: int) -> List[Tuple[int, int]]:
    rng = range(-radius, radius + 1)
    return [(dy, dx) for dy in rng for dx in rng]


# ----- action library -----


class ActionLibrary:
    def __init__(self, trans_radius: int = 1):
        self.trans_radius = int(trans_radius)
        self._translations = translation_grid(self.trans_radius)
        self._actions: List[Tuple[int, Tuple[int, int]]] = []
        for d in range(8):  # D4
            for dy, dx in self._translations:
                self._actions.append((d, (dy, dx)))

    @property
    def count(self) -> int:
        return len(self._actions)

    def action_spec(self, a: int) -> Tuple[int, Tuple[int, int]]:
        return self._actions[a]

    def apply_action_mask(
        self, src_mask: np.ndarray, a: int, H: int, W: int
    ) -> np.ndarray:
        d, (dy, dx) = self.action_spec(a)
        m = apply_dihedral_mask(src_mask, d)
        out = np.zeros((H, W), dtype=bool)
        ys, xs = np.where(m)
        if ys.size == 0:
            return out
        y2 = ys + dy
        x2 = xs + dx
        keep = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        out[y2[keep], x2[keep]] = True
        return out

    def best_action_by_iou(self, src_mask: np.ndarray, tgt_mask: np.ndarray) -> int:
        H, W = tgt_mask.shape
        tgt = tgt_mask.astype(bool)
        best_a, best_iou = 0, -1.0
        for a in range(self.count):
            pred = self.apply_action_mask(src_mask, a, H, W)
            inter = np.logical_and(pred, tgt).sum()
            union = np.logical_or(pred, tgt).sum()
            iou = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
            if iou > best_iou:
                best_iou, best_a = iou, a
        return best_a
