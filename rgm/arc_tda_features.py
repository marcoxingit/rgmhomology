# # arc_tda_features.py (manual features + selectable matching metric)
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Tuple, Optional
# import numpy as np
# from scipy import ndimage as ndi

# from gtda.homology import CubicalPersistence

# # ---- runtime flag ----
# MATCH_METRIC = "custom"  # options: "custom", "feat-l2", "gtda-w"
# _WARNED_GTD = False


# def set_matching_metric(name: str):
#     """Set matching metric globally without changing call sites."""
#     global MATCH_METRIC
#     name = (name or "").strip().lower()
#     if name in ("custom", "feat-l2", "gtda-w"):
#         MATCH_METRIC = name
#     else:
#         raise ValueError(
#             f"Unknown match metric '{name}'. Choices: custom, feat-l2, gtda-w"
#         )


# @dataclass
# class ARCObject:
#     color: int
#     mask: np.ndarray
#     bbox: Tuple[int, int, int, int]
#     centroid_rc: Tuple[float, float]
#     diagram: Optional[np.ndarray] = None
#     features: Optional[np.ndarray] = None

#     def crop(self) -> np.ndarray:
#         r0, c0, r1, c1 = self.bbox
#         return self.mask[r0:r1, c0:c1]


# def _connected_components(mask: np.ndarray, connectivity: int = 1):
#     structure = ndi.generate_binary_structure(2, connectivity)
#     labeled, n = ndi.label(mask, structure=structure)
#     return labeled, n


# def extract_objects_from_grid(
#     grid: np.ndarray, background: int = 0, connectivity: int = 1
# ) -> List[ARCObject]:
#     H, W = grid.shape
#     objects: List[ARCObject] = []
#     colors = sorted(int(c) for c in np.unique(grid) if int(c) != background)
#     for color in colors:
#         layer = grid == color
#         labeled, n = _connected_components(layer, connectivity=connectivity)
#         for lab in range(1, n + 1):
#             m = labeled == lab
#             if not m.any():
#                 continue
#             rows, cols = np.where(m)
#             r0, r1 = rows.min(), rows.max() + 1
#             c0, c1 = cols.min(), cols.max() + 1
#             cy, cx = rows.mean(), cols.mean()
#             objects.append(
#                 ARCObject(
#                     color=color, mask=m, bbox=(r0, c0, r1, c1), centroid_rc=(cy, cx)
#                 )
#             )
#     return objects


# def _signed_distance_filtration(mask: np.ndarray) -> np.ndarray:
#     dist_in = ndi.distance_transform_edt(mask.astype(np.uint8))
#     dist_out = ndi.distance_transform_edt((~mask).astype(np.uint8))
#     return (dist_in - dist_out).astype(np.float32)


# def compute_diagram_for_mask(
#     mask: np.ndarray, homology_dimensions=(0, 1), reduced_homology=True
# ) -> np.ndarray:
#     cp = CubicalPersistence(
#         homology_dimensions=homology_dimensions, reduced_homology=reduced_homology
#     )
#     X = _signed_distance_filtration(mask)[None, ...]  # (1,H,W)
#     D = cp.fit_transform(X)
#     return D[0]  # (n_points, 3) with columns [birth, death, dim]


# # ------------------ Manual fixed-length feature vector ------------------
# def _diagram_stats(diag: np.ndarray, dims=(0, 1)) -> np.ndarray:
#     """Return a fixed-length feature vector summarizing the diagram for given homology dims.
#     Per dim: [count, sum_pers, max_pers, mean_pers, entropy, q25, q50, q75, mean_birth, mean_death] -> 10 features
#     Total length = 10 * len(dims)
#     """
#     features = []
#     for q in dims:
#         if diag is None or len(diag) == 0:
#             t = np.array([], dtype=float)
#             births = np.array([], dtype=float)
#             deaths = np.array([], dtype=float)
#         else:
#             sel = diag[:, 2] == q
#             pts = diag[sel]
#             if pts.size == 0:
#                 t = np.array([], dtype=float)
#                 births = np.array([], dtype=float)
#                 deaths = np.array([], dtype=float)
#             else:
#                 births = pts[:, 0].astype(float)
#                 deaths = pts[:, 1].astype(float)
#                 t = np.maximum(deaths - births, 0.0)
#         n = float(len(t))
#         s = float(t.sum())
#         mx = float(t.max()) if n > 0 else 0.0
#         mean = float(s / n) if n > 0 else 0.0
#         if s > 0:
#             p = t / s
#             ent = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
#         else:
#             ent = 0.0
#         if n > 0:
#             q25, q50, q75 = [float(x) for x in np.quantile(t, [0.25, 0.5, 0.75])]
#             mb = float(births.mean())
#             md = float(deaths.mean())
#         else:
#             q25 = q50 = q75 = 0.0
#             mb = md = 0.0
#         features.extend([n, s, mx, mean, ent, q25, q50, q75, mb, md])
#     return np.asarray(features, dtype=np.float32)


# def batch_vectorize_diagrams(diagrams: List[np.ndarray]) -> np.ndarray:
#     """Manual, always-fixed-length features."""
#     if len(diagrams) == 0:
#         return np.zeros((0, 20), dtype=np.float32)
#     feats = [_diagram_stats(Di, dims=(0, 1)) for Di in diagrams]
#     return np.stack(feats, axis=0)  # (N, 20)


# # ------------------ Custom diagram distance (fast) ------------------
# def _topk_persistences(diag: np.ndarray, dim: int, K: int = 12) -> np.ndarray:
#     if diag is None or len(diag) == 0:
#         return np.zeros((K,), dtype=float)
#     pts = diag[diag[:, 2] == dim]
#     if pts.size == 0:
#         return np.zeros((K,), dtype=float)
#     pers = np.maximum(pts[:, 1] - pts[:, 0], 0.0).astype(float)
#     pers.sort()
#     pers = pers[::-1]  # desc
#     if len(pers) >= K:
#         return pers[:K]
#     out = np.zeros((K,), dtype=float)
#     out[: len(pers)] = pers
#     return out


# def _diagram_distance_custom(
#     diA: np.ndarray, diB: np.ndarray, dims=(0, 1), K: int = 12, w_stats: float = 0.5
# ) -> float:
#     """Distance combining top-K persistence profiles and summary stats."""
#     vecs = []
#     for q in dims:
#         a = _topk_persistences(diA, q, K)
#         b = _topk_persistences(diB, q, K)
#         vecs.append(np.abs(a - b).sum())
#     d_topk = float(sum(vecs))
#     sa = _diagram_stats(diA, dims=dims)
#     sb = _diagram_stats(diB, dims=dims)
#     d_stats = float(np.linalg.norm(sa - sb, ord=1))
#     return d_topk + w_stats * d_stats


# # ------------------ Matching ------------------
# def _distance_matrix_custom(objs_A, objs_B):
#     nA, nB = len(objs_A), len(objs_B)
#     D = np.zeros((nA, nB), dtype=float)
#     for i, oa in enumerate(objs_A):
#         for j, ob in enumerate(objs_B):
#             D[i, j] = _diagram_distance_custom(oa.diagram, ob.diagram)
#     return D


# def _distance_matrix_feat_l2(objs_A, objs_B):
#     nA, nB = len(objs_A), len(objs_B)
#     D = np.zeros((nA, nB), dtype=float)
#     for i, oa in enumerate(objs_A):
#         for j, ob in enumerate(objs_B):
#             D[i, j] = float(np.linalg.norm((oa.features - ob.features), ord=2))
#     return D


# # def _distance_matrix_gtda_w(objs_A, objs_B):
# #     """Try giotto-tda PairwiseDistance; fall back to custom on any error."""
# #     global _WARNED_GTD
# #     try:
# #         from gtda.diagrams import PairwiseDistance
# #         Di_A = [o.diagram for o in objs_A]
# #         Di_B = [o.diagram for o in objs_B]
# #         Di = list(Di_A + Di_B)  # python list; gtda will attempt to handle
# #         pwd = PairwiseDistance(metric="wasserstein", metric_params={"p": 1}, n_jobs=-1)
# #         D_all = pwd.fit_transform(Di)  # may raise on some builds
# #         if D_all.ndim == 3:
# #             D_all = D_all.mean(axis=2)
# #         nA = len(Di_A)
# #         return D_all[:nA, nA:]
# #     except Exception as e:
# #         if not _WARNED_GTD:
# #             print(f"[arc_tda_features] Warning: gtda-w failed ({e}). Falling back to custom distance.", flush=True)
# #             _WARNED_GTD = True
# #         return _distance_matrix_custom(objs_A, objs_B)


# def _distance_matrix_gtda_w(objs_A, objs_B):
#     """Compute true Wasserstein via giotto per pair (avoids ragged batch)."""
#     global _WARNED_GTD
#     try:
#         try:
#             # preferred public path
#             from gtda.diagrams.metrics import wasserstein_distance as wdist
#         except Exception:
#             # fallback to private path (works in some versions)
#             from gtda.diagrams._metrics import wasserstein_distance as wdist
#         nA, nB = len(objs_A), len(objs_B)
#         D = np.zeros((nA, nB), dtype=float)
#         for i, oa in enumerate(objs_A):
#             for j, ob in enumerate(objs_B):
#                 # order=1 is standard; adjust if you want p=2, etc.
#                 D[i, j] = float(wdist(oa.diagram, ob.diagram, order=1))
#         return D
#     except Exception as e:
#         if not _WARNED_GTD:
#             print(
#                 f"[arc_tda_features] Warning: gtda-w pairwise failed ({e}). Falling back to custom.",
#                 flush=True,
#             )
#             _WARNED_GTD = True
#         return _distance_matrix_custom(objs_A, objs_B)


# def wasserstein_matching(
#     objs_A: List[ARCObject],
#     objs_B: List[ARCObject],
#     order: int = 1,
#     return_matrix: bool = False,
# ):
#     """Match objects by minimizing a selectable diagram distance + Hungarian assignment."""
#     if len(objs_A) == 0 or len(objs_B) == 0:
#         return ([], None) if return_matrix else ([], None)
#     if MATCH_METRIC == "custom":
#         D = _distance_matrix_custom(objs_A, objs_B)
#     elif MATCH_METRIC == "feat-l2":
#         D = _distance_matrix_feat_l2(objs_A, objs_B)
#     elif MATCH_METRIC == "gtda-w":
#         D = _distance_matrix_gtda_w(objs_A, objs_B)
#     else:
#         D = _distance_matrix_custom(objs_A, objs_B)
#     from scipy.optimize import linear_sum_assignment

#     if D.size == 0:
#         return ([], None) if return_matrix else ([], None)
#     rows, cols = linear_sum_assignment(D)
#     matches = list(zip(rows.tolist(), cols.tolist()))
#     return (matches, D) if return_matrix else (matches, None)


# # ------------------ High-level helpers ------------------
# def enrich_objects_with_tda(
#     objs: List[ARCObject], homology_dimensions=(0, 1)
# ) -> List[ARCObject]:
#     if not objs:
#         return objs
#     for o in objs:
#         o.diagram = compute_diagram_for_mask(
#             o.mask, homology_dimensions=homology_dimensions
#         )
#     feats = batch_vectorize_diagrams([o.diagram for o in objs])
#     for o, f in zip(objs, feats):
#         o.features = f
#     return objs


# def grid_to_objects_with_tda(
#     grid: np.ndarray,
#     background: int = 0,
#     connectivity: int = 1,
#     homology_dimensions=(0, 1),
# ) -> List[ARCObject]:
#     objs = extract_objects_from_grid(
#         grid, background=background, connectivity=connectivity
#     )
#     return enrich_objects_with_tda(objs, homology_dimensions=homology_dimensions)
# arc_tda_features.py
# (manual features + selectable matching metric + pairwise giotto Wasserstein)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage as ndi

from gtda.homology import CubicalPersistence

# ---- runtime flag ----
MATCH_METRIC = "custom"  # options: "custom", "feat-l2", "gtda-w"
_WARNED_GTD = False


def set_matching_metric(name: str):
    """Set matching metric globally without changing call sites."""
    global MATCH_METRIC
    name = (name or "").strip().lower()
    if name in ("custom", "feat-l2", "gtda-w"):
        MATCH_METRIC = name
    else:
        raise ValueError(
            f"Unknown match metric '{name}'. Choices: custom, feat-l2, gtda-w"
        )


@dataclass
class ARCObject:
    color: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid_rc: Tuple[float, float]
    diagram: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

    def crop(self) -> np.ndarray:
        r0, c0, r1, c1 = self.bbox
        return self.mask[r0:r1, c0:c1]


def _connected_components(mask: np.ndarray, connectivity: int = 1):
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(mask, structure=structure)
    return labeled, n


def extract_objects_from_grid(
    grid: np.ndarray, background: int = 0, connectivity: int = 1
) -> List[ARCObject]:
    H, W = grid.shape
    objects: List[ARCObject] = []
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != background)
    for color in colors:
        layer = grid == color
        labeled, n = _connected_components(layer, connectivity=connectivity)
        for lab in range(1, n + 1):
            m = labeled == lab
            if not m.any():
                continue
            rows, cols = np.where(m)
            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1
            cy, cx = rows.mean(), cols.mean()
            objects.append(
                ARCObject(
                    color=color, mask=m, bbox=(r0, c0, r1, c1), centroid_rc=(cy, cx)
                )
            )
    return objects


def _signed_distance_filtration(mask: np.ndarray) -> np.ndarray:
    dist_in = ndi.distance_transform_edt(mask.astype(np.uint8))
    dist_out = ndi.distance_transform_edt((~mask).astype(np.uint8))
    return (dist_in - dist_out).astype(np.float32)


def compute_diagram_for_mask(
    mask: np.ndarray, homology_dimensions=(0, 1), reduced_homology=True
) -> np.ndarray:
    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions, reduced_homology=reduced_homology
    )
    X = _signed_distance_filtration(mask)[None, ...]  # (1,H,W)
    D = cp.fit_transform(X)
    return D[0]  # (n_points, 3) with columns [birth, death, dim]


# ------------------ Manual fixed-length feature vector ------------------
def _diagram_stats(diag: np.ndarray, dims=(0, 1)) -> np.ndarray:
    """Per dim: [count, sum_pers, max_pers, mean_pers, entropy, q25, q50, q75, mean_birth, mean_death] -> 10 features"""
    features = []
    for q in dims:
        if diag is None or len(diag) == 0:
            t = np.array([], dtype=float)
            births = np.array([], dtype=float)
            deaths = np.array([], dtype=float)
        else:
            sel = diag[:, 2] == q
            pts = diag[sel]
            if pts.size == 0:
                t = np.array([], dtype=float)
                births = np.array([], dtype=float)
                deaths = np.array([], dtype=float)
            else:
                births = pts[:, 0].astype(float)
                deaths = pts[:, 1].astype(float)
                t = np.maximum(deaths - births, 0.0)
        n = float(len(t))
        s = float(t.sum())
        mx = float(t.max()) if n > 0 else 0.0
        mean = float(s / n) if n > 0 else 0.0
        if s > 0:
            p = t / s
            ent = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
        else:
            ent = 0.0
        if n > 0:
            q25, q50, q75 = [float(x) for x in np.quantile(t, [0.25, 0.5, 0.75])]
            mb = float(births.mean())
            md = float(deaths.mean())
        else:
            q25 = q50 = q75 = 0.0
            mb = md = 0.0
        features.extend([n, s, mx, mean, ent, q25, q50, q75, mb, md])
    return np.asarray(features, dtype=np.float32)


def batch_vectorize_diagrams(diagrams: List[np.ndarray]) -> np.ndarray:
    """Manual, always-fixed-length features."""
    if len(diagrams) == 0:
        return np.zeros((0, 20), dtype=np.float32)
    feats = [_diagram_stats(Di, dims=(0, 1)) for Di in diagrams]
    return np.stack(feats, axis=0)  # (N, 20)


# ------------------ Custom diagram distance (fast) ------------------
def _topk_persistences(diag: np.ndarray, dim: int, K: int = 12) -> np.ndarray:
    if diag is None or len(diag) == 0:
        return np.zeros((K,), dtype=float)
    pts = diag[diag[:, 2] == dim]
    if pts.size == 0:
        return np.zeros((K,), dtype=float)
    pers = np.maximum(pts[:, 1] - pts[:, 0], 0.0).astype(float)
    pers.sort()
    pers = pers[::-1]  # desc
    if len(pers) >= K:
        return pers[:K]
    out = np.zeros((K,), dtype=float)
    out[: len(pers)] = pers
    return out


def _diagram_distance_custom(
    diA: np.ndarray, diB: np.ndarray, dims=(0, 1), K: int = 12, w_stats: float = 0.5
) -> float:
    """Distance combining top-K persistence profiles and summary stats."""
    vecs = []
    for q in dims:
        a = _topk_persistences(diA, q, K)
        b = _topk_persistences(diB, q, K)
        vecs.append(np.abs(a - b).sum())
    d_topk = float(sum(vecs))
    sa = _diagram_stats(diA, dims=dims)
    sb = _diagram_stats(diB, dims=dims)
    d_stats = float(np.linalg.norm(sa - sb, ord=1))
    return d_topk + w_stats * d_stats


# ------------------ Matching ------------------
def _distance_matrix_custom(objs_A, objs_B):
    nA, nB = len(objs_A), len(objs_B)
    D = np.zeros((nA, nB), dtype=float)
    for i, oa in enumerate(objs_A):
        for j, ob in enumerate(objs_B):
            D[i, j] = _diagram_distance_custom(oa.diagram, ob.diagram)
    return D


def _distance_matrix_feat_l2(objs_A, objs_B):
    nA, nB = len(objs_A), len(objs_B)
    D = np.zeros((nA, nB), dtype=float)
    for i, oa in enumerate(objs_A):
        for j, ob in enumerate(objs_B):
            D[i, j] = float(np.linalg.norm((oa.features - ob.features), ord=2))
    return D


def _distance_matrix_gtda_w(objs_A, objs_B):
    """Compute true Wasserstein via giotto **pairwise** (avoids ragged batch)."""
    global _WARNED_GTD
    try:
        try:
            from gtda.diagrams.metrics import wasserstein_distance as wdist
        except Exception:
            from gtda.diagrams._metrics import (
                wasserstein_distance as wdist,
            )  # fallback for some versions
        nA, nB = len(objs_A), len(objs_B)
        D = np.zeros((nA, nB), dtype=float)
        for i, oa in enumerate(objs_A):
            for j, ob in enumerate(objs_B):
                D[i, j] = float(wdist(oa.diagram, ob.diagram, order=1))
        return D
    except Exception as e:
        if not _WARNED_GTD:
            print(
                f"[arc_tda_features] Warning: gtda-w pairwise failed ({e}). Falling back to custom.",
                flush=True,
            )
            _WARNED_GTD = True
        return _distance_matrix_custom(objs_A, objs_B)


def wasserstein_matching(
    objs_A: List[ARCObject],
    objs_B: List[ARCObject],
    order: int = 1,
    return_matrix: bool = False,
):
    """Match objects by minimizing a selectable diagram distance + Hungarian assignment."""
    if len(objs_A) == 0 or len(objs_B) == 0:
        return ([], None) if return_matrix else ([], None)
    if MATCH_METRIC == "custom":
        D = _distance_matrix_custom(objs_A, objs_B)
    elif MATCH_METRIC == "feat-l2":
        D = _distance_matrix_feat_l2(objs_A, objs_B)
    elif MATCH_METRIC == "gtda-w":
        D = _distance_matrix_gtda_w(objs_A, objs_B)
    else:
        D = _distance_matrix_custom(objs_A, objs_B)
    from scipy.optimize import linear_sum_assignment

    if D.size == 0:
        return ([], None) if return_matrix else ([], None)
    rows, cols = linear_sum_assignment(D)
    matches = list(zip(rows.tolist(), cols.tolist()))
    return (matches, D) if return_matrix else (matches, None)


# ------------------ High-level helpers ------------------
def enrich_objects_with_tda(
    objs: List[ARCObject], homology_dimensions=(0, 1)
) -> List[ARCObject]:
    if not objs:
        return objs
    for o in objs:
        o.diagram = compute_diagram_for_mask(
            o.mask, homology_dimensions=homology_dimensions
        )
    feats = batch_vectorize_diagrams([o.diagram for o in objs])
    for o, f in zip(objs, feats):
        o.features = f
    return objs


def grid_to_objects_with_tda(
    grid: np.ndarray,
    background: int = 0,
    connectivity: int = 1,
    homology_dimensions=(0, 1),
) -> List[ARCObject]:
    objs = extract_objects_from_grid(
        grid, background=background, connectivity=connectivity
    )
    return enrich_objects_with_tda(objs, homology_dimensions=homology_dimensions)
