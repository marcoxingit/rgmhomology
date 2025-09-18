
# arc_tda_features.py (manual features + custom diagram distance; no gtda PairwiseDistance)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage as ndi

from gtda.homology import CubicalPersistence

@dataclass
class ARCObject:
    color: int
    mask: np.ndarray
    bbox: Tuple[int,int,int,int]
    centroid_rc: Tuple[float,float]
    diagram: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

    def crop(self) -> np.ndarray:
        r0,c0,r1,c1 = self.bbox
        return self.mask[r0:r1, c0:c1]

def _connected_components(mask: np.ndarray, connectivity: int = 1):
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(mask, structure=structure)
    return labeled, n

def extract_objects_from_grid(grid: np.ndarray, background: int = 0, connectivity: int = 1) -> List[ARCObject]:
    H, W = grid.shape
    objects: List[ARCObject] = []
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != background)
    for color in colors:
        layer = (grid == color)
        labeled, n = _connected_components(layer, connectivity=connectivity)
        for lab in range(1, n+1):
            m = (labeled == lab)
            if not m.any():
                continue
            rows, cols = np.where(m)
            r0, r1 = rows.min(), rows.max()+1
            c0, c1 = cols.min(), cols.max()+1
            cy, cx = rows.mean(), cols.mean()
            objects.append(ARCObject(color=color, mask=m, bbox=(r0,c0,r1,c1), centroid_rc=(cy,cx)))
    return objects

def _signed_distance_filtration(mask: np.ndarray) -> np.ndarray:
    dist_in = ndi.distance_transform_edt(mask.astype(np.uint8))
    dist_out = ndi.distance_transform_edt((~mask).astype(np.uint8))
    return (dist_in - dist_out).astype(np.float32)

def compute_diagram_for_mask(mask: np.ndarray, homology_dimensions=(0,1), reduced_homology=True) -> np.ndarray:
    cp = CubicalPersistence(homology_dimensions=homology_dimensions, reduced_homology=reduced_homology)
    X = _signed_distance_filtration(mask)[None, ...]  # (1,H,W)
    D = cp.fit_transform(X)
    return D[0]  # (n_points, 3) with columns [birth, death, dim]

# ------------------ Manual fixed-length feature vector ------------------
def _diagram_stats(diag: np.ndarray, dims=(0,1)) -> np.ndarray:
    """Return a fixed-length feature vector summarizing the diagram for given homology dims.
    Per dim: [count, sum_pers, max_pers, mean_pers, entropy, q25, q50, q75, mean_birth, mean_death] -> 10 features
    Total length = 10 * len(dims)
    """
    features = []
    for q in dims:
        if diag is None or len(diag) == 0:
            t = np.array([], dtype=float)
            births = np.array([], dtype=float)
            deaths = np.array([], dtype=float)
        else:
            sel = (diag[:, 2] == q)
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
    feats = [_diagram_stats(Di, dims=(0,1)) for Di in diagrams]
    return np.stack(feats, axis=0)  # (N, 20)

# ------------------ Custom diagram distance (no gtda PairwiseDistance) ------------------
def _topk_persistences(diag: np.ndarray, dim: int, K: int = 12) -> np.ndarray:
    if diag is None or len(diag) == 0:
        return np.zeros((K,), dtype=float)
    pts = diag[diag[:,2] == dim]
    if pts.size == 0:
        return np.zeros((K,), dtype=float)
    pers = np.maximum(pts[:,1] - pts[:,0], 0.0).astype(float)
    pers.sort()
    pers = pers[::-1]  # desc
    if len(pers) >= K:
        return pers[:K]
    out = np.zeros((K,), dtype=float)
    out[:len(pers)] = pers
    return out

def diagram_distance(diA: np.ndarray, diB: np.ndarray, dims=(0,1), K:int=12, w_stats: float=0.5) -> float:
    """A simple, fast distance combining top-K persistence profiles and summary stats."""
    vecs = []
    for q in dims:
        a = _topk_persistences(diA, q, K)
        b = _topk_persistences(diB, q, K)
        vecs.append(np.abs(a - b).sum())
    d_topk = float(sum(vecs))
    # add difference of summary stats for stability
    sa = _diagram_stats(diA, dims=dims)
    sb = _diagram_stats(diB, dims=dims)
    d_stats = float(np.linalg.norm(sa - sb, ord=1))
    return d_topk + w_stats * d_stats

# ------------------ Matching ------------------
def wasserstein_matching(objs_A: List[ARCObject], objs_B: List[ARCObject], order: int = 1, return_matrix: bool = False):
    """Match objects by minimizing a custom diagram distance (approx Wasserstein-like)."""
    if len(objs_A) == 0 or len(objs_B) == 0:
        return ([], None) if return_matrix else ([], None)
    nA, nB = len(objs_A), len(objs_B)
    D = np.zeros((nA, nB), dtype=float)
    for i, oa in enumerate(objs_A):
        for j, ob in enumerate(objs_B):
            D[i,j] = diagram_distance(oa.diagram, ob.diagram)
    # Hungarian assignment
    from scipy.optimize import linear_sum_assignment
    if D.size == 0:
        return ([], None) if return_matrix else ([], None)
    rows, cols = linear_sum_assignment(D)
    matches = list(zip(rows.tolist(), cols.tolist()))
    return (matches, D) if return_matrix else (matches, None)

# ------------------ High-level helpers ------------------
def enrich_objects_with_tda(objs: List[ARCObject], homology_dimensions=(0,1)) -> List[ARCObject]:
    if not objs:
        return objs
    for o in objs:
        o.diagram = compute_diagram_for_mask(o.mask, homology_dimensions=homology_dimensions)
    feats = batch_vectorize_diagrams([o.diagram for o in objs])
    for o, f in zip(objs, feats):
        o.features = f
    return objs

def grid_to_objects_with_tda(grid: np.ndarray, background: int = 0, connectivity: int = 1, homology_dimensions=(0,1)) -> List[ARCObject]:
    objs = extract_objects_from_grid(grid, background=background, connectivity=connectivity)
    return enrich_objects_with_tda(objs, homology_dimensions=homology_dimensions)
