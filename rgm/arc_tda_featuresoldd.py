
# arc_tda_features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage as ndi

from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve, PairwiseDistance

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
    X = _signed_distance_filtration(mask)[None, ...]
    D = cp.fit_transform(X)
    return D[0]

def batch_vectorize_diagrams(diagrams: List[np.ndarray], n_betti_bins: int = 16) -> np.ndarray:
    if len(diagrams) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    Di = np.array(diagrams, dtype=object)
    pe  = PersistenceEntropy(normalize=True)
    amp = Amplitude(metric="wasserstein", metric_params={"p": 1})
    bc  = BettiCurve(n_bins=n_betti_bins)
    pe.fit(Di); amp.fit(Di); bc.fit(Di)
    PE  = pe.transform(Di)
    AMP = amp.transform(Di)
    BC  = bc.transform(Di)
    feats = [np.concatenate([PE[i].ravel(), AMP[i].ravel(), BC[i].ravel()]) for i in range(len(Di))]
    return np.stack(feats, axis=0)

def enrich_objects_with_tda(objs: List[ARCObject], homology_dimensions=(0,1), n_betti_bins: int = 16) -> List[ARCObject]:
    if not objs:
        return objs
    for o in objs:
        o.diagram = compute_diagram_for_mask(o.mask, homology_dimensions=homology_dimensions)
    feats = batch_vectorize_diagrams([o.diagram for o in objs], n_betti_bins=n_betti_bins)
    for o, f in zip(objs, feats):
        o.features = f
    return objs

def grid_to_objects_with_tda(grid: np.ndarray, background: int = 0, connectivity: int = 1, homology_dimensions=(0,1), n_betti_bins: int = 16) -> List[ARCObject]:
    objs = extract_objects_from_grid(grid, background=background, connectivity=connectivity)
    return enrich_objects_with_tda(objs, homology_dimensions=homology_dimensions, n_betti_bins=n_betti_bins)

def wasserstein_matching(objs_A: List[ARCObject], objs_B: List[ARCObject], order: int = 1, return_matrix: bool = False):
    if len(objs_A) == 0 or len(objs_B) == 0:
        return ([], None) if return_matrix else ([], None)
    if any(o.diagram is None for o in objs_A + objs_B):
        raise ValueError("Call enrich_objects_with_tda on both lists first.")
    Di_A = [o.diagram for o in objs_A]
    Di_B = [o.diagram for o in objs_B]
    pwd = PairwiseDistance(metric="wasserstein", metric_params={"p": order}, n_jobs=-1)
    Di = np.array(Di_A + Di_B, dtype=object)
    D_all = pwd.fit_transform(Di)
    if D_all.ndim == 3:
        D_all = D_all.mean(axis=2)
    nA = len(Di_A)
    D = D_all[:nA, nA:]
    from scipy.optimize import linear_sum_assignment
    if D.size == 0:
        return ([], None) if return_matrix else ([], None)
    rows, cols = linear_sum_assignment(D)
    matches = list(zip(rows.tolist(), cols.tolist()))
    return (matches, D) if return_matrix else (matches, None)
