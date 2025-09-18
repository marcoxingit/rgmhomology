# arc_tda_features.py
# Utilities to extract topological features (via giotto-tda) from ARC-style grids
# and to match objects across grids using Wasserstein distances on persistence diagrams.
#
# Author: ChatGPT (for Marco)
# License: MIT
#
# Requirements (install locally):
#   pip install giotto-tda scikit-learn scipy numpy
#
# Notes:
# - We use CubicalPersistence on (signed) distance transforms of binary masks to obtain
#   stable diagrams that characterize object shape (connected components and holes).
# - We then vectorize with PersistenceEntropy, Amplitude, and BettiCurve for fast use,
#   and we keep the raw diagrams for optimal-transport matching (Wasserstein).
#
# ARC conventions assumed:
# - Grid is a 2D numpy array of small integers (colors), with 0 treated as background by default.
# - 'Objectness' = connected components in each color layer (4-connectivity by default).

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy import ndimage as ndi

try:
    # giotto-tda imports
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceEntropy,
        Amplitude,
        BettiCurve,
        PairwiseDistance,
    )
except Exception as e:
    raise ImportError(
        "This module requires giotto-tda. Install with:\n"
        "    pip install giotto-tda\n"
        f"Original import error: {e}"
    )


@dataclass
class ARCObject:
    color: int
    mask: np.ndarray  # bool array, same HxW as grid, True where object pixels are
    bbox: Tuple[
        int, int, int, int
    ]  # (min_row, min_col, max_row_exclusive, max_col_exclusive)
    centroid_rc: Tuple[float, float]  # (row, col) centroid in image coordinates
    diagram: Optional[np.ndarray] = (
        None  # raw persistence diagram (n_points, 3) for dims 0/1
    )
    features: Optional[np.ndarray] = None  # vectorized features (1D)

    def crop(self) -> np.ndarray:
        r0, c0, r1, c1 = self.bbox
        return self.mask[r0:r1, c0:c1]


def _connected_components(
    mask: np.ndarray, connectivity: int = 1
) -> Tuple[np.ndarray, int]:
    structure = ndi.generate_binary_structure(2, connectivity)
    labeled, n = ndi.label(mask, structure=structure)
    return labeled, n


def extract_objects_from_grid(
    grid: np.ndarray,
    background: int = 0,
    connectivity: int = 1,
) -> List[ARCObject]:
    """Segments an ARC grid into per-color connected components (objects).

    Returns a list of ARCObject instances with masks and bounding boxes populated.
    """
    H, W = grid.shape
    objects: List[ARCObject] = []
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != background)
    for color in colors:
        layer = grid == color
        labeled, n = _connected_components(layer, connectivity=connectivity)
        for lab in range(1, n + 1):
            obj_mask = labeled == lab
            if not obj_mask.any():
                continue
            rows, cols = np.where(obj_mask)
            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1
            cy, cx = rows.mean(), cols.mean()
            objects.append(
                ARCObject(
                    color=color,
                    mask=obj_mask,
                    bbox=(r0, c0, r1, c1),
                    centroid_rc=(cy, cx),
                )
            )
    return objects


def _signed_distance_filtration(mask: np.ndarray) -> np.ndarray:
    """Returns a signed distance transform suitable for a robust cubical filtration.

    Positive values inside the object (distance to background), negative outside
    (negative distance to object). Zeros on the boundary.
    """
    # Distance inside the object
    dist_in = ndi.distance_transform_edt(mask.astype(np.uint8))
    # Distance outside the object
    dist_out = ndi.distance_transform_edt((~mask).astype(np.uint8))
    signed = dist_in - dist_out
    return signed.astype(np.float32)


def compute_diagram_for_mask(
    mask: np.ndarray,
    homology_dimensions=(0, 1),
    reduced_homology=True,
) -> np.ndarray:
    """Compute a persistence diagram for a single binary mask via CubicalPersistence.

    Uses the signed distance transform as the filtering function.
    Returns an array of shape (n_points, 3) with columns [birth, death, homology_dim].
    """
    cp = CubicalPersistence(
        homology_dimensions=homology_dimensions,
        reduced_homology=reduced_homology,
    )
    filt = _signed_distance_filtration(mask)
    # CubicalPersistence expects shape (n_samples, H, W)
    X = filt[None, ...]
    diagrams = cp.fit_transform(X)
    return diagrams[0]


def vectorize_diagram(
    diagram: np.ndarray,
    n_betti_bins: int = 16,
    amplitude_metric: str = "wasserstein",
    amplitude_order: int = 1,
    normalize_entropy: bool = True,
) -> np.ndarray:
    """Vectorize a persistence diagram into a compact 1D feature vector.

    Features = [PersistenceEntropy per homology dim | Amplitude per dim | BettiCurve samples per dim].
    """
    # Wrap single diagram as a batch of one sample
    Di = diagram[None, :, :]

    pe = PersistenceEntropy(normalize=normalize_entropy)
    # Returns shape (n_samples, n_dims) -> (1, d)
    pe_f = pe.fit_transform(Di)[0]

    amp = Amplitude(metric=amplitude_metric, metric_params={"p": amplitude_order})
    amp_f = amp.fit_transform(Di)[0]  # shape (1, d)

    bc = BettiCurve(n_bins=n_betti_bins)
    bc_f = bc.fit_transform(Di)[0]  # shape (n_dims * n_bins,)

    return np.concatenate([pe_f.ravel(), amp_f.ravel(), bc_f.ravel()])


def enrich_objects_with_tda(
    objects: List[ARCObject],
    homology_dimensions=(0, 1),
    n_betti_bins: int = 16,
) -> List[ARCObject]:
    """Compute (diagram, features) for each ARCObject and return the list."""
    for obj in objects:
        diag = compute_diagram_for_mask(
            obj.mask, homology_dimensions=homology_dimensions
        )
        feats = vectorize_diagram(diag, n_betti_bins=n_betti_bins)
        obj.diagram = diag
        obj.features = feats
    return objects


def wasserstein_matching(
    objs_A: List[ARCObject],
    objs_B: List[ARCObject],
    order: int = 1,
    return_matrix: bool = False,
) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray]]:
    """Match objects across two grids by minimizing Wasserstein distance between diagrams.

    Returns: list of (i, j) indices where i indexes objs_A and j indexes objs_B.
    Uses a simple Hungarian solver on the pairwise distance matrix.
    """
    # def wasserstein_matching(objs_A, objs_B, order: int = 1, return_matrix: bool = False):
    # NEW: guard empties
    if len(objs_A) == 0 or len(objs_B) == 0:
        return ([], None) if return_matrix else ([], None)

    if any(o.diagram is None for o in objs_A + objs_B):
        raise ValueError("Call enrich_objects_with_tda on both lists first.")

    # if any(o.diagram is None for o in objs_A + objs_B):
    #     raise ValueError("Call enrich_objects_with_tda on both lists first.")

    # Stack diagrams into batches
    Di_A = [o.diagram for o in objs_A]
    Di_B = [o.diagram for o in objs_B]

    # Pairwise distances between all diagrams from A and B
    # PairwiseDistance expects a *single* list; we compute a full matrix by concatenating
    # and then slicing, or more simply compute between the two sets by looping.
    pwd = PairwiseDistance(metric="wasserstein", metric_params={"p": order}, n_jobs=-1)

    # Transform expects an array of diagrams. We'll compute distances A vs B with broadcasting:
    # Concatenate and then compute full pairwise matrix is simpler for clarity.
    Di = np.array(Di_A + Di_B, dtype=object)
    # Create index slices
    nA = len(Di_A)
    nB = len(Di_B)
    # Full pairwise distance among all (A+B). We then slice the top-right block.
    D_all = pwd.fit_transform(Di)  # shape (nA+nB, nA+nB, n_dims) or (nA+nB, nA+nB)
    # If distances per-dimension are returned as a 3D array, average across dims.
    if D_all.ndim == 3:
        D_all = D_all.mean(axis=2)
    D = D_all[:nA, nA : nA + nB]

    # Hungarian assignment
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(D)
    matches = list(zip(rows.tolist(), cols.tolist()))
    return (matches, D) if return_matrix else (matches, None)


def grid_to_objects_with_tda(
    grid: np.ndarray,
    background: int = 0,
    connectivity: int = 1,
    homology_dimensions=(0, 1),
    n_betti_bins: int = 16,
) -> List[ARCObject]:
    """Convenience function: segment grid -> objects -> compute TDA features."""
    objs = extract_objects_from_grid(
        grid, background=background, connectivity=connectivity
    )
    objs = enrich_objects_with_tda(
        objs, homology_dimensions=homology_dimensions, n_betti_bins=n_betti_bins
    )
    return objs


def demo__toy_example():
    """Tiny self-test on random blobs; run this file directly to try."""
    rng = np.random.default_rng(0)
    H = W = 20
    grid_A = np.zeros((H, W), dtype=int)
    # Create two colored squares
    grid_A[2:6, 2:6] = 3
    grid_A[10:15, 4:9] = 5

    # Translate in grid_B
    grid_B = np.zeros_like(grid_A)
    grid_B[4:8, 7:11] = 3  # moved
    grid_B[12:17, 10:15] = 5  # moved

    objs_A = grid_to_objects_with_tda(grid_A)
    objs_B = grid_to_objects_with_tda(grid_B)
    matches, D = wasserstein_matching(objs_A, objs_B, return_matrix=True)
    print("Matches (A->B):", matches)
    print("Distance matrix:\n", np.round(D, 3))


if __name__ == "__main__":
    demo__toy_example()
