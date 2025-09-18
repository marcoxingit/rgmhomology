
# arc_tda_adapter.py (manual features; reuses custom matching)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

from arc_tda_features import (
    ARCObject,
    extract_objects_from_grid,
    compute_diagram_for_mask,
    wasserstein_matching,
    batch_vectorize_diagrams,
)

@dataclass
class TDATrack:
    idx: int
    objs: List[Optional[ARCObject]]
    token_ids: List[int]
    centroids: List[Optional[Tuple[float, float]]]

def _cluster_tokens(all_features: np.ndarray, n_token_bins: int, random_state: int = 0):
    n_samples = len(all_features)
    n_clusters = max(1, min(int(n_token_bins), int(n_samples)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(all_features)
    return kmeans

def _vectorize_sequence(grids: List[np.ndarray], background: int = 0):
    seq_objs: List[List[ARCObject]] = []
    for g in grids:
        objs = extract_objects_from_grid(g, background=background)
        for o in objs:
            o.diagram = compute_diagram_for_mask(o.mask)  # fills diagram
            o.features = None
        seq_objs.append(objs)
    return seq_objs

def _build_tracks(seq_objs: List[List[ARCObject]]) -> List[TDATrack]:
    T = len(seq_objs)
    if T == 0:
        return []
    tracks: List[TDATrack] = [TDATrack(i, [o], [], [o.centroid_rc]) for i, o in enumerate(seq_objs[0])]
    for t in range(1, T):
        prev = [tr.objs[-1] for tr in tracks]
        curr = seq_objs[t]
        prev_valid = [(i, o) for i, o in enumerate(prev) if o is not None]
        if len(prev_valid) == 0:
            for o in curr:
                tracks.append(TDATrack(len(tracks), [None]*t + [o], [], [None]*t + [o.centroid_rc]))
            continue
        if len(curr) == 0:
            for tr in tracks:
                tr.objs.append(None)
                tr.centroids.append(None)
            continue
        idx_prev, objs_prev = zip(*prev_valid)
        matches, _ = wasserstein_matching(list(objs_prev), curr, return_matrix=False)
        matched_prev, matched_curr = set(), set()
        for i_prev_local, j_curr in matches:
            i_prev = idx_prev[i_prev_local]
            matched_prev.add(i_prev); matched_curr.add(j_curr)
            tr = tracks[i_prev]
            tr.objs.append(curr[j_curr])
            tr.centroids.append(curr[j_curr].centroid_rc)
        for i_prev in range(len(prev)):
            if i_prev not in matched_prev:
                tracks[i_prev].objs.append(None)
                tracks[i_prev].centroids.append(None)
        for j in range(len(curr)):
            if j not in matched_curr:
                tracks.append(TDATrack(len(tracks), [None]*t + [curr[j]], [], [None]*t + [curr[j].centroid_rc]))
    return tracks

def tda_sequence_to_onehots(grids: List[np.ndarray], n_bins: int = 9, background: int = 0, random_state: int = 0):
    assert n_bins >= 2, "n_bins must be at least 2 (>=1 token + 1 none)"
    seq_objs = _vectorize_sequence(grids, background=background)
    # batch vectorize (manual fixed-length features)
    all_diags, index_map = [], []
    for t, objs in enumerate(seq_objs):
        for k, o in enumerate(objs):
            all_diags.append(o.diagram)
            index_map.append((t, k))
    if len(all_diags) == 0:
        raise ValueError("No objects found in any frame.")
    features = batch_vectorize_diagrams(all_diags)  # (N, F)
    for idx, (t, k) in enumerate(index_map):
        seq_objs[t][k].features = features[idx]
    tracks = _build_tracks(seq_objs)
    feats = []
    for tr in tracks:
        for o in tr.objs:
            if o is not None:
                feats.append(o.features)
    if len(feats) == 0:
        raise ValueError("No objects found in any frame.")
    feats = np.stack(feats, axis=0)
    req = n_bins - 1
    kmeans = _cluster_tokens(feats, req, random_state=random_state)
    eff = int(kmeans.n_clusters)
    total_bins = eff + 1
    T = len(grids)
    for tr in tracks:
        tr.token_ids = []
        for o in tr.objs:
            if o is None:
                tr.token_ids.append(eff)
            else:
                tr.token_ids.append(int(kmeans.predict(o.features[None, :])[0]))
    one_hots = np.zeros((len(tracks), T, total_bins), dtype=np.float32)
    for m, tr in enumerate(tracks):
        for t, tok in enumerate(tr.token_ids):
            one_hots[m, t, tok] = 1.0
    meta = {"tracks": tracks, "n_token_bins": eff, "actual_total_bins": total_bins}
    return one_hots, meta
