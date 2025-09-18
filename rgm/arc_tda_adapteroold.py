# arc_tda_adapter.py
# Turn a sequence of ARC-style integer grids into one-hots for rgm2.RGM using giotto-tda.
#
# Pipeline:
#   - segment objects per color (nonzero) with 4-connectivity
#   - for each object -> persistence diagrams via CubicalPersistence on signed distance
#   - vectorize (entropy, amplitude, betti curve)
#   - across the whole sequence, KMeans on vectors -> discrete tokens (n_bins - 1 used for tokens)
#   - track objects over time with Wasserstein matching to keep stable modality indices
#   - allocate one extra bin for 'none' (no object for a given track/time)
#
# Output:
#   - one_hots: (n_modalities, T, n_bins) float32 one-hot array
#   - meta: dict with tracks, centroids, codebook, token ids per (t, obj)
#
# Requirements:
#   pip install giotto-tda scikit-learn numpy scipy
#
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.cluster import KMeans

from .arc_tda_features import (
    ARCObject,
    grid_to_objects_with_tda,
    wasserstein_matching,
)


@dataclass
class TDATrack:
    # Holds a single object "identity" across time
    idx: int
    objs: List[Optional[ARCObject]]  # length T, None if missing at that time
    token_ids: List[int]  # length T, token index in [0..K] where K is 'none'
    centroids: List[Optional[Tuple[float, float]]]


def _cluster_tokens(all_features: np.ndarray, n_token_bins: int, random_state: int = 0):
    kmeans = KMeans(n_clusters=n_token_bins, n_init="auto", random_state=random_state)
    kmeans.fit(all_features)
    return kmeans


def _vectorize_sequence(grids: List[np.ndarray], background: int = 0):
    # Per time-step list of ARCObject (with .features filled)
    seq_objs: List[List[ARCObject]] = []
    for g in grids:
        objs = grid_to_objects_with_tda(g, background=background)
        seq_objs.append(objs)
    return seq_objs


def _build_tracks(seq_objs: List[List[ARCObject]]) -> List[TDATrack]:
    """Track objects across time with Wasserstein matching (diagram-level) to keep index stability.

    Returns a list of TDATrack with variable length depending on appearance/disappearance.
    """
    T = len(seq_objs)
    # Initialize tracks from t=0
    tracks: List[TDATrack] = [
        TDATrack(i, [o], [], [o.centroid_rc]) for i, o in enumerate(seq_objs[0])
    ]

    for t in range(1, T):
        prev = [tr.objs[-1] for tr in tracks]  # last known at t-1
        curr = seq_objs[t]

        # Build matching sets where both sides have an object
        prev_valid = [(i, o) for i, o in enumerate(prev) if o is not None]
        if len(prev_valid) == 0:
            # If no prior objects, start new tracks for all current
            for o in curr:
                tracks.append(
                    TDATrack(
                        len(tracks),
                        [None] * (t) + [o],
                        [],
                        [None] * (t) + [o.centroid_rc],
                    )
                )
            continue

        idx_prev, objs_prev = zip(*prev_valid) if prev_valid else ([], [])

        if len(curr) == 0:
            # No objects now: append None for every track
            for tr in tracks:
                tr.objs.append(None)
                tr.centroids.append(None)
            continue

        matches, _ = wasserstein_matching(list(objs_prev), curr, return_matrix=False)
        matched_prev = set()
        matched_curr = set()
        # Assign matches
        for i_prev_local, j_curr in matches:
            i_prev = idx_prev[i_prev_local]
            matched_prev.add(i_prev)
            matched_curr.add(j_curr)
            tr = tracks[i_prev]
            tr.objs.append(curr[j_curr])
            tr.centroids.append(curr[j_curr].centroid_rc)

        # Unmatched previous tracks -> append None (object disappeared)
        for i_prev in range(len(prev)):
            if i_prev not in matched_prev:
                tracks[i_prev].objs.append(None)
                tracks[i_prev].centroids.append(None)

        # Unmatched current objs -> start new tracks
        for j in range(len(curr)):
            if j not in matched_curr:
                new = TDATrack(
                    idx=len(tracks),
                    objs=[None] * t + [curr[j]],
                    token_ids=[],
                    centroids=[None] * t + [curr[j].centroid_rc],
                )
                tracks.append(new)

    return tracks


def tda_sequence_to_onehots(
    grids: List[np.ndarray],
    n_bins: int = 9,
    background: int = 0,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Turn a sequence of ARC grids into one-hots (modalities x time x bins).

    The last bin (index n_bins-1) is reserved for 'none' (missing object).
    """
    assert n_bins >= 2, "n_bins must be at least 2 (>=1 token + 1 none)"
    # 1) Extract objs + TDA features across time
    seq_objs = _vectorize_sequence(grids, background=background)
    # 2) Track objects over time (diagram-level matching)
    tracks = _build_tracks(seq_objs)

    # 3) Collect all features to learn a token codebook (n_bins - 1 clusters)
    feats = []
    for tr in tracks:
        for o in tr.objs:
            if o is not None:
                feats.append(o.features)
    if len(feats) == 0:
        raise ValueError("No objects found in any frame.")
    feats = np.stack(feats, axis=0)

    n_token_bins = n_bins - 1
    kmeans = _cluster_tokens(feats, n_token_bins, random_state=random_state)

    # 4) Assign token IDs per (track, time)
    T = len(grids)
    for tr in tracks:
        tr.token_ids = []
        for o in tr.objs:
            if o is None:
                tr.token_ids.append(n_token_bins)  # 'none' bin
            else:
                token = int(kmeans.predict(o.features[None, :])[0])
                tr.token_ids.append(token)

    # 5) Build one-hots array: (n_modalities, T, n_bins)
    n_modalities = len(tracks)
    one_hots = np.zeros((n_modalities, T, n_bins), dtype=np.float32)
    for m, tr in enumerate(tracks):
        for t, tok in enumerate(tr.token_ids):
            one_hots[m, t, tok] = 1.0

    meta = {
        "tracks": tracks,
        "kmeans_centers": kmeans.cluster_centers_,
        "n_token_bins": n_token_bins,
    }
    return one_hots, meta


if __name__ == "__main__":
    # quick self-check
    H = W = 15
    g0 = np.zeros((H, W), int)
    g0[2:5, 2:5] = 3
    g0[8:12, 4:7] = 5
    g1 = np.zeros((H, W), int)
    g1[3:6, 7:10] = 3
    g1[9:13, 10:13] = 5
    grids = [g0, g1]
    one_hots, meta = tda_sequence_to_onehots(grids, n_bins=6)
    print("one_hots shape:", one_hots.shape)
    print(
        "non-empty tracks:",
        sum(any(o is not None for o in tr.objs) for tr in meta["tracks"]),
    )
