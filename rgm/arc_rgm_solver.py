# arc_rgm_solver.py
# RGM-based ARC solver using giotto-TDA tokens as observations.
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from sklearn.cluster import KMeans

# Your existing pieces
from fast_structure_learning import spm_mb_structure_learning
from arc_tda_features import (
    grid_to_objects_with_tda,
    batch_vectorize_diagrams,
    wasserstein_matching,  # used for train-time color/translation stats
)

# if you have a set_matching_metric, it's harmless not to call it here


# ---------- small utils ----------
def grid_from_list(lst: List[List[int]]) -> np.ndarray:
    return np.asarray(lst, dtype=np.int32)


def grid_to_list(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


def infer_background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def move_mask_into(
    canvas: np.ndarray, mask: np.ndarray, dy: int, dx: int, color: int
) -> None:
    """Paste 'color' where mask==True translated by (dy,dx) with clipping into 'canvas' in-place."""
    H, W = canvas.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return
    y2 = ys + int(dy)
    x2 = xs + int(dx)
    keep = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
    y2 = y2[keep]
    x2 = x2[keep]
    canvas[y2, x2] = color


# ---------- TDA → tokens ----------
def _extract_objs_and_tokens(
    grid: np.ndarray,
    background: int,
    kmeans: Optional[KMeans] = None,
) -> Tuple[List[Any], np.ndarray]:
    """
    Returns:
      objs: list of objects with .mask, .color, .centroid_rc, .diagram
      tokens: np.array shape (len(objs),) of ints
    If kmeans is None, also returns tokens via a temporary fit on this grid's features.
    """
    objs = grid_to_objects_with_tda(grid, background=background)
    if len(objs) == 0:
        return [], np.array([], dtype=int)
    feats = batch_vectorize_diagrams([o.diagram for o in objs])
    if kmeans is None:
        # trivial fallback
        kmeans = KMeans(n_clusters=min(3, len(objs)), n_init=10, random_state=0)
        kmeans.fit(feats)
    toks = kmeans.predict(feats)
    return objs, toks


# ---------- Learn token mapping from an RGM ----------
def _derive_token_transition_from_agent(agent) -> Dict[int, int]:
    """
    Build an observation-token transition f(o_in) -> o_out from learned A and B.
    We do it per-modality and then majority-vote across modalities.
    """
    # agent.A: list over modalities; each A[m] has shape (batch=1, n_obs, n_states)
    # agent.B: list over groups; we take group 0: (batch=1, n_states, n_states, n_controls=1)
    votes: Dict[int, Dict[int, int]] = {}
    for A_m in agent.A:
        A = A_m[0]  # (n_obs, n_states)
        n_obs, n_states = A.shape
        # obs->state (MAP)
        obs2state = np.argmax(A, axis=1)  # (n_obs,)
        # state->obs (MAP)
        state2obs = np.argmax(A, axis=0)  # (n_states,)

        # One (the first) group's B
        B = agent.B[0][0, :, :, 0]  # (n_states, n_states)
        for o in range(n_obs):
            s = int(obs2state[o])
            s_next = int(np.argmax(B[s]))
            o_next = int(state2obs[s_next])
            votes.setdefault(o, {})
            votes[o][o_next] = votes[o].get(o_next, 0) + 1

    # majority vote across modalities
    o2o: Dict[int, int] = {}
    for o, d in votes.items():
        o2o[o] = max(d.items(), key=lambda kv: kv[1])[0]
    return o2o


# ---------- Train-time motion and color stats (from TDA matchings) ----------
def _learn_motion_and_colors(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    background: int,
) -> Tuple[
    Dict[int, Tuple[int, int]], Tuple[int, int], Dict[int, int], Tuple[int, int]
]:
    """
    Returns:
      deltas_per_color: in_color -> (dy, dx) (median over matches)
      global_delta: (dy, dx) median over all matches
      color_map: in_color -> out_color (majority over matches)
      out_shape: (H_out, W_out) if constant across train outputs else from last example
    """
    deltas_by_color: Dict[int, List[Tuple[int, int]]] = {}
    all_deltas: List[Tuple[int, int]] = []
    color_votes: Dict[int, Dict[int, int]] = {}
    out_shape = None

    for xi, yi in train_pairs:
        out_shape = yi.shape
        objs_x = grid_to_objects_with_tda(xi, background=background)
        objs_y = grid_to_objects_with_tda(yi, background=background)
        if len(objs_x) == 0 or len(objs_y) == 0:
            continue
        matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
        for i, j in matches:
            ox, oy = objs_x[i], objs_y[j]
            cin, cout = int(ox.color), int(oy.color)
            cy_in, cx_in = ox.centroid_rc
            cy_out, cx_out = oy.centroid_rc
            dy = int(np.round(cy_out - cy_in))
            dx = int(np.round(cx_out - cx_in))
            deltas_by_color.setdefault(cin, []).append((dy, dx))
            all_deltas.append((dy, dx))
            color_votes.setdefault(cin, {})
            color_votes[cin][cout] = color_votes[cin].get(cout, 0) + 1

    # medians for robustness
    deltas_per_color: Dict[int, Tuple[int, int]] = {}
    for cin, lst in deltas_by_color.items():
        dys = np.array([d[0] for d in lst], dtype=int)
        dxs = np.array([d[1] for d in lst], dtype=int)
        deltas_per_color[cin] = (int(np.median(dys)), int(np.median(dxs)))

    if all_deltas:
        global_delta = (
            int(np.median([d[0] for d in all_deltas])),
            int(np.median([d[1] for d in all_deltas])),
        )
    else:
        global_delta = (0, 0)

    color_map: Dict[int, int] = {}
    for cin, d in color_votes.items():
        color_map[cin] = max(d.items(), key=lambda kv: kv[1])[0]

    if out_shape is None:
        out_shape = train_pairs[-1][1].shape

    return deltas_per_color, global_delta, color_map, out_shape


# ---------- The solver ----------
class ARCRGMSolver:
    def __init__(
        self, n_bins: int = 7, dx: int = 2, background: int = 0, random_state: int = 0
    ):
        self.n_bins = int(n_bins)
        self.dx = int(dx)
        self.background = int(background)
        self.random_state = int(random_state)
        # learned artifacts
        self.kmeans: Optional[KMeans] = None
        self.agent = None
        self.o2o: Dict[int, int] = {}
        self.deltas_per_color: Dict[int, Tuple[int, int]] = {}
        self.global_delta: Tuple[int, int] = (0, 0)
        self.color_map: Dict[int, int] = {}
        self.out_shape: Tuple[int, int] = None

    def fit_task(self, task: Dict[str, Any]) -> None:
        # 1) Gather all train grids (input and output) to fit KMeans on TDA features
        train_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        all_objs_feats: List[np.ndarray] = []
        for p in task["train"]:
            xi = grid_from_list(p["input"])
            yi = grid_from_list(p["output"])
            train_pairs.append((xi, yi))
            for g in (xi, yi):
                objs = grid_to_objects_with_tda(g, background=self.background)
                if len(objs) == 0:
                    continue
                feats = batch_vectorize_diagrams([o.diagram for o in objs])
                all_objs_feats.append(feats)
        if len(all_objs_feats) == 0:
            # degenerate: no objects anywhere; fall back to identity KMeans
            self.kmeans = KMeans(
                n_clusters=self.n_bins, n_init=10, random_state=self.random_state
            )
            # Fit on a dummy single vector
            self.kmeans.fit(np.zeros((self.n_bins, 8), dtype=float))
        else:
            X = np.vstack(all_objs_feats)
            k = min(self.n_bins, max(2, min(32, len(np.unique(X, axis=0)))))
            self.kmeans = KMeans(
                n_clusters=k, n_init=10, random_state=self.random_state
            )
            self.kmeans.fit(X)

        # 2) Build one-hot sequences per train pair: time steps t=0 (input), t=1 (output)
        # We define modalities as "tracks" tied by Wasserstein matching between t=0 and t=1 per pair.
        # For the RGM learner we just need (modalities x time x bins) and a locations matrix for modalities.
        # We'll build modalities as "objects at t=0" and match to t=1 tokens for the same track.
        seq_tokens: List[np.ndarray] = []  # each is shape (T=2, k)
        locs_xy: List[Tuple[float, float]] = []  # one per modality (from t=0 centroid)
        for xi, yi in train_pairs:
            objs_x, toks_x = _extract_objs_and_tokens(xi, self.background, self.kmeans)
            objs_y, toks_y = _extract_objs_and_tokens(yi, self.background, self.kmeans)
            if len(objs_x) == 0:
                continue
            # match objects x->y within this pair (TDA Wasserstein)
            matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
            # we’ll define a “track” per x-object; if unmatched, keep its y-token as self (or ignore)
            k = self.kmeans.n_clusters
            onehots = np.zeros((2, k), dtype=np.int32)
            # t=0 tokens
            for i, ox in enumerate(objs_x):
                onehots[0, int(toks_x[i])] += 1
            # t=1 tokens for matched y
            for i, j in matches:
                onehots[1, int(toks_y[j])] += 1
            # unmatched y’s do not increase modality count (keeps same set per pair)
            seq_tokens.append(onehots)
            for ox in objs_x:
                cy, cx = ox.centroid_rc
                locs_xy.append((float(cx), float(cy)))

        if len(seq_tokens) == 0:
            # edge case: nothing found; fabricate a tiny sequence to avoid crashes
            k = max(2, self.kmeans.n_clusters if self.kmeans is not None else 2)
            one_hots = np.zeros((1, 2, k), dtype=np.int32)
            locations_matrix = np.array([[0.5, 0.5]], dtype=float)
        else:
            # stack modalities: each track becomes its own "modality"
            # (modalities x T x k)
            one_hots = np.stack(seq_tokens, axis=0)  # (M, 2, k)
            locations_matrix = np.asarray(locs_xy, dtype=float)

        # 3) Learn RGM
        H_guess, W_guess = train_pairs[0][0].shape
        agents, RG, _ = spm_mb_structure_learning(
            one_hots,
            locations_matrix=locations_matrix,
            size=(W_guess, H_guess, one_hots.shape[0]),  # (width, height, modalities)
            dx=self.dx,
            num_controls=0,
            max_levels=4,
            agents=None,
            RG=None,
        )
        self.agent = agents[0]  # top-level agent

        # 4) Build token transition f(o)->o' from A & B
        self.o2o = _derive_token_transition_from_agent(self.agent)

        # 5) Learn motion and global color map from train matches
        self.deltas_per_color, self.global_delta, self.color_map, self.out_shape = (
            _learn_motion_and_colors(train_pairs, self.background)
        )

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        """Predict one test grid using learned RGM token map + motion/color stats."""
        assert self.kmeans is not None and self.agent is not None
        H_out, W_out = self.out_shape if self.out_shape is not None else grid.shape
        out_bg = infer_background_color(
            grid
        )  # fallback; could also learn from train outputs
        canvas = np.full((H_out, W_out), out_bg, dtype=grid.dtype)

        objs, toks = _extract_objs_and_tokens(grid, self.background, self.kmeans)
        for i, o in enumerate(objs):
            cin = int(o.color)
            tok_in = int(toks[i])
            tok_out = self.o2o.get(tok_in, tok_in)  # default identity if unseen
            # recolor: use global color map if available, else keep
            cout = self.color_map.get(cin, cin)
            # motion: per-color median if available, else global
            dy, dx = self.deltas_per_color.get(cin, self.global_delta)
            move_mask_into(canvas, o.mask, dy, dx, cout)

        return canvas

    def solve_task(self, task: Dict[str, Any]) -> List[np.ndarray]:
        self.fit_task(task)
        preds: List[np.ndarray] = []
        for t in task["test"]:
            x = grid_from_list(t["input"])
            y = self.predict_grid(x)
            preds.append(y)
        return preds
