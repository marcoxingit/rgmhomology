# # # rgm/arc_rgm_solver_controls.py
# # # RGM-based ARC solver using giotto-TDA tokens as observations and
# # # *discrete actions as controls* (if fast_structure_learning supports it).

# # from __future__ import annotations
# # from typing import List, Tuple, Dict, Any, Optional
# # import numpy as np
# # from sklearn.cluster import KMeans

# # from rgm.fast_structure_learning import spm_mb_structure_learning
# # from rgm.arc_tda_features import (
# #     grid_to_objects_with_tda,
# #     batch_vectorize_diagrams,
# #     wasserstein_matching,
# # )
# # from rgm.arc_actions import ActionLibrary


# # # ---------- small utils ----------
# # def grid_from_list(lst: List[List[int]]) -> np.ndarray:
# #     return np.asarray(lst, dtype=np.int32)


# # def grid_to_list(arr: np.ndarray) -> List[List[int]]:
# #     return arr.astype(int).tolist()


# # def infer_background_color(grid: np.ndarray) -> int:
# #     vals, counts = np.unique(grid, return_counts=True)
# #     return int(vals[counts.argmax()])


# # def _clip_and_jitter_locations(
# #     locs_xy: np.ndarray, W: int, H: int, eps: float = 1e-3
# # ) -> np.ndarray:
# #     if locs_xy.size == 0:
# #         return locs_xy
# #     locs = locs_xy.astype(float).copy()
# #     locs[:, 0] = np.clip(locs[:, 0], 0.5, max(0.5, W - 1.5))
# #     locs[:, 1] = np.clip(locs[:, 1], 0.5, max(0.5, H - 1.5))
# #     rng = np.random.RandomState(0)
# #     jitter = (rng.rand(*locs.shape) - 0.5) * eps
# #     return locs + jitter


# # def _synthetic_uniform_layout(M: int, W: int, H: int) -> np.ndarray:
# #     if M <= 0:
# #         return np.zeros((0, 2), dtype=float)
# #     r = int(np.ceil(np.sqrt(M)))
# #     c = int(np.ceil(M / r))
# #     xs = np.linspace(0.5, max(0.5, W - 1.5), c)
# #     ys = np.linspace(0.5, max(0.5, H - 1.5), r)
# #     coords = []
# #     for i in range(M):
# #         rr = i // c
# #         cc = i % c
# #         coords.append([xs[min(cc, len(xs) - 1)], ys[min(rr, len(ys) - 1)]])
# #     return np.asarray(coords, dtype=float)


# # # ---------- TDA → tokens ----------
# # def _extract_objs_and_tokens(
# #     grid: np.ndarray, background: int, kmeans: Optional[KMeans]
# # ) -> Tuple[List[Any], np.ndarray, np.ndarray]:
# #     objs = grid_to_objects_with_tda(grid, background=background)
# #     if len(objs) == 0:
# #         return [], np.array([], dtype=int), np.zeros((0, 0), dtype=float)
# #     feats = batch_vectorize_diagrams([o.diagram for o in objs])
# #     toks = kmeans.predict(feats)
# #     return objs, toks, feats


# # # ---------- A/B decoding ----------
# # def _derive_token_transition_from_agent(agent) -> Dict[int, int]:
# #     votes: Dict[int, Dict[int, int]] = {}
# #     for A_m in agent.A:
# #         A = A_m[0]  # (n_obs, n_states)
# #         obs2state = np.argmax(A, axis=1)  # (n_obs,)
# #         state2obs = np.argmax(A, axis=0)  # (n_states,)
# #         # Choose a default control slice 0 for decoding a base mapping
# #         B = agent.B[0][0, :, :, 0]  # (n_states, n_states)
# #         for o in range(A.shape[0]):
# #             s = int(obs2state[o])
# #             s2 = int(np.argmax(B[s]))
# #             o2 = int(state2obs[s2])
# #             votes.setdefault(o, {})
# #             votes[o][o2] = votes[o].get(o2, 0) + 1
# #     return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}


# # def _derive_token_transition_for_action(agent, a: int) -> Dict[int, int]:
# #     votes: Dict[int, Dict[int, int]] = {}
# #     for A_m in agent.A:
# #         A = A_m[0]
# #         obs2state = np.argmax(A, axis=1)
# #         state2obs = np.argmax(A, axis=0)
# #         # Take group 0, control slice a (if available)
# #         Bfull = agent.B[0][0]  # (n_states, n_states, n_controls)
# #         a_idx = min(a, Bfull.shape[-1] - 1)
# #         B = Bfull[:, :, a_idx]
# #         for o in range(A.shape[0]):
# #             s = int(obs2state[o])
# #             s2 = int(np.argmax(B[s]))
# #             o2 = int(state2obs[s2])
# #             votes.setdefault(o, {})
# #             votes[o][o2] = votes[o].get(o2, 0) + 1
# #     return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}


# # # ---------- action & color priors ----------
# # def _learn_actions_and_colors(train_pairs, background: int, actions: ActionLibrary):
# #     """
# #     Learn P(a|color_in) (mode), P(a) (mode), and color map (majority).
# #     """
# #     A = actions.count
# #     action_counts_global = np.zeros(A, dtype=np.int64)
# #     action_counts_by_color: Dict[int, np.ndarray] = {}
# #     color_votes: Dict[int, Dict[int, int]] = {}
# #     out_shape = None

# #     for xi, yi in train_pairs:
# #         out_shape = yi.shape
# #         objs_x = grid_to_objects_with_tda(xi, background=background)
# #         objs_y = grid_to_objects_with_tda(yi, background=background)
# #         if len(objs_x) == 0 or len(objs_y) == 0:
# #             continue
# #         matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
# #         match_map = {i: j for (i, j) in matches}

# #         for i, ox in enumerate(objs_x):
# #             j = match_map.get(i)
# #             cin = int(ox.color)
# #             if cin not in action_counts_by_color:
# #                 action_counts_by_color[cin] = np.zeros(A, dtype=np.int64)

# #             if j is None:
# #                 a = 0  # identity
# #             else:
# #                 oy = objs_y[j]
# #                 a = actions.best_action_by_iou(ox.mask, oy.mask)
# #                 cout = int(oy.color)
# #                 color_votes.setdefault(cin, {})
# #                 color_votes[cin][cout] = color_votes[cin].get(cout, 0) + 1

# #             action_counts_by_color[cin][a] += 1
# #             action_counts_global[a] += 1

# #     action_mode_by_color = {
# #         c: int(np.argmax(cnts)) for c, cnts in action_counts_by_color.items()
# #     }
# #     action_global_mode = (
# #         int(np.argmax(action_counts_global)) if action_counts_global.sum() else 0
# #     )
# #     color_map = {
# #         cin: max(v.items(), key=lambda kv: kv[1])[0] for cin, v in color_votes.items()
# #     }  # default: keep if missing
# #     if out_shape is None:
# #         out_shape = train_pairs[-1][1].shape
# #     return action_mode_by_color, action_global_mode, color_map, out_shape


# # # ---------- Solver ----------
# # class ARCRGMSolverControls:
# #     def __init__(
# #         self,
# #         n_bins: int = 7,
# #         dx: int = 2,
# #         background: int = 0,
# #         random_state: int = 0,
# #         single_group: bool = True,
# #         levels: int = 1,
# #         trans_radius: int = 1,
# #     ):
# #         self.n_bins = int(n_bins)
# #         self.dx = int(dx)
# #         self.background = int(background)
# #         self.random_state = int(random_state)
# #         self.single_group = bool(single_group)
# #         self.levels = int(max(1, levels))
# #         self.actions = ActionLibrary(trans_radius=trans_radius)

# #         # learned artifacts
# #         self.kmeans: Optional[KMeans] = None
# #         self.agent = None
# #         self.o2o_default: Dict[int, int] = {}
# #         self.o2o_by_action: Dict[int, Dict[int, int]] = {}
# #         self.action_mode_by_color: Dict[int, int] = {}
# #         self.action_global_mode: int = 0
# #         self.color_map: Dict[int, int] = {}
# #         self.out_shape: Tuple[int, int] = None
# #         self._controls_used = False  # whether SPM accepted control_sequences

# #     # ---- fit ----
# #     def fit_task(self, task: Dict[str, Any]) -> None:
# #         # 1) KMeans over all object TDA features (train inputs + outputs)
# #         train_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
# #         feats_list: List[np.ndarray] = []
# #         for p in task["train"]:
# #             xi = grid_from_list(p["input"])
# #             yi = grid_from_list(p["output"])
# #             train_pairs.append((xi, yi))
# #             for g in (xi, yi):
# #                 objs = grid_to_objects_with_tda(g, background=self.background)
# #                 if len(objs) == 0:
# #                     continue
# #                 feats = batch_vectorize_diagrams([o.diagram for o in objs])
# #                 feats_list.append(feats)
# #         if len(feats_list) == 0:
# #             self.kmeans = KMeans(
# #                 n_clusters=max(2, self.n_bins),
# #                 n_init=10,
# #                 random_state=self.random_state,
# #             )
# #             self.kmeans.fit(np.zeros((max(2, self.n_bins), 8), dtype=float))
# #         else:
# #             X = np.vstack(feats_list)
# #             k = min(self.n_bins, max(2, min(32, len(np.unique(X, axis=0)))))
# #             self.kmeans = KMeans(
# #                 n_clusters=k, n_init=10, random_state=self.random_state
# #             )
# #             self.kmeans.fit(X)

# #         # 2) Build per-object sequences (T=2) and action controls (T-1=1)
# #         seq_tokens: List[np.ndarray] = []  # (2, K)
# #         seq_controls: List[np.ndarray] = []  # (1, A)
# #         locs_xy: List[Tuple[float, float]] = []

# #         for xi, yi in train_pairs:
# #             objs_x, toks_x, _ = _extract_objs_and_tokens(
# #                 xi, self.background, self.kmeans
# #             )
# #             objs_y, toks_y, _ = _extract_objs_and_tokens(
# #                 yi, self.background, self.kmeans
# #             )

# #             matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
# #             match_map = {i: j for (i, j) in matches}
# #             K = self.kmeans.n_clusters
# #             A = self.actions.count

# #             for i, ox in enumerate(objs_x):
# #                 onehots = np.zeros((2, K), dtype=np.int32)
# #                 controls = np.zeros((1, A), dtype=np.int32)

# #                 tok_x = int(toks_x[i])
# #                 onehots[0, tok_x] = 1
# #                 j = match_map.get(i)
# #                 if j is not None:
# #                     tok_y = int(toks_y[j])
# #                     onehots[1, tok_y] = 1
# #                     a = self.actions.best_action_by_iou(ox.mask, objs_y[j].mask)
# #                     controls[0, a] = 1
# #                 else:
# #                     # unmatched: identity
# #                     onehots[1, tok_x] = 1
# #                     controls[0, 0] = 1

# #                 seq_tokens.append(onehots)
# #                 seq_controls.append(controls)
# #                 cy, cx = ox.centroid_rc
# #                 locs_xy.append((float(cx), float(cy)))

# #         if len(seq_tokens) == 0:
# #             K = max(2, self.kmeans.n_clusters if self.kmeans is not None else 2)
# #             A = self.actions.count
# #             one_hots = np.zeros((1, 2, K), dtype=np.int32)
# #             controls = np.zeros((1, 1, A), dtype=np.int32)
# #             controls[0, 0, 0] = 1
# #             locations_matrix = np.array([[0.5, 0.5]], dtype=float)
# #         else:
# #             one_hots = np.stack(seq_tokens, axis=0)  # (M, 2, K)
# #             controls = np.stack(seq_controls, axis=0)  # (M, 1, A)
# #             H_guess, W_guess = train_pairs[0][0].shape
# #             locations_matrix = _clip_and_jitter_locations(
# #                 np.asarray(locs_xy, dtype=float), W_guess, H_guess
# #             )

# #         # 3) Structure learning with controls if supported
# #         H_guess, W_guess = train_pairs[0][0].shape
# #         M, _, K = one_hots.shape
# #         A = self.actions.count

# #         def _run_spm(
# #             RG_override=None, dx_override=None, levels_override=None, with_controls=True
# #         ):
# #             kwargs = dict(
# #                 observations=one_hots,
# #                 locations_matrix=locations_matrix,
# #                 size=(W_guess, H_guess, M),
# #                 dx=(dx_override if dx_override is not None else self.dx),
# #                 num_controls=(A if with_controls else 0),
# #                 max_levels=(
# #                     levels_override if levels_override is not None else self.levels
# #                 ),
# #                 agents=None,
# #                 RG=RG_override,
# #             )
# #             # Try to pass controls explicitly if API supports it
# #             if with_controls:
# #                 try:
# #                     return spm_mb_structure_learning(
# #                         control_sequences=controls, **kwargs
# #                     )
# #                 except TypeError:
# #                     # API doesn't accept control_sequences; try without it (still set num_controls>0)
# #                     return spm_mb_structure_learning(**kwargs)
# #             else:
# #                 return spm_mb_structure_learning(**kwargs)

# #         try:
# #             if self.single_group:
# #                 RG0 = [[int(i) for i in range(M)]]
# #                 agents, RG, _ = _run_spm(
# #                     RG_override=[RG0], levels_override=1, with_controls=True
# #                 )
# #             else:
# #                 agents, RG, _ = _run_spm(with_controls=True)
# #             self._controls_used = True
# #         except Exception:
# #             # Fallback #1: uniform layout + single group + big dx
# #             locations_matrix = _synthetic_uniform_layout(M, W_guess, H_guess)
# #             try:
# #                 RG0 = [[int(i) for i in range(M)]]
# #                 agents, RG, _ = _run_spm(
# #                     RG_override=[RG0],
# #                     dx_override=max(W_guess, H_guess),
# #                     levels_override=1,
# #                     with_controls=True,
# #                 )
# #                 self._controls_used = True
# #             except Exception:
# #                 # Fallback #2: drop controls entirely
# #                 try:
# #                     if self.single_group:
# #                         RG0 = [[int(i) for i in range(M)]]
# #                         agents, RG, _ = _run_spm(
# #                             RG_override=[RG0], levels_override=1, with_controls=False
# #                         )
# #                     else:
# #                         agents, RG, _ = _run_spm(with_controls=False)
# #                 except Exception:
# #                     # Fallback #3: trivial identity agent
# #                     class _TrivialAgent:
# #                         def __init__(self, K):
# #                             A_ = np.eye(K, K)[None, :, :]
# #                             self.A = [A_]
# #                             self.B = [np.eye(K, K)[None, :, :, None]]

# #                     self.agent = _TrivialAgent(K)
# #                     self.o2o_default = {o: o for o in range(K)}
# #                     (
# #                         self.action_mode_by_color,
# #                         self.action_global_mode,
# #                         self.color_map,
# #                         self.out_shape,
# #                     ) = _learn_actions_and_colors(
# #                         train_pairs, self.background, self.actions
# #                     )
# #                     return

# #         self.agent = agents[0]
# #         self.o2o_default = _derive_token_transition_from_agent(self.agent)
# #         # Optional: a per-action mapping (useful later); cheap for small A
# #         self.o2o_by_action = {
# #             a: _derive_token_transition_for_action(self.agent, a)
# #             for a in range(min(A, 8))
# #         }

# #         (
# #             self.action_mode_by_color,
# #             self.action_global_mode,
# #             self.color_map,
# #             self.out_shape,
# #         ) = _learn_actions_and_colors(train_pairs, self.background, self.actions)

# #     # ---- predict ----
# #     def predict_grid(self, grid: np.ndarray) -> np.ndarray:
# #         assert self.kmeans is not None and self.agent is not None
# #         H_out, W_out = self.out_shape if self.out_shape is not None else grid.shape
# #         out_bg = infer_background_color(grid)
# #         canvas = np.full((H_out, W_out), out_bg, dtype=grid.dtype)

# #         objs = grid_to_objects_with_tda(grid, background=self.background)
# #         if len(objs) == 0:
# #             return canvas
# #         feats = batch_vectorize_diagrams([o.diagram for o in objs])
# #         toks = self.kmeans.predict(feats)

# #         for i, o in enumerate(objs):
# #             cin = int(o.color)
# #             tok_in = int(toks[i])
# #             # pick action by color prior (fallback global)
# #             a = self.action_mode_by_color.get(cin, self.action_global_mode)
# #             # choose token transition conditioned on action if available, else default
# #             o2o = self.o2o_by_action.get(a, self.o2o_default)
# #             tok_out = o2o.get(tok_in, tok_in)

# #             # recolor by learned map (currently color-based; you can map by token if you wish)
# #             cout = self.color_map.get(cin, cin)

# #             # apply action to geometry
# #             from .arc_actions import (
# #                 ActionLibrary,
# #             )  # to avoid circular import at module level

# #             transformed = self.actions.apply_action_mask(o.mask, a, H_out, W_out)
# #             canvas[transformed] = cout

# #         return canvas

# #     # ---- API ----
# #     def solve_task(self, task: Dict[str, Any]) -> List[np.ndarray]:
# #         self.fit_task(task)
# #         preds: List[np.ndarray] = []
# #         for t in task.get("test", []):
# #             x = grid_from_list(t["input"])
# #             preds.append(self.predict_grid(x))
# #         return preds
# # rgm/arc_rgm_solver_controls.py
# # RGM-based ARC solver using giotto-TDA tokens as observations and
# # *discrete actions as controls* (if fast_structure_learning supports it).

# from __future__ import annotations
# from typing import List, Tuple, Dict, Any, Optional
# import json, argparse, sys
# import numpy as np
# from sklearn.cluster import KMeans

# from rgm.fast_structure_learning import spm_mb_structure_learning
# from rgm.arc_tda_features import (
#     grid_to_objects_with_tda,
#     batch_vectorize_diagrams,
#     wasserstein_matching,
# )
# from rgm.arc_actions import ActionLibrary

# # ---------- small utils ----------
# def grid_from_list(lst: List[List[int]]) -> np.ndarray:
#     return np.asarray(lst, dtype=np.int32)

# def grid_to_list(arr: np.ndarray) -> List[List[int]]:
#     return arr.astype(int).tolist()

# def infer_background_color(grid: np.ndarray) -> int:
#     vals, counts = np.unique(grid, return_counts=True)
#     return int(vals[counts.argmax()])

# def _clip_and_jitter_locations(locs_xy: np.ndarray, W: int, H: int, eps: float = 1e-3) -> np.ndarray:
#     if locs_xy.size == 0:
#         return locs_xy
#     locs = locs_xy.astype(float).copy()
#     locs[:, 0] = np.clip(locs[:, 0], 0.5, max(0.5, W - 1.5))
#     locs[:, 1] = np.clip(locs[:, 1], 0.5, max(0.5, H - 1.5))
#     rng = np.random.RandomState(0)
#     jitter = (rng.rand(*locs.shape) - 0.5) * eps
#     return locs + jitter

# def _synthetic_uniform_layout(M: int, W: int, H: int) -> np.ndarray:
#     if M <= 0:
#         return np.zeros((0, 2), dtype=float)
#     r = int(np.ceil(np.sqrt(M)))
#     c = int(np.ceil(M / r))
#     xs = np.linspace(0.5, max(0.5, W - 1.5), c)
#     ys = np.linspace(0.5, max(0.5, H - 1.5), r)
#     coords = []
#     for i in range(M):
#         rr = i // c; cc = i % c
#         coords.append([xs[min(cc, len(xs)-1)], ys[min(rr, len(ys)-1)]])
#     return np.asarray(coords, dtype=float)

# # ---------- TDA → tokens ----------
# def _extract_objs_and_tokens(grid: np.ndarray, background: int, kmeans: Optional[KMeans]) -> Tuple[List[Any], np.ndarray, np.ndarray]:
#     objs = grid_to_objects_with_tda(grid, background=background)
#     if len(objs) == 0:
#         return [], np.array([], dtype=int), np.zeros((0, 0), dtype=float)
#     feats = batch_vectorize_diagrams([o.diagram for o in objs])
#     toks = kmeans.predict(feats)
#     return objs, toks, feats

# # ---------- A/B decoding ----------
# def _derive_token_transition_from_agent(agent) -> Dict[int, int]:
#     votes: Dict[int, Dict[int, int]] = {}
#     for A_m in agent.A:
#         A = A_m[0]                       # (n_obs, n_states)
#         obs2state = np.argmax(A, axis=1) # (n_obs,)
#         state2obs = np.argmax(A, axis=0) # (n_states,)
#         B = agent.B[0][0, :, :, 0]       # (n_states, n_states)
#         for o in range(A.shape[0]):
#             s  = int(obs2state[o])
#             s2 = int(np.argmax(B[s]))
#             o2 = int(state2obs[s2])
#             votes.setdefault(o, {})
#             votes[o][o2] = votes[o].get(o2, 0) + 1
#     return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}

# def _derive_token_transition_for_action(agent, a: int) -> Dict[int, int]:
#     votes: Dict[int, Dict[int, int]] = {}
#     for A_m in agent.A:
#         A = A_m[0]
#         obs2state = np.argmax(A, axis=1)
#         state2obs = np.argmax(A, axis=0)
#         Bfull = agent.B[0][0]  # (n_states, n_states, n_controls)
#         a_idx = min(a, Bfull.shape[-1]-1)
#         B = Bfull[:, :, a_idx]
#         for o in range(A.shape[0]):
#             s  = int(obs2state[o])
#             s2 = int(np.argmax(B[s]))
#             o2 = int(state2obs[s2])
#             votes.setdefault(o, {})
#             votes[o][o2] = votes[o].get(o2, 0) + 1
#     return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}

# # ---------- action & color priors ----------
# def _learn_actions_and_colors(train_pairs, background: int, actions: ActionLibrary):
#     A = actions.count
#     action_counts_global = np.zeros(A, dtype=np.int64)
#     action_counts_by_color: Dict[int, np.ndarray] = {}
#     color_votes: Dict[int, Dict[int,int]] = {}
#     out_shape = None

#     for xi, yi in train_pairs:
#         out_shape = yi.shape
#         objs_x = grid_to_objects_with_tda(xi, background=background)
#         objs_y = grid_to_objects_with_tda(yi, background=background)
#         if len(objs_x) == 0 or len(objs_y) == 0:
#             continue
#         matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
#         match_map = {i: j for (i, j) in matches}

#         for i, ox in enumerate(objs_x):
#             j = match_map.get(i)
#             cin = int(ox.color)
#             if cin not in action_counts_by_color:
#                 action_counts_by_color[cin] = np.zeros(A, dtype=np.int64)

#             if j is None:
#                 a = 0
#             else:
#                 oy = objs_y[j]
#                 a = actions.best_action_by_iou(ox.mask, oy.mask)
#                 cout = int(oy.color)
#                 color_votes.setdefault(cin, {})
#                 color_votes[cin][cout] = color_votes[cin].get(cout, 0) + 1

#             action_counts_by_color[cin][a] += 1
#             action_counts_global[a] += 1

#     action_mode_by_color = {c: int(np.argmax(cnts)) for c, cnts in action_counts_by_color.items()}
#     action_global_mode = int(np.argmax(action_counts_global)) if action_counts_global.sum() else 0
#     color_map = {cin: max(v.items(), key=lambda kv: kv[1])[0] for cin, v in color_votes.items()}
#     if out_shape is None:
#         out_shape = train_pairs[-1][1].shape
#     return action_mode_by_color, action_global_mode, color_map, out_shape

# # ---------- Solver ----------
# class ARCRGMSolverControls:
#     def __init__(
#         self,
#         n_bins: int = 7,
#         dx: int = 2,
#         background: int = 0,
#         random_state: int = 0,
#         single_group: bool = True,
#         levels: int = 1,
#         trans_radius: int = 1,
#     ):
#         self.n_bins = int(n_bins)
#         self.dx = int(dx)
#         self.background = int(background)
#         self.random_state = int(random_state)
#         self.single_group = bool(single_group)
#         self.levels = int(max(1, levels))
#         self.actions = ActionLibrary(trans_radius=trans_radius)

#         self.kmeans: Optional[KMeans] = None
#         self.agent = None
#         self.o2o_default: Dict[int, int] = {}
#         self.o2o_by_action: Dict[int, Dict[int, int]] = {}
#         self.action_mode_by_color: Dict[int, int] = {}
#         self.action_global_mode: int = 0
#         self.color_map: Dict[int,int] = {}
#         self.out_shape: Tuple[int,int] = None
#         self._controls_used = False

#     def fit_task(self, task: Dict[str, Any]) -> None:
#         # KMeans over TDA features
#         train_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
#         feats_list: List[np.ndarray] = []
#         for p in task["train"]:
#             xi = grid_from_list(p["input"]); yi = grid_from_list(p["output"])
#             train_pairs.append((xi, yi))
#             for g in (xi, yi):
#                 objs = grid_to_objects_with_tda(g, background=self.background)
#                 if len(objs) == 0:
#                     continue
#                 feats = batch_vectorize_diagrams([o.diagram for o in objs])
#                 feats_list.append(feats)
#         if len(feats_list) == 0:
#             self.kmeans = KMeans(n_clusters=max(2, self.n_bins), n_init=10, random_state=self.random_state)
#             self.kmeans.fit(np.zeros((max(2, self.n_bins), 8), dtype=float))
#         else:
#             X = np.vstack(feats_list)
#             k = min(self.n_bins, max(2, min(32, len(np.unique(X, axis=0)))))
#             self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
#             self.kmeans.fit(X)

#         # Build sequences + controls
#         seq_tokens: List[np.ndarray] = []    # (2, K)
#         seq_controls: List[np.ndarray] = []  # (1, A)
#         locs_xy: List[Tuple[float, float]] = []

#         for xi, yi in train_pairs:
#             objs_x, toks_x, _ = _extract_objs_and_tokens(xi, self.background, self.kmeans)
#             objs_y, toks_y, _ = _extract_objs_and_tokens(yi, self.background, self.kmeans)

#             matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
#             match_map = {i: j for (i, j) in matches}
#             K = self.kmeans.n_clusters
#             A = self.actions.count

#             for i, ox in enumerate(objs_x):
#                 onehots = np.zeros((2, K), dtype=np.int32)
#                 controls = np.zeros((1, A), dtype=np.int32)
#                 tok_x = int(toks_x[i]); onehots[0, tok_x] = 1
#                 j = match_map.get(i)
#                 if j is not None:
#                     tok_y = int(toks_y[j]); onehots[1, tok_y] = 1
#                     a = self.actions.best_action_by_iou(ox.mask, objs_y[j].mask)
#                     controls[0, a] = 1
#                 else:
#                     onehots[1, tok_x] = 1
#                     controls[0, 0] = 1
#                 seq_tokens.append(onehots)
#                 seq_controls.append(controls)
#                 cy, cx = ox.centroid_rc
#                 locs_xy.append((float(cx), float(cy)))

#         if len(seq_tokens) == 0:
#             K = max(2, self.kmeans.n_clusters if self.kmeans is not None else 2)
#             A = self.actions.count
#             one_hots = np.zeros((1, 2, K), dtype=np.int32)
#             controls = np.zeros((1, 1, A), dtype=np.int32); controls[0, 0, 0] = 1
#             locations_matrix = np.array([[0.5, 0.5]], dtype=float)
#         else:
#             one_hots = np.stack(seq_tokens, axis=0)     # (M, 2, K)
#             controls = np.stack(seq_controls, axis=0)   # (M, 1, A)
#             H_guess, W_guess = train_pairs[0][0].shape
#             locations_matrix = _clip_and_jitter_locations(np.asarray(locs_xy, dtype=float), W_guess, H_guess)

#         # Structure learning (with controls if supported)
#         H_guess, W_guess = train_pairs[0][0].shape
#         M, _, K = one_hots.shape
#         A = self.actions.count

#         def _run_spm(RG_override=None, dx_override=None, levels_override=None, with_controls=True):
#             kwargs = dict(
#                 observations=one_hots,
#                 locations_matrix=locations_matrix,
#                 size=(W_guess, H_guess, M),
#                 dx=(dx_override if dx_override is not None else self.dx),
#                 num_controls=(A if with_controls else 0),
#                 max_levels=(levels_override if levels_override is not None else self.levels),
#                 agents=None,
#                 RG=RG_override,
#             )
#             if with_controls:
#                 try:
#                     return spm_mb_structure_learning(control_sequences=controls, **kwargs)
#                 except TypeError:
#                     return spm_mb_structure_learning(**kwargs)
#             else:
#                 return spm_mb_structure_learning(**kwargs)

#         try:
#             if self.single_group:
#                 RG0 = [[int(i) for i in range(M)]]
#                 agents, RG, _ = _run_spm(RG_override=[RG0], levels_override=1, with_controls=True)
#             else:
#                 agents, RG, _ = _run_spm(with_controls=True)
#             self._controls_used = True
#         except Exception:
#             locations_matrix = _synthetic_uniform_layout(M, W_guess, H_guess)
#             try:
#                 RG0 = [[int(i) for i in range(M)]]
#                 agents, RG, _ = _run_spm(RG_override=[RG0], dx_override=max(W_guess, H_guess), levels_override=1, with_controls=True)
#                 self._controls_used = True
#             except Exception:
#                 try:
#                     if self.single_group:
#                         RG0 = [[int(i) for i in range(M)]]
#                         agents, RG, _ = _run_spm(RG_override=[RG0], levels_override=1, with_controls=False)
#                     else:
#                         agents, RG, _ = _run_spm(with_controls=False)
#                 except Exception:
#                     class _TrivialAgent:
#                         def __init__(self, K):
#                             A_ = np.eye(K, K)[None, :, :]
#                             self.A = [A_]
#                             self.B = [np.eye(K, K)[None, :, :, None]]
#                     self.agent = _TrivialAgent(K)
#                     self.o2o_default = {o: o for o in range(K)}
#                     self.action_mode_by_color, self.action_global_mode, self.color_map, self.out_shape = \
#                         _learn_actions_and_colors(train_pairs, self.background, self.actions)
#                     return

#         self.agent = agents[0]
#         self.o2o_default = _derive_token_transition_from_agent(self.agent)
#         self.o2o_by_action = {a: _derive_token_transition_for_action(self.agent, a) for a in range(min(A, 8))}
#         self.action_mode_by_color, self.action_global_mode, self.color_map, self.out_shape = \
#             _learn_actions_and_colors(train_pairs, self.background, self.actions)

#     def predict_grid(self, grid: np.ndarray) -> np.ndarray:
#         assert self.kmeans is not None and self.agent is not None
#         H_out, W_out = self.out_shape if self.out_shape is not None else grid.shape
#         out_bg = infer_background_color(grid)
#         canvas = np.full((H_out, W_out), out_bg, dtype=grid.dtype)

#         objs = grid_to_objects_with_tda(grid, background=self.background)
#         if len(objs) == 0:
#             return canvas
#         feats = batch_vectorize_diagrams([o.diagram for o in objs])
#         toks = self.kmeans.predict(feats)

#         for i, o in enumerate(objs):
#             cin = int(o.color)
#             tok_in = int(toks[i])
#             a = self.action_mode_by_color.get(cin, self.action_global_mode)
#             o2o = self.o2o_by_action.get(a, self.o2o_default)
#             tok_out = o2o.get(tok_in, tok_in)
#             cout = self.color_map.get(cin, cin)
#             transformed = self.actions.apply_action_mask(o.mask, a, H_out, W_out)
#             canvas[transformed] = cout
#         return canvas

#     def solve_task(self, task: Dict[str, Any]) -> List[np.ndarray]:
#         self.fit_task(task)
#         preds: List[np.ndarray] = []
#         for t in task.get("test", []):
#             x = grid_from_list(t["input"])
#             preds.append(self.predict_grid(x))
#         return preds

# # ---------- CLI ----------
# def _cli():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("task_json", help="Path to a single ARC task json")
#     ap.add_argument("--k", type=int, default=7)
#     ap.add_argument("--dx", type=int, default=2)
#     ap.add_argument("--levels", type=int, default=1)
#     ap.add_argument("--single_group", action="store_true")
#     ap.add_argument("--bg", type=int, default=0)
#     ap.add_argument("--trans_r", type=int, default=1)
#     args = ap.parse_args()

#     with open(args.task_json, "r") as f:
#         task = json.load(f)

#     solver = ARCRGMSolverControls(
#         n_bins=args.k, dx=args.dx, background=args.bg,
#         single_group=args.single_group, levels=max(1, args.levels),
#         trans_radius=args.trans_r,
#     )
#     preds = solver.solve_task(task)

#     # Debug summary to stderr
#     sys.stderr.write(
#         f"[arc_rgm_solver_controls] controls_used={solver._controls_used} "
#         f"kmeans_k={solver.kmeans.n_clusters if solver.kmeans else None} "
#         f"levels={args.levels} actions={solver.actions.count}\n"
#     )

#     # Print predictions as Kaggle-like single file output for this task
#     out = {"predictions": [grid_to_list(g) for g in preds]}
#     print(json.dumps(out))

# if __name__ == "__main__":
#     _cli()
# rgm/arc_rgm_solver_controls.py
# RGM-based ARC solver using giotto-TDA tokens as observations and
# *discrete actions as controls* (if fast_structure_learning supports it).

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import json, argparse, sys
import numpy as np
from sklearn.cluster import KMeans

from rgm.fast_structure_learning import spm_mb_structure_learning
from rgm.arc_tda_features import (
    grid_to_objects_with_tda,
    batch_vectorize_diagrams,
    wasserstein_matching,
)
from rgm.arc_actions import ActionLibrary


# ---------- small utils ----------
def grid_from_list(lst: List[List[int]]) -> np.ndarray:
    return np.asarray(lst, dtype=np.int32)


def grid_to_list(arr: np.ndarray) -> List[List[int]]:
    return arr.astype(int).tolist()


def infer_background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _clip_and_jitter_locations(
    locs_xy: np.ndarray, W: int, H: int, eps: float = 1e-3
) -> np.ndarray:
    if locs_xy.size == 0:
        return locs_xy
    locs = locs_xy.astype(float).copy()
    locs[:, 0] = np.clip(locs[:, 0], 0.5, max(0.5, W - 1.5))
    locs[:, 1] = np.clip(locs[:, 1], 0.5, max(0.5, H - 1.5))
    rng = np.random.RandomState(0)
    jitter = (rng.rand(*locs.shape) - 0.5) * eps
    return locs + jitter


def _synthetic_uniform_layout(M: int, W: int, H: int) -> np.ndarray:
    if M <= 0:
        return np.zeros((0, 2), dtype=float)
    r = int(np.ceil(np.sqrt(M)))
    c = int(np.ceil(M / r))
    xs = np.linspace(0.5, max(0.5, W - 1.5), c)
    ys = np.linspace(0.5, max(0.5, H - 1.5), r)
    coords = []
    for i in range(M):
        rr = i // c
        cc = i % c
        coords.append([xs[min(cc, len(xs) - 1)], ys[min(rr, len(ys) - 1)]])
    return np.asarray(coords, dtype=float)


# ---------- TDA → tokens ----------
def _extract_objs_and_tokens(
    grid: np.ndarray, background: int, kmeans: Optional[KMeans]
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    objs = grid_to_objects_with_tda(grid, background=background)
    if len(objs) == 0:
        return [], np.array([], dtype=int), np.zeros((0, 0), dtype=float)
    feats = batch_vectorize_diagrams([o.diagram for o in objs])
    toks = kmeans.predict(feats)
    return objs, toks, feats


# ---------- A/B decoding ----------
def _derive_token_transition_from_agent(agent) -> Dict[int, int]:
    votes: Dict[int, Dict[int, int]] = {}
    for A_m in agent.A:
        A = A_m[0]  # (n_obs, n_states)
        obs2state = np.argmax(A, axis=1)  # (n_obs,)
        state2obs = np.argmax(A, axis=0)  # (n_states,)
        B = agent.B[0][0, :, :, 0]  # (n_states, n_states)
        for o in range(A.shape[0]):
            s = int(obs2state[o])
            s2 = int(np.argmax(B[s]))
            o2 = int(state2obs[s2])
            votes.setdefault(o, {})
            votes[o][o2] = votes[o].get(o2, 0) + 1
    return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}


def _derive_token_transition_for_action(agent, a: int) -> Dict[int, int]:
    votes: Dict[int, Dict[int, int]] = {}
    for A_m in agent.A:
        A = A_m[0]
        obs2state = np.argmax(A, axis=1)
        state2obs = np.argmax(A, axis=0)
        Bfull = agent.B[0][0]  # (n_states, n_states, n_controls)
        a_idx = min(a, Bfull.shape[-1] - 1)
        B = Bfull[:, :, a_idx]
        for o in range(A.shape[0]):
            s = int(obs2state[o])
            s2 = int(np.argmax(B[s]))
            o2 = int(state2obs[s2])
            votes.setdefault(o, {})
            votes[o][o2] = votes[o].get(o2, 0) + 1
    return {o: max(d.items(), key=lambda kv: kv[1])[0] for o, d in votes.items()}


# ---------- action & color priors ----------
def _learn_actions_and_colors(train_pairs, background: int, actions: ActionLibrary):
    A = actions.count
    action_counts_global = np.zeros(A, dtype=np.int64)
    action_counts_by_color: Dict[int, np.ndarray] = {}
    color_votes: Dict[int, Dict[int, int]] = {}
    out_shape = None

    for xi, yi in train_pairs:
        out_shape = yi.shape
        objs_x = grid_to_objects_with_tda(xi, background=background)
        objs_y = grid_to_objects_with_tda(yi, background=background)
        if len(objs_x) == 0 or len(objs_y) == 0:
            continue
        matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
        match_map = {i: j for (i, j) in matches}

        for i, ox in enumerate(objs_x):
            j = match_map.get(i)
            cin = int(ox.color)
            if cin not in action_counts_by_color:
                action_counts_by_color[cin] = np.zeros(A, dtype=np.int64)

            if j is None:
                a = 0
            else:
                oy = objs_y[j]
                a = actions.best_action_by_iou(ox.mask, oy.mask)
                cout = int(oy.color)
                color_votes.setdefault(cin, {})
                color_votes[cin][cout] = color_votes[cin].get(cout, 0) + 1

            action_counts_by_color[cin][a] += 1
            action_counts_global[a] += 1

    action_mode_by_color = {
        c: int(np.argmax(cnts)) for c, cnts in action_counts_by_color.items()
    }
    action_global_mode = (
        int(np.argmax(action_counts_global)) if action_counts_global.sum() else 0
    )
    color_map = {
        cin: max(v.items(), key=lambda kv: kv[1])[0] for cin, v in color_votes.items()
    }
    if out_shape is None:
        out_shape = train_pairs[-1][1].shape
    return action_mode_by_color, action_global_mode, color_map, out_shape


# ---------- Solver ----------
class ARCRGMSolverControls:
    def __init__(
        self,
        n_bins: int = 7,
        dx: int = 2,
        background: int = 0,
        random_state: int = 0,
        single_group: bool = True,
        levels: int = 1,
        trans_radius: int = 1,
    ):
        self.n_bins = int(n_bins)
        self.dx = int(dx)
        self.background = int(background)
        self.random_state = int(random_state)
        self.single_group = bool(single_group)
        self.levels = int(max(1, levels))
        self.actions = ActionLibrary(trans_radius=trans_radius)

        self.kmeans: Optional[KMeans] = None
        self.agent = None
        self.o2o_default: Dict[int, int] = {}
        self.o2o_by_action: Dict[int, Dict[int, int]] = {}
        self.action_mode_by_color: Dict[int, int] = {}
        self.action_global_mode: int = 0
        self.color_map: Dict[int, int] = {}
        self.out_shape: Tuple[int, int] = None
        self._controls_used = False

    def fit_task(self, task: Dict[str, Any]) -> None:
        # KMeans over TDA features
        train_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        feats_list: List[np.ndarray] = []
        for p in task["train"]:
            xi = grid_from_list(p["input"])
            yi = grid_from_list(p["output"])
            train_pairs.append((xi, yi))
            for g in (xi, yi):
                objs = grid_to_objects_with_tda(g, background=self.background)
                if len(objs) == 0:
                    continue
                feats = batch_vectorize_diagrams([o.diagram for o in objs])
                feats_list.append(feats)
        if len(feats_list) == 0:
            self.kmeans = KMeans(
                n_clusters=max(2, self.n_bins),
                n_init=10,
                random_state=self.random_state,
            )
            self.kmeans.fit(np.zeros((max(2, self.n_bins), 8), dtype=float))
        else:
            X = np.vstack(feats_list)
            k = min(self.n_bins, max(2, min(32, len(np.unique(X, axis=0)))))
            self.kmeans = KMeans(
                n_clusters=k, n_init=10, random_state=self.random_state
            )
            self.kmeans.fit(X)

        # Build sequences + controls
        seq_tokens: List[np.ndarray] = []  # (2, K)
        seq_controls: List[np.ndarray] = []  # (1, A)
        locs_xy: List[Tuple[float, float]] = []

        for xi, yi in train_pairs:
            objs_x, toks_x, _ = _extract_objs_and_tokens(
                xi, self.background, self.kmeans
            )
            objs_y, toks_y, _ = _extract_objs_and_tokens(
                yi, self.background, self.kmeans
            )

            matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
            match_map = {i: j for (i, j) in matches}
            K = self.kmeans.n_clusters
            A = self.actions.count

            for i, ox in enumerate(objs_x):
                onehots = np.zeros((2, K), dtype=np.int32)
                controls = np.zeros((1, A), dtype=np.int32)
                tok_x = int(toks_x[i])
                onehots[0, tok_x] = 1
                j = match_map.get(i)
                if j is not None:
                    tok_y = int(toks_y[j])
                    onehots[1, tok_y] = 1
                    a = self.actions.best_action_by_iou(ox.mask, objs_y[j].mask)
                    controls[0, a] = 1
                else:
                    onehots[1, tok_x] = 1
                    controls[0, 0] = 1
                seq_tokens.append(onehots)
                seq_controls.append(controls)
                cy, cx = ox.centroid_rc
                locs_xy.append((float(cx), float(cy)))

        if len(seq_tokens) == 0:
            K = max(2, self.kmeans.n_clusters if self.kmeans is not None else 2)
            A = self.actions.count
            one_hots = np.zeros((1, 2, K), dtype=np.int32)
            controls = np.zeros((1, 1, A), dtype=np.int32)
            controls[0, 0, 0] = 1
            locations_matrix = np.array([[0.5, 0.5]], dtype=float)
        else:
            one_hots = np.stack(seq_tokens, axis=0)  # (M, 2, K)
            controls = np.stack(seq_controls, axis=0)  # (M, 1, A)
            H_guess, W_guess = train_pairs[0][0].shape
            locations_matrix = _clip_and_jitter_locations(
                np.asarray(locs_xy, dtype=float), W_guess, H_guess
            )

        # Structure learning (with controls if supported)
        H_guess, W_guess = train_pairs[0][0].shape
        M, _, K = one_hots.shape
        A = self.actions.count

        def _run_spm(
            RG_override=None, dx_override=None, levels_override=None, with_controls=True
        ):
            kwargs = dict(
                observations=one_hots,
                locations_matrix=locations_matrix,
                size=(W_guess, H_guess, M),
                dx=(dx_override if dx_override is not None else self.dx),
                num_controls=(A if with_controls else 0),
                max_levels=(
                    levels_override if levels_override is not None else self.levels
                ),
                agents=None,
                RG=RG_override,
            )
            if with_controls:
                try:
                    return spm_mb_structure_learning(
                        control_sequences=controls, **kwargs
                    )
                except TypeError:
                    return spm_mb_structure_learning(**kwargs)
            else:
                return spm_mb_structure_learning(**kwargs)

        try:
            if self.single_group:
                RG0 = [[int(i) for i in range(M)]]
                agents, RG, _ = _run_spm(
                    RG_override=[RG0], levels_override=1, with_controls=True
                )
            else:
                agents, RG, _ = _run_spm(with_controls=True)
            self._controls_used = True
        except Exception:
            locations_matrix = _synthetic_uniform_layout(M, W_guess, H_guess)
            try:
                RG0 = [[int(i) for i in range(M)]]
                agents, RG, _ = _run_spm(
                    RG_override=[RG0],
                    dx_override=max(W_guess, H_guess),
                    levels_override=1,
                    with_controls=True,
                )
                self._controls_used = True
            except Exception:
                try:
                    if self.single_group:
                        RG0 = [[int(i) for i in range(M)]]
                        agents, RG, _ = _run_spm(
                            RG_override=[RG0], levels_override=1, with_controls=False
                        )
                    else:
                        agents, RG, _ = _run_spm(with_controls=False)
                except Exception:

                    class _TrivialAgent:
                        def __init__(self, K):
                            A_ = np.eye(K, K)[None, :, :]
                            self.A = [A_]
                            self.B = [np.eye(K, K)[None, :, :, None]]

                    self.agent = _TrivialAgent(K)
                    self.o2o_default = {o: o for o in range(K)}
                    (
                        self.action_mode_by_color,
                        self.action_global_mode,
                        self.color_map,
                        self.out_shape,
                    ) = _learn_actions_and_colors(
                        train_pairs, self.background, self.actions
                    )
                    return

        self.agent = agents[0]
        self.o2o_default = _derive_token_transition_from_agent(self.agent)
        self.o2o_by_action = {
            a: _derive_token_transition_for_action(self.agent, a)
            for a in range(min(A, 8))
        }
        (
            self.action_mode_by_color,
            self.action_global_mode,
            self.color_map,
            self.out_shape,
        ) = _learn_actions_and_colors(train_pairs, self.background, self.actions)

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        assert self.kmeans is not None and self.agent is not None
        H_out, W_out = self.out_shape if self.out_shape is not None else grid.shape
        out_bg = infer_background_color(grid)
        canvas = np.full((H_out, W_out), out_bg, dtype=grid.dtype)

        objs = grid_to_objects_with_tda(grid, background=self.background)
        if len(objs) == 0:
            return canvas
        feats = batch_vectorize_diagrams([o.diagram for o in objs])
        toks = self.kmeans.predict(feats)

        for i, o in enumerate(objs):
            cin = int(o.color)
            tok_in = int(toks[i])
            a = self.action_mode_by_color.get(cin, self.action_global_mode)
            o2o = self.o2o_by_action.get(a, self.o2o_default)
            tok_out = o2o.get(tok_in, tok_in)
            cout = self.color_map.get(cin, cin)
            transformed = self.actions.apply_action_mask(o.mask, a, H_out, W_out)
            canvas[transformed] = cout
        return canvas

    def solve_task(self, task: Dict[str, Any]) -> List[np.ndarray]:
        self.fit_task(task)
        preds: List[np.ndarray] = []
        for t in task.get("test", []):
            x = grid_from_list(t["input"])
            preds.append(self.predict_grid(x))
        return preds


# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_json", help="Path to a single ARC task json")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--dx", type=int, default=2)
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--single_group", action="store_true")
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--trans_r", type=int, default=1)
    args = ap.parse_args()

    with open(args.task_json, "r") as f:
        task = json.load(f)

    solver = ARCRGMSolverControls(
        n_bins=args.k,
        dx=args.dx,
        background=args.bg,
        single_group=args.single_group,
        levels=max(1, args.levels),
        trans_radius=args.trans_r,
    )
    preds = solver.solve_task(task)

    # Debug summary to stderr
    sys.stderr.write(
        f"[arc_rgm_solver_controls] controls_used={solver._controls_used} "
        f"kmeans_k={solver.kmeans.n_clusters if solver.kmeans else None} "
        f"levels={args.levels} actions={solver.actions.count}\n"
    )

    # Print predictions as Kaggle-like single file output for this task
    out = {"predictions": [grid_to_list(g) for g in preds]}
    print(json.dumps(out))


if __name__ == "__main__":
    _cli()
