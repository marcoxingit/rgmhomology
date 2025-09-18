# # # # # run_arc_solver.py
# # # # from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
# # # # from arc_tda_features import set_matching_metric
# # # # import sys, json

# # # # def main():
# # # #     if len(sys.argv) < 2:
# # # #         print("Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w]")
# # # #         return
# # # #     path = sys.argv[1]
# # # #     match = None
# # # #     if len(sys.argv) >= 4 and sys.argv[2] == "--match":
# # # #         match = sys.argv[3]
# # # #     if match:
# # # #         set_matching_metric(match)
# # # #     task = load_arc_task_json(path)
# # # #     solver = ARCSolver()
# # # #     preds = solver.solve_task(task)
# # # #     out = {"predictions": [grid_to_list(g) for g in preds]}
# # # #     print(json.dumps(out))

# # # # if __name__ == "__main__":
# # # #     main()
# # # # run_arc_solver.py
# # # # (single-task runner; --match, --attempt2, --kaggle for Kaggle JSON)
# # # from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
# # # from arc_tda_features import set_matching_metric
# # # import sys, json, os
# # # import numpy as np


# # # def make_second_attempt(pred: np.ndarray, mode: str = "duplicate") -> np.ndarray:
# # #     mode = (mode or "duplicate").lower()
# # #     if mode == "duplicate":
# # #         return pred.copy()
# # #     if mode == "flip_h":
# # #         return np.asanyarray(pred)[:, ::-1]
# # #     if mode == "flip_v":
# # #         return np.asanyarray(pred)[::-1, :]
# # #     if mode == "transpose":
# # #         return np.asanyarray(pred).T
# # #     if mode == "rotate90":
# # #         return np.rot90(np.asanyarray(pred), 1)
# # #     if mode == "rotate270":
# # #         return np.rot90(np.asanyarray(pred), 3)
# # #     return pred.copy()


# # # def main():
# # #     if len(sys.argv) < 2:
# # #         print(
# # #             "Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w] [--attempt2 MODE] [--kaggle]"
# # #         )
# # #         return
# # #     path = sys.argv[1]
# # #     # parse simple flags
# # #     match = None
# # #     attempt2 = "duplicate"
# # #     kaggle = False
# # #     if "--match" in sys.argv:
# # #         i = sys.argv.index("--match")
# # #         if i + 1 < len(sys.argv):
# # #             match = sys.argv[i + 1]
# # #     if "--attempt2" in sys.argv:
# # #         i = sys.argv.index("--attempt2")
# # #         if i + 1 < len(sys.argv):
# # #             attempt2 = sys.argv[i + 1]
# # #     if "--kaggle" in sys.argv:
# # #         kaggle = True
# # #     if match:
# # #         set_matching_metric(match)
# # #     task = load_arc_task_json(path)
# # #     solver = ARCSolver()
# # #     preds1 = solver.solve_task(task)
# # #     if kaggle:
# # #         # emit two attempts per test
# # #         preds2 = [make_second_attempt(g, mode=attempt2) for g in preds1]
# # #         tid = os.path.splitext(os.path.basename(path))[0]
# # #         out = {
# # #             tid: [
# # #                 {"attempt_1": grid_to_list(a1), "attempt_2": grid_to_list(a2)}
# # #                 for a1, a2 in zip(preds1, preds2)
# # #             ]
# # #         }
# # #         print(json.dumps(out))
# # #     else:
# # #         out = {"predictions": [grid_to_list(g) for g in preds1]}
# # #         print(json.dumps(out))


# # # if __name__ == "__main__":
# # #     main()

# # from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
# # from arc_tda_features import set_matching_metric
# # import sys, json, os
# # import numpy as np


# # def make_second_attempt(pred: np.ndarray, mode: str = "duplicate") -> np.ndarray:
# #     mode = (mode or "duplicate").lower()
# #     if mode == "duplicate":
# #         return pred.copy()
# #     if mode == "flip_h":
# #         return np.asanyarray(pred)[:, ::-1]
# #     if mode == "flip_v":
# #         return np.asanyarray(pred)[::-1, :]
# #     if mode == "transpose":
# #         return np.asanyarray(pred).T
# #     if mode == "rotate90":
# #         return np.rot90(np.asanyarray(pred), 1)
# #     if mode == "rotate270":
# #         return np.rot90(np.asanyarray(pred), 3)
# #     return pred.copy()


# # def main():
# #     if len(sys.argv) < 2:
# #         print(
# #             "Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w] [--attempt2 MODE] [--kaggle]"
# #         )
# #         return
# #     path = sys.argv[1]
# #     match = None
# #     attempt2 = "duplicate"
# #     kaggle = False
# #     if "--match" in sys.argv:
# #         i = sys.argv.index("--match")
# #         if i + 1 < len(sys.argv):
# #             match = sys.argv[i + 1]
# #     if "--attempt2" in sys.argv:
# #         i = sys.argv.index("--attempt2")
# #         if i + 1 < len(sys.argv):
# #             attempt2 = sys.argv[i + 1]
# #     if "--kaggle" in sys.argv:
# #         kaggle = True
# #     if match:
# #         set_matching_metric(match)
# #     task = load_arc_task_json(path)
# #     solver = ARCSolver()
# #     preds1 = solver.solve_task(task)
# #     if kaggle:
# #         preds2 = [make_second_attempt(g, mode=attempt2) for g in preds1]
# #         tid = os.path.splitext(os.path.basename(path))[0]
# #         out = {
# #             tid: [
# #                 {"attempt_1": grid_to_list(a1), "attempt_2": grid_to_list(a2)}
# #                 for a1, a2 in zip(preds1, preds2)
# #             ]
# #         }
# #         print(json.dumps(out))
# #     else:
# #         out = {"predictions": [grid_to_list(g) for g in preds1]}
# #         print(json.dumps(out))


# # if __name__ == "__main__":
# #     main()
# from __future__ import annotations
# import json
# from typing import List, Tuple, Dict, Any
# import numpy as np

# from arc_rules import infer_rule  # rule-based learner
# from arc_transforms import apply_dihedral, DIHEDRAL_CODES

# # TDA helpers (already in your repo)
# from arc_tda_features import grid_to_objects_with_tda, wasserstein_matching


# # --- IO helpers (runners import these) ---
# def grid_from_list(lst: List[List[int]]) -> np.ndarray:
#     return np.asarray(lst, dtype=np.int32)


# def grid_to_list(arr: np.ndarray) -> List[List[int]]:
#     return arr.astype(int).tolist()


# def load_arc_task_json(path: str) -> Dict[str, Any]:
#     with open(path, "r") as f:
#         return json.load(f)


# # --- small utils ---
# def infer_background_color(grid: np.ndarray) -> int:
#     vals, counts = np.unique(grid, return_counts=True)
#     return int(vals[counts.argmax()])


# def move_mask_into(
#     canvas: np.ndarray, mask: np.ndarray, dy: int, dx: int, color: int
# ) -> None:
#     """Paste 'color' where mask==True translated by (dy,dx) with clipping into 'canvas' in-place."""
#     H, W = canvas.shape
#     ys, xs = np.where(mask)
#     if ys.size == 0:
#         return
#     y2 = ys + int(dy)
#     x2 = xs + int(dx)
#     keep = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
#     y2 = y2[keep]
#     x2 = x2[keep]
#     canvas[y2, x2] = color


# # =======================
# # TDA per-object learner
# # =======================
# def _tda_best_dihedral(
#     train_pairs: List[Tuple[np.ndarray, np.ndarray]], background: int
# ) -> str:
#     """Pick dihedral that minimizes total TDA matching cost across train pairs."""
#     best_code = "I"
#     best_cost = float("inf")
#     for code in DIHEDRAL_CODES:
#         tot = 0.0
#         feasible = True
#         for xi, yi in train_pairs:
#             x_d = apply_dihedral(xi, code)
#             objs_x = grid_to_objects_with_tda(x_d, background=background)
#             objs_y = grid_to_objects_with_tda(yi, background=background)
#             if len(objs_x) == 0 or len(objs_y) == 0:
#                 # If either has no object, skip cost for this pair to avoid bias
#                 continue
#             matches, D = wasserstein_matching(objs_x, objs_y, return_matrix=True)
#             if D is None or len(matches) == 0:
#                 feasible = False
#                 break
#             tot += float(sum(D[i, j] for (i, j) in matches))
#         if feasible and tot < best_cost:
#             best_cost = tot
#             best_code = code
#     return best_code


# def _tda_learn_motion(
#     train_pairs: List[Tuple[np.ndarray, np.ndarray]], background: int
# ) -> Tuple[str, Dict[int, int], Dict[int, Tuple[int, int]], Tuple[int, int], int]:
#     """
#     Return:
#       dihedral_code,
#       color_map: in_color -> out_color,
#       deltas_per_color: in_color -> (dy, dx),
#       global_delta: (dy, dx),
#       out_background: majority background from outputs
#     """
#     # Pick dihedral
#     code = _tda_best_dihedral(train_pairs, background=background)

#     # Aggregate matches across train pairs under chosen dihedral
#     colmap_counts: Dict[Tuple[int, int], int] = {}
#     deltas_by_color: Dict[int, List[Tuple[int, int]]] = {}
#     all_deltas: List[Tuple[int, int]] = []
#     out_bg_votes: List[int] = []

#     for xi, yi in train_pairs:
#         out_bg_votes.append(infer_background_color(yi))
#         x_d = apply_dihedral(xi, code)
#         objs_x = grid_to_objects_with_tda(x_d, background=background)
#         objs_y = grid_to_objects_with_tda(yi, background=background)
#         if len(objs_x) == 0 or len(objs_y) == 0:
#             continue
#         matches, _ = wasserstein_matching(objs_x, objs_y, return_matrix=False)
#         # For each matched pair, collect color map votes and centroid deltas
#         for i, j in matches:
#             ox, oy = objs_x[i], objs_y[j]
#             col_in, col_out = int(ox.color), int(oy.color)
#             colmap_counts[(col_in, col_out)] = (
#                 colmap_counts.get((col_in, col_out), 0) + 1
#             )
#             # centroid in the dihedral-transformed input frame
#             cy_in, cx_in = ox.centroid_rc
#             cy_out, cx_out = oy.centroid_rc
#             dy = int(np.round(cy_out - cy_in))
#             dx = int(np.round(cx_out - cx_in))
#             deltas_by_color.setdefault(col_in, []).append((dy, dx))
#             all_deltas.append((dy, dx))

#     # Build color map: for each input color, choose the most common output color
#     color_map: Dict[int, int] = {}
#     if colmap_counts:
#         # group by in_color
#         votes_by_in: Dict[int, Dict[int, int]] = {}
#         for (cin, cout), cnt in colmap_counts.items():
#             votes_by_in.setdefault(cin, {})
#             votes_by_in[cin][cout] = votes_by_in[cin].get(cout, 0) + cnt
#         for cin, d in votes_by_in.items():
#             cout = max(d.items(), key=lambda kv: kv[1])[0]
#             color_map[cin] = cout

#     # Per-color deltas as median (robust)
#     deltas_per_color: Dict[int, Tuple[int, int]] = {}
#     for cin, lst in deltas_by_color.items():
#         dys = np.array([p[0] for p in lst], dtype=int)
#         dxs = np.array([p[1] for p in lst], dtype=int)
#         dy_med = int(np.median(dys))
#         dx_med = int(np.median(dxs))
#         deltas_per_color[cin] = (dy_med, dx_med)

#     # Global delta as median across all
#     if all_deltas:
#         dy_med = int(np.median([d[0] for d in all_deltas]))
#         dx_med = int(np.median([d[1] for d in all_deltas]))
#         global_delta = (dy_med, dx_med)
#     else:
#         global_delta = (0, 0)

#     # Output background
#     out_bg = int(np.median(out_bg_votes)) if out_bg_votes else background

#     return code, color_map, deltas_per_color, global_delta, out_bg


# def _tda_predict_one(
#     x: np.ndarray,
#     dihedral_code: str,
#     color_map: Dict[int, int],
#     deltas_per_color: Dict[int, Tuple[int, int]],
#     global_delta: Tuple[int, int],
#     out_background: int,
#     background: int,
# ) -> np.ndarray:
#     # Apply global dihedral
#     x_d = apply_dihedral(x, dihedral_code)
#     # Extract objects on transformed input
#     objs = grid_to_objects_with_tda(x_d, background=background)
#     H, W = x_d.shape
#     canvas = np.full((H, W), out_background, dtype=x_d.dtype)

#     # Render each object with per-color delta and color map
#     for o in objs:
#         cin = int(o.color)
#         cout = color_map.get(cin, cin)
#         dy, dx = deltas_per_color.get(cin, global_delta)
#         move_mask_into(canvas, o.mask, dy, dx, cout)

#     return canvas


# # =======================
# # Unified solver
# # =======================
# class ARCSolver:
#     def __init__(self, background: int = 0, mode: str = "rule"):
#         """
#         mode:
#           - 'rule': dihedral+color+translation+crop/border/tile (fast)
#           - 'tda' : per-object TDA matching (slower, better on motion tasks)
#         """
#         self.background = background
#         self.mode = (mode or "rule").lower()

#     def solve_task(self, task: Dict[str, Any]) -> List[np.ndarray]:
#         train_pairs = []
#         for p in task["train"]:
#             x = grid_from_list(p["input"])
#             y = grid_from_list(p["output"])
#             train_pairs.append((x, y))

#         preds: List[np.ndarray] = []

#         if self.mode == "rule":
#             rule1, rule2, _ = infer_rule(train_pairs, background=self.background)
#             for t in task["test"]:
#                 x = grid_from_list(t["input"])
#                 y1 = rule1.apply(x)
#                 preds.append(y1)
#             return preds

#         # --- TDA mode ---
#         dihedral_code, color_map, deltas_per_color, global_delta, out_bg = (
#             _tda_learn_motion(train_pairs, background=self.background)
#         )
#         for t in task["test"]:
#             x = grid_from_list(t["input"])
#             y = _tda_predict_one(
#                 x,
#                 dihedral_code=dihedral_code,
#                 color_map=color_map,
#                 deltas_per_color=deltas_per_color,
#                 global_delta=global_delta,
#                 out_background=out_bg,
#                 background=self.background,
#             )
#             preds.append(y)
#         return preds
from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
from arc_tda_features import set_matching_metric
import sys, json, os
import numpy as np


def make_second_attempt(pred: np.ndarray, mode: str = "duplicate") -> np.ndarray:
    mode = (mode or "duplicate").lower()
    if mode == "duplicate":
        return pred.copy()
    if mode == "flip_h":
        return np.asanyarray(pred)[:, ::-1]
    if mode == "flip_v":
        return np.asanyarray(pred)[::-1, :]
    if mode == "transpose":
        return np.asanyarray(pred).T
    if mode == "rotate90":
        return np.rot90(np.asanyarray(pred), 1)
    if mode == "rotate270":
        return np.rot90(np.asanyarray(pred), 3)
    return pred.copy()


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python run_arc_solver.py /path/to/task.json [--mode rule|tda] [--match custom|feat-l2|gtda-w] [--attempt2 MODE] [--kaggle]"
        )
        return
    path = sys.argv[1]
    mode = "rule"
    match = None
    attempt2 = "duplicate"
    kaggle = False
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
    if "--match" in sys.argv:
        i = sys.argv.index("--match")
        if i + 1 < len(sys.argv):
            match = sys.argv[i + 1]
    if "--attempt2" in sys.argv:
        i = sys.argv.index("--attempt2")
        if i + 1 < len(sys.argv):
            attempt2 = sys.argv[i + 1]
    if "--kaggle" in sys.argv:
        kaggle = True

    if match:
        set_matching_metric(match)

    task = load_arc_task_json(path)
    solver = ARCSolver(mode=mode)
    preds1 = solver.solve_task(task)

    if kaggle:
        preds2 = [make_second_attempt(g, mode=attempt2) for g in preds1]
        tid = os.path.splitext(os.path.basename(path))[0]
        out = {
            tid: [
                {"attempt_1": grid_to_list(a1), "attempt_2": grid_to_list(a2)}
                for a1, a2 in zip(preds1, preds2)
            ]
        }
        print(json.dumps(out))
    else:
        out = {"predictions": [grid_to_list(g) for g in preds1]}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
