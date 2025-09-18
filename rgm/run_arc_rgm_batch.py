# # run_arc_rgm_batch.py
# import argparse, os, json, glob, random, traceback
# import numpy as np
# from typing import List, Dict, Any
# from arc_rgm_solver import ARCRGMSolver, grid_from_list, grid_to_list


# def iter_tasks(task_dir: str):
#     return sorted(glob.glob(os.path.join(task_dir, "*.json")))


# def task_pass_at_2(task, preds1, preds2):
#     y_true = []
#     for t in task.get("test", []):
#         if "output" not in t:
#             return (0, 0)
#         y_true.append(grid_from_list(t["output"]))
#     if len(y_true) != len(preds1):
#         return (0, len(y_true))
#     total = len(y_true)
#     good = 0
#     for i in range(total):
#         gt = y_true[i]
#         p1 = preds1[i]
#         p2 = preds2[i] if i < len(preds2) else p1
#         ok = (p1.shape == gt.shape and np.array_equal(p1, gt)) or (
#             p2.shape == gt.shape and np.array_equal(p2, gt)
#         )
#         good += 1 if ok else 0
#     return (good, total)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--tasks", required=True)
#     ap.add_argument("--out", required=True)
#     ap.add_argument("--max", type=int, default=0)
#     ap.add_argument("--shuffle", action="store_true")
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--eval", action="store_true")
#     ap.add_argument("--k", type=int, default=7, help="KMeans token bins")
#     ap.add_argument("--dx", type=int, default=2)
#     ap.add_argument("--bg", type=int, default=0)
#     ap.add_argument(
#         "--attempt2",
#         type=str,
#         default="duplicate",
#         choices=["duplicate", "flip_h", "flip_v", "transpose", "rotate90", "rotate270"],
#     )
#     ap.add_argument("--metric", type=str, default="arc", choices=["arc", "exact"])
#     args = ap.parse_args()

#     paths = iter_tasks(args.tasks)
#     if args.shuffle:
#         random.seed(args.seed)
#         random.shuffle(paths)
#     if args.max and args.max < len(paths):
#         paths = paths[: args.max]

#     os.makedirs(args.out, exist_ok=True)

#     arc_eval = arc_correct = 0
#     ex_eval = ex_correct = 0

#     for i, p in enumerate(paths, 1):
#         tid = os.path.splitext(os.path.basename(p))[0]
#         try:
#             with open(p, "r") as f:
#                 task = json.load(f)

#             solver = ARCRGMSolver(n_bins=args.k, dx=args.dx, background=args.bg)
#             preds1 = solver.solve_task(task)

#             out_path = os.path.join(args.out, os.path.basename(p))
#             with open(out_path, "w") as f:
#                 json.dump({"predictions": [grid_to_list(g) for g in preds1]}, f)

#             msg = ""
#             if args.eval:
#                 if args.metric == "arc":
#                     # Use same prediction twice (we don't generate a second attempt here yet)
#                     preds2 = preds1
#                     good, tot = task_pass_at_2(task, preds1, preds2)
#                     if tot > 0:
#                         arc_correct += good
#                         arc_eval += tot
#                         msg = f" (pass@2 {good}/{tot})"
#                     else:
#                         msg = " (no GT)"
#                 else:
#                     y_true = []
#                     for t in task.get("test", []):
#                         if "output" not in t:
#                             y_true = None
#                             break
#                         y_true.append(grid_from_list(t["output"]))
#                     if y_true is not None and len(y_true) == len(preds1):
#                         ok = all(
#                             a.shape == b.shape and np.array_equal(a, b)
#                             for a, b in zip(preds1, y_true)
#                         )
#                         ex_correct += 1 if ok else 0
#                         ex_eval += 1
#                         msg = " (correct)" if ok else " (wrong)"
#                     else:
#                         msg = " (no GT)"

#             print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}{msg}")

#         except Exception as e:
#             print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")
#             traceback.print_exc()

#     if args.eval:
#         if args.metric == "arc" and arc_eval > 0:
#             print(
#                 f"==> ARC metric (pass@2) accuracy: {arc_correct}/{arc_eval} = {arc_correct/arc_eval:.3f}"
#             )
#         if args.metric == "exact" and ex_eval > 0:
#             print(
#                 f"==> Exact-match task accuracy: {ex_correct}/{ex_eval} = {ex_correct/ex_eval:.3f}"
#             )


# if __name__ == "__main__":
#     main()
import argparse, os, json, glob, random, traceback
import numpy as np
from arc_rgm_solver import ARCRGMSolver, grid_from_list, grid_to_list


def iter_tasks(task_dir: str):
    return sorted(glob.glob(os.path.join(task_dir, "*.json")))


def task_pass_at_2(task, preds1, preds2):
    y_true = []
    for t in task.get("test", []):
        if "output" not in t:
            return (0, 0)
        y_true.append(grid_from_list(t["output"]))
    if len(y_true) != len(preds1):
        return (0, len(y_true))
    total = len(y_true)
    good = 0
    for i in range(total):
        gt = y_true[i]
        p1 = preds1[i]
        p2 = preds2[i] if i < len(preds2) else p1
        ok = (p1.shape == gt.shape and np.array_equal(p1, gt)) or (
            p2.shape == gt.shape and np.array_equal(p2, gt)
        )
        good += 1 if ok else 0
    return (good, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--k", type=int, default=7, help="KMeans token bins")
    ap.add_argument("--dx", type=int, default=2)
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--levels", type=int, default=1, help="RGM max levels")
    ap.add_argument(
        "--single_group", action="store_true", help="force single group at level 0"
    )
    ap.add_argument("--metric", type=str, default="arc", choices=["arc", "exact"])
    args = ap.parse_args()

    paths = iter_tasks(args.tasks)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    if args.max and args.max < len(paths):
        paths = paths[: args.max]

    os.makedirs(args.out, exist_ok=True)

    arc_eval = arc_correct = 0
    ex_eval = ex_correct = 0

    for i, p in enumerate(paths, 1):
        try:
            with open(p, "r") as f:
                task = json.load(f)

            solver = ARCRGMSolver(
                n_bins=args.k,
                dx=args.dx,
                background=args.bg,
                single_group=args.single_group,  # <-- respect the flag
                levels=max(1, args.levels),  # <-- allow multi-level
            )
            preds1 = solver.solve_task(task)

            out_path = os.path.join(args.out, os.path.basename(p))
            with open(out_path, "w") as f:
                json.dump({"predictions": [grid_to_list(g) for g in preds1]}, f)

            msg = ""
            if args.eval:
                if args.metric == "arc":
                    preds2 = preds1
                    good, tot = task_pass_at_2(task, preds1, preds2)
                    if tot > 0:
                        arc_correct += good
                        arc_eval += tot
                        msg = f" (pass@2 {good}/{tot})"
                    else:
                        msg = " (no GT)"
                else:
                    y_true = []
                    for t in task.get("test", []):
                        if "output" not in t:
                            y_true = None
                            break
                        y_true.append(grid_from_list(t["output"]))
                    if y_true is not None and len(y_true) == len(preds1):
                        ok = all(
                            a.shape == b.shape and np.array_equal(a, b)
                            for a, b in zip(preds1, y_true)
                        )
                        ex_correct += 1 if ok else 0
                        ex_eval += 1
                        msg = " (correct)" if ok else " (wrong)"
                    else:
                        msg = " (no GT)"

            print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}{msg}")

        except Exception as e:
            print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")
            traceback.print_exc()

    if args.eval:
        if args.metric == "arc" and arc_eval > 0:
            print(
                f"==> ARC metric (pass@2) accuracy: {arc_correct}/{arc_eval} = {arc_correct/arc_eval:.3f}"
            )
        if args.metric == "exact" and ex_eval > 0:
            print(
                f"==> Exact-match task accuracy: {ex_correct}/{ex_eval} = {ex_correct/ex_eval:.3f}"
            )


if __name__ == "__main__":
    main()
