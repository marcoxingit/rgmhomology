# # run_arc_batch.py
# # Run ARCSolver over a directory of ARC JSON tasks.
# # Example:
# #   python run_arc_batch.py --tasks ./arc_data/v1/training --out ./preds_v1_train --max 50 --shuffle
# import argparse, os, json, random, glob
# from typing import List
# from arc_solver import ARCSolver, grid_to_list, load_arc_task_json

# def iter_tasks(task_dir: str) -> List[str]:
#     return sorted(glob.glob(os.path.join(task_dir, "*.json")))

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--tasks", required=True, help="Directory with ARC task JSON files")
#     ap.add_argument("--out", required=True, help="Output directory to write predictions")
#     ap.add_argument("--max", type=int, default=0, help="Max number of tasks (0 = all)")
#     ap.add_argument("--shuffle", action="store_true", help="Shuffle task order")
#     ap.add_argument("--seed", type=int, default=0)
#     args = ap.parse_args()

#     paths = iter_tasks(args.tasks)
#     if args.shuffle:
#         random.seed(args.seed)
#         random.shuffle(paths)
#     if args.max and args.max < len(paths):
#         paths = paths[:args.max]

#     os.makedirs(args.out, exist_ok=True)
#     solver = ARCSolver()
#     for i, p in enumerate(paths, 1):
#         try:
#             task = load_arc_task_json(p)
#             preds = solver.solve_task(task)
#             out_path = os.path.join(args.out, os.path.basename(p))
#             with open(out_path, "w") as f:
#                 json.dump({"predictions": [grid_to_list(g) for g in preds]}, f)
#             print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}")
#         except Exception as e:
#             print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")

# if __name__ == "__main__":
#     main()
import argparse, os, json, random, glob, traceback
import numpy as np
from arc_solver import ARCSolver, grid_to_list, load_arc_task_json, grid_from_list
from arc_tda_features import (
    set_matching_metric,
)  # optional; only matters if you use TDA distances


def iter_tasks(task_dir: str):
    return sorted(glob.glob(os.path.join(task_dir, "*.json")))


# ---------- second-attempt generators ----------
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


# ---------- metrics ----------
def task_exact_match(task, preds):
    y_true = []
    for t in task.get("test", []):
        if "output" not in t:
            return None
        y_true.append(grid_from_list(t["output"]))
    if len(y_true) != len(preds):
        return False
    for a, b in zip(preds, y_true):
        if a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


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
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--match", type=str, default="custom", choices=["custom", "feat-l2", "gtda-w"]
    )
    ap.add_argument(
        "--mode", type=str, default="rule", choices=["rule", "tda"], help="solver mode"
    )
    ap.add_argument("--metric", type=str, default="exact", choices=["exact", "arc"])
    ap.add_argument(
        "--attempt2",
        type=str,
        default="duplicate",
        choices=["duplicate", "flip_h", "flip_v", "transpose", "rotate90", "rotate270"],
    )
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--kaggle", action="store_true")
    args = ap.parse_args()

    set_matching_metric(args.match)  # harmless if you stay on rule mode

    paths = iter_tasks(args.tasks)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    if args.max and args.max < len(paths):
        paths = paths[: args.max]

    os.makedirs(args.out, exist_ok=True)
    solver = ARCSolver(mode=args.mode)
    ex_eval = ex_correct = 0
    arc_eval = arc_correct = 0
    kaggle_dict = {}

    for i, p in enumerate(paths, 1):
        tid = os.path.splitext(os.path.basename(p))[0]
        try:
            task = load_arc_task_json(p)
            preds1 = solver.solve_task(task)
            with open(os.path.join(args.out, os.path.basename(p)), "w") as f:
                json.dump({"predictions": [grid_to_list(g) for g in preds1]}, f)
            msg_suffix = ""
            if args.metric == "exact" and args.eval:
                r = task_exact_match(task, preds1)
                if r is True:
                    ex_correct += 1
                    ex_eval += 1
                    msg_suffix = " (correct)"
                elif r is False:
                    ex_eval += 1
                    msg_suffix = " (wrong)"
                else:
                    msg_suffix = " (no GT)"
            elif args.metric == "arc":
                preds2 = [make_second_attempt(g, mode=args.attempt2) for g in preds1]
                if args.kaggle:
                    kaggle_dict[tid] = [
                        {"attempt_1": grid_to_list(a1), "attempt_2": grid_to_list(a2)}
                        for a1, a2 in zip(preds1, preds2)
                    ]
                if args.eval:
                    good, tot = task_pass_at_2(task, preds1, preds2)
                    if tot > 0:
                        arc_correct += good
                        arc_eval += tot
                        msg_suffix = f" (pass@2 {good}/{tot})"
                    else:
                        msg_suffix = " (no GT)"
            print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}{msg_suffix}")
        except Exception as e:
            print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")
            if args.debug:
                traceback.print_exc()

    if args.metric == "exact" and args.eval:
        if ex_eval > 0:
            acc = ex_correct / ex_eval
            print(f"==> Exact-match task accuracy: {ex_correct}/{ex_eval} = {acc:.3f}")
        else:
            print(
                "==> No ground truth on this split; cannot compute exact-match accuracy."
            )
    if args.metric == "arc" and args.eval:
        if arc_eval > 0:
            acc = arc_correct / arc_eval
            print(
                f"==> ARC metric (pass@2) accuracy: {arc_correct}/{arc_eval} = {acc:.3f}"
            )
        else:
            print("==> No ground truth on this split; cannot compute ARC metric.")

    if args.metric == "arc" and args.kaggle:
        sub_path = os.path.join(args.out, "submission.json")
        with open(sub_path, "w") as f:
            json.dump(kaggle_dict, f)
        print(f"Wrote Kaggle submission JSON -> {sub_path}")


if __name__ == "__main__":
    main()
