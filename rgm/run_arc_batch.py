
# run_arc_batch.py
import argparse, os, json, random, glob, traceback
import numpy as np
from arc_solver import ARCSolver, grid_to_list, load_arc_task_json, grid_from_list
from arc_tda_features import set_matching_metric

def iter_tasks(task_dir: str):
    return sorted(glob.glob(os.path.join(task_dir, "*.json")))

def task_exact_match(task, preds):
    # Return True (correct), False (wrong), or None (cannot evaluate: no ground truth in 'test')
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="Directory with ARC task JSON files")
    ap.add_argument("--out", required=True, help="Output directory to write predictions")
    ap.add_argument("--max", type=int, default=0, help="Max number of tasks (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle task order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--match", type=str, default="custom", choices=["custom","feat-l2","gtda-w"], help="matching distance")
    ap.add_argument("--eval", action="store_true", help="Compute exact-match accuracy if ground truth is present")
    args = ap.parse_args()

    set_matching_metric(args.match)

    paths = iter_tasks(args.tasks)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    if args.max and args.max < len(paths):
        paths = paths[:args.max]

    os.makedirs(args.out, exist_ok=True)
    solver = ARCSolver()
    n_done = 0
    n_eval = 0
    n_correct = 0
    for i, p in enumerate(paths, 1):
        try:
            task = load_arc_task_json(p)
            preds = solver.solve_task(task)
            out_path = os.path.join(args.out, os.path.basename(p))
            with open(out_path, "w") as f:
                json.dump({"predictions": [grid_to_list(g) for g in preds]}, f)
            if args.eval:
                res = task_exact_match(task, preds)
                if res is True:
                    n_correct += 1; n_eval += 1
                    print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}  (correct)")
                elif res is False:
                    n_eval += 1
                    print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}  (wrong)")
                else:
                    print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}  (no GT)")
            else:
                print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}")
            n_done += 1
        except Exception as e:
            print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")
            if args.debug:
                traceback.print_exc()
    if args.eval and n_eval > 0:
        acc = n_correct / n_eval
        print(f"==> Exact-match accuracy: {n_correct}/{n_eval} = {acc:.3f}")
    elif args.eval:
        print("==> No ground truth outputs found in this split; cannot compute accuracy.")

if __name__ == "__main__":
    main()
