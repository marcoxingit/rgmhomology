
# run_arc_batch.py
import argparse, os, json, random, glob, traceback
from arc_solver import ARCSolver, grid_to_list, load_arc_task_json

def iter_tasks(task_dir: str):
    return sorted(glob.glob(os.path.join(task_dir, "*.json")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="Directory with ARC task JSON files")
    ap.add_argument("--out", required=True, help="Output directory to write predictions")
    ap.add_argument("--max", type=int, default=0, help="Max number of tasks (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle task order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paths = iter_tasks(args.tasks)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    if args.max and args.max < len(paths):
        paths = paths[:args.max]

    os.makedirs(args.out, exist_ok=True)
    solver = ARCSolver()
    for i, p in enumerate(paths, 1):
        try:
            task = load_arc_task_json(p)
            preds = solver.solve_task(task)
            out_path = os.path.join(args.out, os.path.basename(p))
            with open(out_path, "w") as f:
                json.dump({"predictions": [grid_to_list(g) for g in preds]}, f)
            print(f"[{i}/{len(paths)}] ✓ {os.path.basename(p)}")
        except Exception as e:
            print(f"[{i}/{len(paths)}] ✗ {os.path.basename(p)} :: {e}")
            if args.debug:
                traceback.print_exc()

if __name__ == "__main__":
    main()
