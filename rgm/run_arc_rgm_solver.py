# run_arc_rgm_solver.py
import sys, json, os
import numpy as np
from arc_rgm_solver import ARCRGMSolver, grid_to_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_arc_rgm_solver.py /path/to/task.json [k_bins]")
        return
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    with open(path, "r") as f:
        task = json.load(f)
    solver = ARCRGMSolver(n_bins=k)
    preds = solver.solve_task(task)
    print(json.dumps({"predictions": [grid_to_list(g) for g in preds]}))


if __name__ == "__main__":
    main()
