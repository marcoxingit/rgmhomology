
# run_arc_solver.py
from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
from arc_tda_features import set_matching_metric
import sys, json

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w]")
        return
    path = sys.argv[1]
    match = None
    if len(sys.argv) >= 4 and sys.argv[2] == "--match":
        match = sys.argv[3]
    if match:
        set_matching_metric(match)
    task = load_arc_task_json(path)
    solver = ARCSolver()
    preds = solver.solve_task(task)
    out = {"predictions": [grid_to_list(g) for g in preds]}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
