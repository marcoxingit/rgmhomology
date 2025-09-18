
# run_arc_solver.py
from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
import sys, json

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_arc_solver.py /path/to/task.json")
        return
    path = sys.argv[1]
    task = load_arc_task_json(path)
    solver = ARCSolver()
    preds = solver.solve_task(task)
    out = {"predictions": [grid_to_list(g) for g in preds]}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
