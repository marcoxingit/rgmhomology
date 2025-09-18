# # # run_arc_solver.py
# # from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
# # from arc_tda_features import set_matching_metric
# # import sys, json

# # def main():
# #     if len(sys.argv) < 2:
# #         print("Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w]")
# #         return
# #     path = sys.argv[1]
# #     match = None
# #     if len(sys.argv) >= 4 and sys.argv[2] == "--match":
# #         match = sys.argv[3]
# #     if match:
# #         set_matching_metric(match)
# #     task = load_arc_task_json(path)
# #     solver = ARCSolver()
# #     preds = solver.solve_task(task)
# #     out = {"predictions": [grid_to_list(g) for g in preds]}
# #     print(json.dumps(out))

# # if __name__ == "__main__":
# #     main()
# # run_arc_solver.py
# # (single-task runner; --match, --attempt2, --kaggle for Kaggle JSON)
# from arc_solver import ARCSolver, load_arc_task_json, grid_to_list
# from arc_tda_features import set_matching_metric
# import sys, json, os
# import numpy as np


# def make_second_attempt(pred: np.ndarray, mode: str = "duplicate") -> np.ndarray:
#     mode = (mode or "duplicate").lower()
#     if mode == "duplicate":
#         return pred.copy()
#     if mode == "flip_h":
#         return np.asanyarray(pred)[:, ::-1]
#     if mode == "flip_v":
#         return np.asanyarray(pred)[::-1, :]
#     if mode == "transpose":
#         return np.asanyarray(pred).T
#     if mode == "rotate90":
#         return np.rot90(np.asanyarray(pred), 1)
#     if mode == "rotate270":
#         return np.rot90(np.asanyarray(pred), 3)
#     return pred.copy()


# def main():
#     if len(sys.argv) < 2:
#         print(
#             "Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w] [--attempt2 MODE] [--kaggle]"
#         )
#         return
#     path = sys.argv[1]
#     # parse simple flags
#     match = None
#     attempt2 = "duplicate"
#     kaggle = False
#     if "--match" in sys.argv:
#         i = sys.argv.index("--match")
#         if i + 1 < len(sys.argv):
#             match = sys.argv[i + 1]
#     if "--attempt2" in sys.argv:
#         i = sys.argv.index("--attempt2")
#         if i + 1 < len(sys.argv):
#             attempt2 = sys.argv[i + 1]
#     if "--kaggle" in sys.argv:
#         kaggle = True
#     if match:
#         set_matching_metric(match)
#     task = load_arc_task_json(path)
#     solver = ARCSolver()
#     preds1 = solver.solve_task(task)
#     if kaggle:
#         # emit two attempts per test
#         preds2 = [make_second_attempt(g, mode=attempt2) for g in preds1]
#         tid = os.path.splitext(os.path.basename(path))[0]
#         out = {
#             tid: [
#                 {"attempt_1": grid_to_list(a1), "attempt_2": grid_to_list(a2)}
#                 for a1, a2 in zip(preds1, preds2)
#             ]
#         }
#         print(json.dumps(out))
#     else:
#         out = {"predictions": [grid_to_list(g) for g in preds1]}
#         print(json.dumps(out))


# if __name__ == "__main__":
#     main()

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
            "Usage: python run_arc_solver.py /path/to/task.json [--match custom|feat-l2|gtda-w] [--attempt2 MODE] [--kaggle]"
        )
        return
    path = sys.argv[1]
    match = None
    attempt2 = "duplicate"
    kaggle = False
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
    solver = ARCSolver()
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
