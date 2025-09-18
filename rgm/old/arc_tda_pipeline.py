
# arc_tda_pipeline.py
# Example: augment a simple ARC graph model (objects as nodes, adjacency as edges)
# with TDA features and perform cross-grid matching.
#
# This is intentionally lightweight so you can adapt it into your own ARGM/RGM code.
#
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

from arc_tda_features import (
    ARCObject, grid_to_objects_with_tda, wasserstein_matching
)

@dataclass
class GraphNode:
    idx: int
    color: int
    centroid_rc: Tuple[float, float]
    tda: np.ndarray             # vectorized features
    raw_diagram: np.ndarray

def build_graph_from_grid(grid: np.ndarray) -> Dict[str, Any]:
    """Build a toy graph: nodes=objects, edges if touching (4-neighborhood)."""
    objs = grid_to_objects_with_tda(grid)
    nodes: List[GraphNode] = [
        GraphNode(i, o.color, o.centroid_rc, o.features, o.diagram) for i, o in enumerate(objs)
    ]
    # Simple edge rule: centroids within L1 <= 2 are neighbors (you can change this)
    edges = []
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if j <= i: continue
            if abs(ni.centroid_rc[0] - nj.centroid_rc[0]) + abs(ni.centroid_rc[1] - nj.centroid_rc[1]) <= 2:
                edges.append((i, j))
    return {"nodes": nodes, "edges": edges}

def match_graphs_by_tda(grid_A: np.ndarray, grid_B: np.ndarray):
    objs_A = grid_to_objects_with_tda(grid_A)
    objs_B = grid_to_objects_with_tda(grid_B)
    matches, D = wasserstein_matching(objs_A, objs_B, return_matrix=True)
    return matches, D

if __name__ == "__main__":
    # Minimal smoke test
    H = W = 20
    grid_A = np.zeros((H, W), dtype=int)
    grid_A[2:6, 2:6] = 3
    grid_A[10:15, 4:9] = 5
    grid_B = np.zeros_like(grid_A)
    grid_B[4:8, 7:11] = 3
    grid_B[12:17, 10:15] = 5
    matches, D = match_graphs_by_tda(grid_A, grid_B)
    print("Matches:", matches)
    print(np.round(D, 3))
