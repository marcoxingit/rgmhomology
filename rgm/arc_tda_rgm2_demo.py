# arc_tda_rgm2_demo.py (v2)
# Use giotto-tda tokens as observations for spm_mb_structure_learning with a locations matrix
import numpy as np
from .arc_tda_adapter import tda_sequence_to_onehots
from .fast_structure_learning import spm_mb_structure_learning

# 1) Build a toy ARC sequence
H = W = 15
g0 = np.zeros((H, W), int)
g0[2:5, 2:5] = 3
g0[8:12, 4:7] = 5
g1 = np.zeros((H, W), int)
g1[3:6, 7:10] = 3
g1[9:13, 10:13] = 5
g2 = np.zeros((H, W), int)
g2[4:7, 9:12] = 3
g2[10:14, 11:14] = 5
grids = [g0, g1, g2]

# 2) Convert to (modalities x time x bins) one-hots via TDA
one_hots, meta = tda_sequence_to_onehots(grids, n_bins=7)
n_modalities, T, total_bins = one_hots.shape

# 3) Build a locations_matrix from TDA centroids (mean over time where present)
locs = []
for tr in meta["tracks"]:
    pts = [c for c in tr.centroids if c is not None]
    if len(pts) == 0:
        locs.append([W / 2, H / 2])  # fallback center
    else:
        rc = np.mean(np.array(pts), axis=0)  # (row, col)
        locs.append([rc[1], rc[0]])  # (x, y) = (col, row)
locations_matrix = np.asarray(locs, dtype=float)

# 4) Run structure learning with locations (so RG groups match our #modalities)
agents, RG, _ = spm_mb_structure_learning(
    one_hots,
    locations_matrix=locations_matrix,
    size=(W, H, n_modalities),
    dx=2,
    num_controls=0,
    max_levels=4,
    agents=None,
    RG=None,
)

print("OK ✓  agents:", len(agents), "levels:", len(RG))
for lvl, groups in enumerate(RG):
    print(" level", lvl, "groups:", [list(map(int, g)) for g in groups])
# after spm_mb_structure_learning(...)
agent = agents[0]  # top (and only) level agent
print("Modalities:", len(agent.A))
print(
    "A shapes (per modality):", [a.shape for a in agent.A]
)  # (batch=1, n_obs, n_states)
print("B shape (group 0):", agent.B[0].shape)  # transition(s)

# A quick sense-check: match each token to its most probable hidden state
import jax.numpy as jnp

A0 = agent.A[0][0]  # remove batch dim -> (n_obs, n_states)
print("Top obs->state map (mod 0):", jnp.argmax(A0, axis=1))
