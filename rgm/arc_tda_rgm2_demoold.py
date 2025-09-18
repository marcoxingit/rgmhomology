# arc_tda_rgm2_demo.py
# Example: use giotto-tda tokens as observations for rgm2.RGM
import numpy as np

from .arc_tda_adapter import tda_sequence_to_onehots
from .rgm2 import RGM

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
one_hots, meta = tda_sequence_to_onehots(grids, n_bins=6)

# 3) Fit a tiny RGM on these discrete observations
model = RGM(
    n_bins=one_hots.shape[-1],
    n_modalities=one_hots.shape[0],
    width=W,
    height=H,
    max_levels=4,
    dx=2,
)
model.fit(one_hots)

# 4) Reconstruct last frame tokens from the model (toy call if available)
states = model.infer_states(one_hots)
print("Latent states shape:", [s.shape for s in states])
