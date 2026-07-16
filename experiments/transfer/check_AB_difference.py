"""Is B different enough from A for the experiment to be well-posed?

Pure ground-truth checks (no model, no training):
  1. Renders A's and B's belief-geometry clouds side by side.
  2. Computes the CORRELATION FLOOR: how well B-beliefs linearly predict
     A-beliefs on the same sequences. This is the reference line for the
     forgetting curve -- and if it's high, A and B are too correlated.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from processes import mess3
from msp import belief_for_sequence

A = mess3(0.85, 0.05, name="A")
B = mess3(0.5, 0.2, name="B")

# shared sequence set
rng = np.random.default_rng(0)
seqs = rng.integers(0, 3, size=(6000, 10))

bA = np.array([belief_for_sequence(A, s) for s in seqs])   # (N, 3)
bB = np.array([belief_for_sequence(B, s) for s in seqs])   # (N, 3)

# --- correlation floor: least-squares B-beliefs -> A-beliefs ---
X = np.hstack([bB, np.ones((len(bB), 1))])                 # affine
Wba, *_ = np.linalg.lstsq(X, bA, rcond=None)
pred = X @ Wba
floor_mse = np.mean((pred - bA) ** 2)
# baseline: predicting A with its own mean (no info)
null_mse = np.mean((bA - bA.mean(0)) ** 2)
print(f"correlation floor  MSE(B-beliefs -> A-beliefs) = {floor_mse:.5f}")
print(f"null (predict mean of A)                       = {null_mse:.5f}")
print(f"floor as fraction of null                      = {floor_mse/null_mse:.2f}")
print("  low fraction -> A,B nearly independent (clean forgetting signal)")
print("  high fraction -> A,B correlated (forgetting harder to read)")

# --- side-by-side geometry ---
def to2d(b):  # project 3-simplex to 2D
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    return b @ v

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for k, (b, name) in enumerate([(bA, A.name), (bB, B.name)]):
    xy = to2d(b)
    ax[k].scatter(xy[:, 0], xy[:, 1], s=2, c=b, alpha=0.5)  # color = belief as RGB
    ax[k].set_title(f"{name} belief geometry"); ax[k].axis("equal"); ax[k].axis("off")
plt.tight_layout()
plt.savefig("AB_geometry.png", dpi=130)
print("\nsaved AB_geometry.png")