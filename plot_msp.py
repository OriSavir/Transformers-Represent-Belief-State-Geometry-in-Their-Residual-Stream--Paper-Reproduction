"""Reproduce the ground-truth belief-state geometries (paper Figs 5B and 7B)."""
import numpy as np
import matplotlib.pyplot as plt
from processes import MESS3, RRXOR
from msp import belief_cloud, distinct_belief_states


def project_simplex_2d(beliefs):
    """3-state beliefs -> 2D triangle coordinates."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return beliefs @ V


def project_pca_2d(beliefs):
    """>3-state beliefs -> 2D via PCA (for visualizing the 4-simplex)."""
    X = beliefs - beliefs.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def plot_mess3(depth=10, out="mess3_msp.png"):
    beliefs, _ = belief_cloud(MESS3, depth)
    xy = project_simplex_2d(beliefs)
    colors = np.clip(beliefs, 0, 1)  # RGB = belief distribution
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=160)
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=1.0, edgecolors="none")
    tri = project_simplex_2d(np.eye(3))
    ax.plot(*np.vstack([tri, tri[0]]).T, color="white", lw=0.5, alpha=0.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Mess3 belief geometry (ground truth, Fig 5B)", color="white", fontsize=11)
    fig.tight_layout(); fig.savefig(out, facecolor="black"); plt.close(fig)
    print(f"saved {out}  ({len(beliefs)} points)")


def plot_rrxor(max_depth=20, out="rrxor_msp.png"):
    states = distinct_belief_states(RRXOR, max_depth)
    xy = project_pca_2d(states)
    colors = plt.cm.hsv(np.linspace(0, 1, len(states)))
    fig, ax = plt.subplots(figsize=(6.0, 5.6), dpi=160)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=60, edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"RRXOR belief geometry: {len(states)} states (Fig 7B)\n"
                 "4-simplex projected to 2D via PCA", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"saved {out}  ({len(states)} distinct states)")


if __name__ == "__main__":
    plot_mess3()
    plot_rrxor()
