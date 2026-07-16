"""Linear probe: recover belief-state geometry from the residual stream (Fig 5C).

Loads a trained checkpoint, runs sequences through it while caching the final-block
residual stream, then fits an affine least-squares map from those 64-dim activations
to the 3-dim ground-truth belief states. Projecting the fit to the 2-simplex should
reproduce the Mess3 fractal. A label-shuffle control is included as a sanity baseline.
"""
import argparse
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, HookedTransformerConfig

from processes import PROCESSES
from msp import belief_update
from model import build_model, final_resid_hook


def fit_affine_probe(acts, beliefs):
    """Least-squares affine map  beliefs ~= [acts | 1] @ W.  Returns W, predictions, MSE."""
    X = np.hstack([acts, np.ones((len(acts), 1))])
    W, *_ = np.linalg.lstsq(X, beliefs, rcond=None)
    pred = X @ W
    mse = float(np.mean((pred - beliefs) ** 2))
    return W, pred, mse


def shuffle_control_mse(acts, beliefs, seed=0):
    """Refit with belief labels shuffled — destroys any real correspondence."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(beliefs))
    _, _, mse = fit_affine_probe(acts, beliefs[perm])
    return mse


def project_simplex_2d(b):
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return b @ V


# ---- model-dependent stuff ---------------------------
def load_model(ckpt_path, process=None, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    if isinstance(cfg, dict):
        cfg = HookedTransformerConfig.from_dict(cfg)
    cfg.device = device 
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def collect_activations_and_beliefs(model, process, seq_len, device, seqs=None):
    """Return (acts[N,64], beliefs[N,n_states]) over every position of every sequence.

    Alignment: the residual at position t has (causally) seen tokens[0..t], so its
    label is the belief state after observing tokens[0..t].
    """
    hook = final_resid_hook(model)
    if seqs is None:
        seqs = [list(s) for s in itertools.product(process.tokens, repeat=seq_len)]
    acts, beliefs = [], []
    bs = 256
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        batch = torch.tensor(chunk, dtype=torch.long, device=device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=hook)
        a = cache[hook].cpu().numpy()  # (B, T, d_model)
        for bi, seq in enumerate(chunk):
            eta = process.stationary.copy()
            for t in range(seq_len):
                eta, _ = belief_update(eta, process.T[seq[t]])
                acts.append(a[bi, t])
                beliefs.append(eta.copy())
    return np.array(acts), np.array(beliefs)


def plot_recovered(pred, beliefs, out="recovered_mess3.png"):
    xy = project_simplex_2d(pred)
    colors = np.clip(beliefs, 0, 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=160)
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.3, alpha=0.3, edgecolors="none")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Recovered from residual stream (Fig 5C)", color="white", fontsize=11)
    fig.tight_layout(); fig.savefig(out, facecolor="black"); plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--process", default="mess3", choices=list(PROCESSES))
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    process = PROCESSES[a.process]
    rng = np.random.default_rng(a.seed)
    model = load_model(a.ckpt, process, a.device)
    acts, beliefs = collect_activations_and_beliefs(
        model, process, a.seq_len, a.device)

    W, pred, mse = fit_affine_probe(acts, beliefs)
    ctrl = shuffle_control_mse(acts, beliefs)
    print(f"probe MSE        : {mse:.6f}")
    print(f"shuffle-ctrl MSE : {ctrl:.6f}   (should be much larger)")
    print(f"ratio            : {ctrl / mse:.1f}x")
    plot_recovered(pred, beliefs)


if __name__ == "__main__":
    main()