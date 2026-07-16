"""STEP 3 -- Read the belief-transfer results out of the checkpoints.

CONTENTS (Q1): forgetting curve (probe-A MSE) + acquisition curve (probe-B MSE),
               with correlation-floor and cold-B reference lines.
LOCATION (Q2): drift comparison (warm-B vs A->A plane angle) and frozen-plane
               readout (can B be read on A's frozen plane?).

NOTE: collect_activations_and_beliefs returns one row per (sequence, position),
with beliefs already aligned to the activation rows. We use those returned
beliefs -- never recompute per-sequence -- and get A-labels vs B-labels by
calling the collector once per process on the SAME eval sequences.
"""
import argparse, os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from processes import mess3
from model import build_model
from probe import fit_affine_probe, collect_activations_and_beliefs
from subspace import plane_basis, plane_distance_deg


def load_model(path, process, seq_len, device):
    import torch
    m = build_model(process, n_ctx=seq_len, seed=0, device=device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def labels_for(model, process, eval_seqs, device):
    """Return beliefs under `process` aligned to this model's activation rows."""
    acts, beliefs = collect_activations_and_beliefs(
        model, process, n_seqs=len(eval_seqs), seq_len=eval_seqs.shape[1],
        rng=None, device=device, seqs=eval_seqs)
    return np.asarray(acts), np.asarray(beliefs)


def readout_on_plane(acts, target_beliefs, Q):
    """MSE of predicting target_beliefs from acts projected onto the 2-dim plane Q.
    Both curves in panel 3 use THIS (2-dim), so the comparison is apples-to-apples:
    the only difference is WHICH plane (A's frozen vs warm-B's own)."""
    proj = acts @ Q                              # (N, 2) -- only 2 directions
    X = np.hstack([proj, np.ones((len(proj), 1))])
    W, *_ = np.linalg.lstsq(X, target_beliefs, rcond=None)
    return float(np.mean((X @ W - target_beliefs) ** 2))


def step_of(path):
    return int(os.path.basename(path).split("step")[1].split(".")[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--n_eval", type=int, default=4000)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.ckpt_dir, "meta.json")))
    A = mess3(meta["a_A"], meta["x_A"], name="A")
    B = mess3(meta["a_B"], meta["x_B"], name="B")
    seq_len = meta["seq_len"]
    device = a.device
    d_model = 64

    ev = np.random.default_rng(a.eval_seed).integers(0, 3, size=(a.n_eval, seq_len))

    # --- A's converged plane (frozen reference) + its aligned A/B label sets ---
    mA = load_model(os.path.join(a.ckpt_dir, "A_converged.pt"), A, seq_len, device)
    actsA, bA_on_A = labels_for(mA, A, ev, device)   # A-model acts, A-beliefs
    _,     bB_on_A = labels_for(mA, B, ev, device)   # same acts rows, B-beliefs
    W_Aconv, _, mseA_conv = fit_affine_probe(actsA, bA_on_A)
    Q_A, _ = plane_basis(W_Aconv, d_model)           # A's FROZEN 2 directions

    # correlation floor: B-beliefs -> A-beliefs (both aligned to the same rows)
    Xf = np.hstack([bB_on_A, np.ones((len(bB_on_A), 1))])
    Wf, *_ = np.linalg.lstsq(Xf, bA_on_A, rcond=None)
    corr_floor = float(np.mean((Xf @ Wf - bA_on_A) ** 2))

    # Ground-truth beliefs depend ONLY on (process, eval_seqs) -- not on any model.
    # bA_on_A / bB_on_A above were computed on mA, but they're the SAME for every
    # checkpoint (same ev, same processes), so reuse them as the fixed label sets.
    bA = bA_on_A   # A-beliefs, aligned to the (seq,pos) row order of ev
    bB = bB_on_A   # B-beliefs, same row order
    assert len(bA) == len(actsA) == len(bB), "belief/activation rows misaligned"

    # cold-B "fully learned B" target
    coldB_mse = None
    cpath = os.path.join(a.ckpt_dir, "coldB.pt")
    if os.path.exists(cpath):
        mC = load_model(cpath, B, seq_len, device)
        actsC, bB_on_C = labels_for(mC, B, ev, device)
        _, _, coldB_mse = fit_affine_probe(actsC, bB_on_C)

    # --- sweep warm-B and A->A checkpoints ---
    warm = sorted(glob.glob(os.path.join(a.ckpt_dir, "warmB_step*.pt")), key=step_of)
    toA  = sorted(glob.glob(os.path.join(a.ckpt_dir, "toA_step*.pt")),   key=step_of)

    rows = []
    for path in warm:
        s = step_of(path)
        m = load_model(path, B, seq_len, device)
        acts, _ = labels_for(m, B, ev, device)       # run model ONCE for activations
        # bA, bB are fixed ground truth (precomputed above) -- model-independent
        _, _, mse_B = fit_affine_probe(acts, bB)     # acquisition (full 64-dim)
        _, _, mse_A = fit_affine_probe(acts, bA)     # forgetting  (full 64-dim)
        W_self, _, _ = fit_affine_probe(acts, bB)    # warm-B's own probe
        Q_self, _ = plane_basis(W_self, d_model)     # warm-B's own 2-dim plane
        drift = plane_distance_deg(W_Aconv, W_self, d_model)
        b_on_Aplane   = readout_on_plane(acts, bB, Q_A)     # B on A's FROZEN 2 dims
        b_on_ownplane = readout_on_plane(acts, bB, Q_self)  # B on its OWN 2 dims
        rows.append((s, mse_A, mse_B, drift, b_on_Aplane, b_on_ownplane))
        print(f"warmB step {s:>4}: forget(A) {mse_A:.5f}  acquire(B) {mse_B:.5f}"
              f"  drift {drift:4.1f}deg  B-on-Aplane {b_on_Aplane:.5f}"
              f"  B-on-ownplane {b_on_ownplane:.5f}")

    drift_null = []
    for path in toA:
        s = step_of(path)
        m = load_model(path, A, seq_len, device)
        acts, _ = labels_for(m, A, ev, device)
        W_self, _, _ = fit_affine_probe(acts, bA)
        drift_null.append((s, plane_distance_deg(W_Aconv, W_self, d_model)))

    steps    = [r[0] for r in rows]
    forget   = [r[1] for r in rows]
    acquire  = [r[2] for r in rows]
    drift    = [r[3] for r in rows]
    froz     = [r[4] for r in rows]   # B on A's frozen 2-dim plane
    own2d    = [r[5] for r in rows]   # B on warm-B's own 2-dim plane
    nsteps   = [d[0] for d in drift_null]
    ndrift   = [d[1] for d in drift_null]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(steps, forget, "o-", label="probe-A MSE (forgetting)", color="crimson")
    ax[0].plot(steps, acquire, "o-", label="probe-B MSE (acquisition)", color="steelblue")
    ax[0].axhline(corr_floor, ls="--", color="crimson", alpha=.6,
                  label=f"corr floor (A forgotten) {corr_floor:.4f}")
    if coldB_mse:
        ax[0].axhline(coldB_mse, ls="--", color="steelblue", alpha=.6,
                      label=f"cold-B target {coldB_mse:.4f}")
    ax[0].set_xscale("symlog"); ax[0].set_xlabel("fine-tuning step on B")
    ax[0].set_ylabel("probe MSE"); ax[0].set_title("Contents: forgetting A / acquiring B")
    ax[0].legend(fontsize=8)

    ax[1].plot(steps, drift, "o-", label="angle(A, warm-B)", color="darkorange")
    ax[1].plot(nsteps, ndrift, "s--", label="angle(A, A->A) [drift null]", color="gray")
    ax[1].set_xscale("symlog"); ax[1].set_xlabel("step")
    ax[1].set_ylabel("plane angle (deg)")
    ax[1].set_title("Location: did the plane move more than drift?")
    ax[1].legend(fontsize=8)

    ax[2].plot(steps, froz,  "o-", label="B on A's FROZEN 2-dim plane", color="purple")
    ax[2].plot(steps, own2d, "o-", label="B on warm-B's OWN 2-dim plane", color="seagreen")
    ax[2].set_xscale("symlog"); ax[2].set_xlabel("fine-tuning step on B")
    ax[2].set_ylabel("B-belief readout MSE (2-dim)")
    ax[2].set_title("Location (fair 2-vs-2): does B live on A's real estate?")
    ax[2].legend(fontsize=8)

    plt.tight_layout()
    fig_name = os.path.join(a.ckpt_dir, "switch_results.png")
    plt.savefig(fig_name, dpi=130); plt.close(fig)

    # dump the numbers so a multi-seed pass can aggregate them
    out = dict(steps=steps, forget=forget, acquire=acquire, drift=drift,
               froz=froz, own2d=own2d, nsteps=nsteps, ndrift=ndrift,
               corr_floor=corr_floor, coldB_mse=coldB_mse,
               mseA_conv=mseA_conv)
    with open(os.path.join(a.ckpt_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\ncorr floor {corr_floor:.5f} | A_conv self-probe {mseA_conv:.5f}"
          f" | cold-B {coldB_mse}")
    print(f"saved {fig_name} and results.json")


if __name__ == "__main__":
    main()