"""Multi-seed replication of the belief-transfer experiment.

For each seed: train (A -> fork warm-B / A->A) into its own ckpt dir, then analyze
(writing results.json there). Finally overlay the load-bearing curves across seeds
with mean +/- spread, so we can see whether the single-seed findings are robust.

Reuses train_switch.py and analyze_switch.py unchanged (just different --out_dir /
--ckpt_dir and --seed per run).
"""
import argparse, os, sys, json, subprocess
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def run(seed, ckpt_dir, device, skip_train):
    if not skip_train:
        subprocess.run([sys.executable, os.path.join(HERE, "train_switch.py"),
                        "--seed", str(seed), "--out_dir", ckpt_dir,
                        "--device", device], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "analyze_switch.py"),
                    "--ckpt_dir", ckpt_dir, "--device", device], check=True)
    return json.load(open(os.path.join(ckpt_dir, "results.json")))


def band(ax, x, ys, color, label):
    """Plot mean +/- min/max band across seeds."""
    ys = np.array(ys)                       # (n_seeds, n_steps)
    mean = ys.mean(0)
    ax.plot(x, mean, "o-", color=color, label=label)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip_train", action="store_true",
                   help="reuse existing ckpt dirs, just re-analyze/plot")
    a = p.parse_args()

    results = []
    for s in a.seeds:
        ckpt_dir = os.path.join(HERE, f"ckpts_seed{s}")
        print(f"\n===== SEED {s}  ({ckpt_dir}) =====")
        results.append(run(s, ckpt_dir, a.device, a.skip_train))

    steps = results[0]["steps"]             # same schedule for all seeds
    forget = [r["forget"] for r in results]
    froz   = [r["froz"]   for r in results]
    own2d  = [r["own2d"]  for r in results]
    floors = [r["corr_floor"] for r in results]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # forgetting across seeds
    band(ax[0], steps, forget, "crimson", "probe-A MSE (forgetting)")
    ax[0].axhline(np.mean(floors), ls="--", color="crimson", alpha=.6,
                  label=f"corr floor ~{np.mean(floors):.4f}")
    ax[0].set_xscale("symlog"); ax[0].set_xlabel("fine-tuning step on B")
    ax[0].set_ylabel("probe MSE"); ax[0].set_title(f"Forgetting A ({len(a.seeds)} seeds)")
    ax[0].legend(fontsize=8)

    # the load-bearing panel: B on A's frozen plane vs B's own plane, across seeds
    band(ax[1], steps, froz,  "purple",   "B on A's FROZEN 2-dim plane")
    band(ax[1], steps, own2d, "seagreen", "B on warm-B's OWN 2-dim plane")
    ax[1].set_xscale("symlog"); ax[1].set_xlabel("fine-tuning step on B")
    ax[1].set_ylabel("B-belief readout MSE (2-dim)")
    ax[1].set_title(f"Does B live on A's real estate? ({len(a.seeds)} seeds)")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "replication.png"), dpi=130)
    print("\nsaved replication.png")

    # numeric summary of the key claim at the last step
    froz_last = np.array([r["froz"][-1]  for r in results])
    own_last  = np.array([r["own2d"][-1] for r in results])
    print(f"\nAt final step, across {len(a.seeds)} seeds:")
    print(f"  B on A's frozen plane : {froz_last.mean():.5f} +/- {froz_last.std():.5f}")
    print(f"  B on own plane        : {own_last.mean():.5f} +/- {own_last.std():.5f}")
    ratio = froz_last / own_last
    print(f"  ratio (frozen/own)    : {ratio.mean():.1f}x +/- {ratio.std():.1f}"
          f"   (>>1 => B moved off A's plane in every seed)")


if __name__ == "__main__":
    main()