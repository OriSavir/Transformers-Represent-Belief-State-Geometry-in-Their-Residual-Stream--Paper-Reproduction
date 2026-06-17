from probe import fit_affine_probe, load_model, collect_activations_and_beliefs
from processes import PROCESSES
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob

def collect_emergence_files(spacing=None):
    """Collect checkpoint paths and their corresponding training steps.
    If spacing is defined, takes every Nth checkpoint saved.
    Returns a list of (step, ckpt_path) tuples sorted by step."""
    files = glob.glob("checkpoints/mess3_step*.pt")
    steps_and_paths = []
    for f in files:
        step = int(f.split("_")[-1].split(".")[0].split("step")[-1])
        steps_and_paths.append((step, f))
    steps_and_paths.sort(key=lambda x: x[0])  # sort by step number
    if spacing is not None:
        steps_and_paths = [sp for i, sp in enumerate(steps_and_paths) if i % spacing == 0]
    return steps_and_paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spacing", type=int, default=3, help="Spacing between plotted checkpoints of emergence plot, ie take every Nth checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_seqs", type=int, default=3000)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    files = collect_emergence_files(a.spacing)
    process = PROCESSES["mess3"]
    rng = np.random.default_rng(a.seed)
    # generate random sequences once and reuse for all checkpoints, to reduce noise in the emergence plot
    seqs = rng.integers(0, process.n_tokens, size=(a.n_seqs, a.seq_len))
    print(f"Found {len(files)} checkpoints for emergence plot (spacing={a.spacing}):")
    results = []
    for step, ckpt in files:
        model = load_model(ckpt, process, a.device)
        acts, beliefs = collect_activations_and_beliefs(
            model, process, a.n_seqs, a.seq_len, rng, a.device, seqs=seqs)
        W, pred, mse = fit_affine_probe(acts, beliefs)
        results.append((step, mse))
        print(f"  Step {step}: Probe MSE = {mse:.6f}")

    steps, mses = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mses, marker='o')
    plt.title("Emergence of Belief State Geometry in the Final Residual Stream")
    plt.xlabel("Training Step")
    plt.ylabel("Probe MSE")
    plt.grid()
    plt.savefig("emergence_curve.png")
    print("Saved emergence curve to emergence_curve.png")
    




if __name__ == "__main__":
    main()