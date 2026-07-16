#To check feasibility for the belief-transfer experiment.
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from processes import mess3
from model import build_model
from probe import fit_affine_probe, shuffle_control_mse, collect_activations_and_beliefs
from subspace import plane_distance_deg, plane_basis, principal_angles_deg


def train_from_scratch(process, steps, batch_size, seq_len, lr, seed, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(process, n_ctx=seq_len, seed=seed, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        seqs = [process.sample(seq_len, rng) for _ in range(batch_size)]
        batch = torch.tensor(seqs, dtype=torch.long, device=device)
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, process.n_tokens),
            batch[:, 1:].reshape(-1),
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"    [{process.name}] step {step:>4}  loss {loss.item():.4f}"
                  f"  ({step/(time.time()-t0):.0f} it/s)")
    model.eval()
    return model


def probe_plane(model, process, eval_seqs, device):
    acts, beliefs = collect_activations_and_beliefs(
        model, process, n_seqs=len(eval_seqs), seq_len=eval_seqs.shape[1],
        rng=None, device=device, seqs=eval_seqs)
    W, _, mse = fit_affine_probe(acts, beliefs)
    ctrl = shuffle_control_mse(acts, beliefs)
    return W, mse, ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a_A", type=float, default=0.85)
    p.add_argument("--x_A", type=float, default=0.05)
    p.add_argument("--a_B", type=float, default=0.5)
    p.add_argument("--x_B", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_eval", type=int, default=4000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed_A", type=int, default=0)
    p.add_argument("--seed_B", type=int, default=1)
    p.add_argument("--eval_seed", type=int, default=1234)
    a = p.parse_args()

    A = mess3(a.a_A, a.x_A, name=f"A(a={a.a_A},x={a.x_A})")
    B = mess3(a.a_B, a.x_B, name=f"B(a={a.a_B},x={a.x_B})")
    print(f"A = {A.name}   B = {B.name}\n")

    ev_rng = np.random.default_rng(a.eval_seed)
    eval_seqs = ev_rng.integers(0, 3, size=(a.n_eval, a.seq_len))

    print("Training A-only model ...")
    model_A = train_from_scratch(A, a.steps, a.batch_size, a.seq_len, a.lr, a.seed_A, a.device)
    print("Training cold-B model ...")
    model_B = train_from_scratch(B, a.steps, a.batch_size, a.seq_len, a.lr, a.seed_B, a.device)

    os.makedirs("ckpts", exist_ok=True)
    torch.save(model_B.state_dict(), os.path.join("ckpts", "coldB.pt"))
    print("saved ckpts/coldB.pt")

    d_model = model_A.cfg.d_model
    W_A, mse_A, ctrl_A = probe_plane(model_A, A, eval_seqs, a.device)
    W_B, mse_B, ctrl_B = probe_plane(model_B, B, eval_seqs, a.device)

    angle = plane_distance_deg(W_A, W_B, d_model)
    Qa, _ = plane_basis(W_A, d_model); Qb, _ = plane_basis(W_B, d_model)
    all_angles = principal_angles_deg(Qa, Qb)

    print("\n" + "=" * 56)
    print("PROBE QUALITY (did each model learn its geometry?)")
    print(f"  A-probe MSE {mse_A:.5f}  | shuffle {ctrl_A:.5f}  | ratio {ctrl_A/mse_A:.1f}x")
    print(f"  B-probe MSE {mse_B:.5f}  | shuffle {ctrl_B:.5f}  | ratio {ctrl_B/mse_B:.1f}x")
    print("-" * 56)
    print("FEASIBILITY: angle between A-plane and cold-B-plane")
    print(f"  principal angles (deg): {np.round(all_angles, 1)}")
    print(f"  headline (max angle)  : {angle:.1f} deg")
    print("  reference: unrelated 2-planes in 64-d ~ 80-85 deg; identical ~ 0 deg")
    print("-" * 56)
    if angle < 5:
        print("  VERDICT: ~degenerate. Pick a more different B.")
    else:
        print("  VERDICT: GO. A and B occupy distinguishable planes.")
    print("=" * 56)


if __name__ == "__main__":
    main()