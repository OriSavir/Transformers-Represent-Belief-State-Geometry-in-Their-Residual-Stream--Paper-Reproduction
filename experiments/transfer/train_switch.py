"""STEP 2 -- Train the A->B fine-tuning run plus the A->A drift control.

Produces (all forked from ONE converged A checkpoint, so init is held fixed):
  A_converged            : A trained from scratch, the shared pre-switch state
  warmB_step****.pt      : A_converged, then fine-tuned on B      (the experiment)
  toA_step****.pt        : A_converged, then trained further on A (drift control)

Checkpoints are dense right after the switch (fast dynamics) and sparse later.
Saved as plain state_dicts; analyze_switch.py reloads and probes them.
"""
import argparse, os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from processes import mess3
from model import build_model

CKPT_STEPS = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 4000]


def sample_batch(process, batch_size, seq_len, rng):
    seqs = [process.sample(seq_len, rng) for _ in range(batch_size)]
    return torch.tensor(seqs, dtype=torch.long)


def train_phase(model, process, steps, batch_size, seq_len, lr, rng, device,
                ckpt_steps=None, ckpt_prefix=None, out_dir=None):
    """Train `model` in place on `process` for `steps`. If ckpt_steps given, save
    a state_dict at each of those step counts (0 = before any update this phase)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_set = set(ckpt_steps or [])
    if 0 in ckpt_set:
        torch.save(model.state_dict(), os.path.join(out_dir, f"{ckpt_prefix}_step0000.pt"))
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sample_batch(process, batch_size, seq_len, rng).to(device)
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, process.n_tokens),
            batch[:, 1:].reshape(-1),
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if step in ckpt_set and out_dir is not None:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"{ckpt_prefix}_step{step:04d}.pt"))
        if step % 500 == 0 or step == 1:
            print(f"    [{process.name}] step {step:>4}  loss {loss.item():.4f}"
                  f"  ({step/(time.time()-t0):.0f} it/s)")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a_A", type=float, default=0.85)
    p.add_argument("--x_A", type=float, default=0.05)
    p.add_argument("--a_B", type=float, default=0.50)
    p.add_argument("--x_B", type=float, default=0.20)
    p.add_argument("--converge_steps", type=int, default=4000)
    p.add_argument("--phase_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="ckpts")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    A = mess3(a.a_A, a.x_A, name=f"A(a={a.a_A},x={a.x_A})")
    B = mess3(a.a_B, a.x_B, name=f"B(a={a.a_B},x={a.x_B})")

    # metadata so analyze_switch.py knows what these checkpoints are
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    device = a.device
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # --- Phase 0: converge on A, save the shared checkpoint ---
    print(f"Converging on A for {a.converge_steps} steps ...")
    model = build_model(A, n_ctx=a.seq_len, seed=a.seed, device=device)
    train_phase(model, A, a.converge_steps, a.batch_size, a.seq_len, a.lr, rng, device)
    a_path = os.path.join(a.out_dir, "A_converged.pt")
    torch.save(model.state_dict(), a_path)
    print(f"  saved {a_path}")

    # --- Fork 1: warm-B = A_converged fine-tuned on B ---
    print(f"\nFork warm-B: fine-tuning A_converged on B for {a.phase_steps} steps ...")
    m_warmB = build_model(B, n_ctx=a.seq_len, seed=a.seed, device=device)
    m_warmB.load_state_dict(torch.load(a_path, map_location=device))
    train_phase(m_warmB, B, a.phase_steps, a.batch_size, a.seq_len, a.lr,
                np.random.default_rng(a.seed + 100), device,
                ckpt_steps=CKPT_STEPS, ckpt_prefix="warmB", out_dir=a.out_dir)

    # --- Fork 2: A->A = A_converged trained further on A (drift control) ---
    print(f"\nFork A->A: continuing A_converged on A for {a.phase_steps} steps ...")
    m_toA = build_model(A, n_ctx=a.seq_len, seed=a.seed, device=device)
    m_toA.load_state_dict(torch.load(a_path, map_location=device))
    train_phase(m_toA, A, a.phase_steps, a.batch_size, a.seq_len, a.lr,
                np.random.default_rng(a.seed + 200), device,
                ckpt_steps=CKPT_STEPS, ckpt_prefix="toA", out_dir=a.out_dir)

    print(f"\nDone. Checkpoints in {a.out_dir}/  (A_converged.pt, warmB_step*.pt, toA_step*.pt)")


if __name__ == "__main__":
    main()