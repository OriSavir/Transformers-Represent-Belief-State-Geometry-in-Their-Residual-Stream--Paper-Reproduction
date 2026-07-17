"""Train the transformer on an HMM process (Shai et al. 2024).

Usage:
    python train.py --process mess3 --steps 2000 --device cpu

Each batch is freshly sampled from the process (the paper trains on a stream, not a
fixed dataset). We log next-token cross-entropy against the theoretical floor — the
average entropy of the true predictive distribution — so you can see the model
approach optimal prediction, which is the precondition for the belief geometry to appear.

Checkpointing is dense early, coarse late: a snapshot every --dense_every steps up to
--dense_until (to capture the geometry *emerging*, which happens in the first 100ish steps),
then every --ckpt_every steps thereafter. This is because I noticed that the geometry emerges very early, at least in my runs.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from processes import PROCESSES
from msp import belief_update
from model import build_model, get_device


def sample_batch(process, batch_size, seq_len, rng):
    seqs = [process.sample(seq_len, rng) for _ in range(batch_size)]
    return torch.tensor(seqs, dtype=torch.long)


def optimal_loss(process, seq_len, n_seqs=4000, rng=None):
    """Predictive-entropy floor: the lowest achievable mean next-token cross-entropy.

    At each position the optimal predictor outputs the true next-token distribution
    given the belief; its cross-entropy equals that distribution's entropy. We average
    over positions and many sampled sequences.
    """
    rng = rng or np.random.default_rng(0)
    ents = []
    for _ in range(n_seqs):
        seq = process.sample(seq_len, rng)
        eta = process.stationary.copy()
        for t in range(seq_len):
            probs = np.array([(eta @ process.T[x]).sum() for x in process.tokens])
            if t >= 1:  # training predicts positions 1..L-1; align the floor to match
                ents.append(-(probs * np.log(probs + 1e-12)).sum())
            eta, _ = belief_update(eta, process.T[seq[t]])
    return float(np.mean(ents))

#### BIG NOTE: the paper seems to train for 1,000,00 steps. This takes me way too long on my local computer (my laptop)
def train(process_name="mess3", steps=2000, batch_size=64, seq_len=10,
          optimizer="sgd", lr=0.01, ckpt_every=200, log_every=50,
          out_dir="checkpoints", seed=0, device=None,
          dense_until=300, dense_every=10):
    device = device or get_device()
    process = PROCESSES[process_name]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = build_model(process, n_ctx=seq_len, seed=seed, device=device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer)

    floor = optimal_loss(process, seq_len)
    print(f"[{process_name}] device={device}  params={sum(p.numel() for p in model.parameters())}"
          f"  optimal-loss floor ~ {floor:.4f} nats/token")

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{process_name}_loss.csv")
    with open(log_path, "w") as f:
        f.write("step,loss,floor\n")

    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sample_batch(process, batch_size, seq_len, rng).to(device)
        logits = model(batch)                          # (B, T, vocab)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, process.n_tokens),
            batch[:, 1:].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.6f},{floor:.6f}\n")
            rate = step / (time.time() - t0)
            print(f"  step {step:>7}  loss {loss.item():.4f}  "
                  f"(floor {floor:.4f}, gap {loss.item()-floor:+.4f})  {rate:.0f} it/s")

        # dense early (catch the emergence), coarse late (track the tail)
        dense = step <= dense_until and step % dense_every == 0
        coarse = step % ckpt_every == 0
        if dense or coarse:
            path = os.path.join(out_dir, f"{process_name}_step{step}.pt")
            torch.save({"step": step, "cfg": model.cfg, "state_dict": model.state_dict()}, path)

    final = os.path.join(out_dir, f"{process_name}_final.pt")
    torch.save({"step": steps, "cfg": model.cfg, "state_dict": model.state_dict()}, final)
    print(f"done in {time.time()-t0:.1f}s -> {final}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--process", default="mess3", choices=list(PROCESSES))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--optimizer", default="sgd", choices=["adam", "sgd"]) # paper uses sgd, just checked this
    p.add_argument("--lr", type=float, default=1e-2)  # paper uses 1e-2, just checked this as well
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--dense_until", type=int, default=300,
                   help="save densely (every --dense_every steps) up to this step")
    p.add_argument("--dense_every", type=int, default=10,
                   help="dense-phase checkpoint interval")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cpu / mps / cuda. Default auto. Use cpu to avoid the MPS warning.")
    a = p.parse_args()
    train(a.process, a.steps, a.batch_size, a.seq_len, a.optimizer, a.lr,
          a.ckpt_every, log_every=a.log_every, out_dir=a.out_dir, seed=a.seed,
          device=a.device, dense_until=a.dense_until, dense_every=a.dense_every)