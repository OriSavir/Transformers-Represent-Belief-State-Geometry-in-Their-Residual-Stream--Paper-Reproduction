"""Is the belief-plane determined by INIT or by PROCESS?

Trains several A-models -- SAME process A, DIFFERENT init seeds -- and prints the
pairwise angles between their probe planes. If these angles are large (~80-90 deg),
the plane is init-determined (same process still gives orthogonal planes), which
means cold-B is a confounded control for the reuse question. If they're small
(~0 deg), the plane is process-determined and cold-B is fine.
"""
import argparse, numpy as np
from feasibility_gate import train_from_scratch, probe_plane
from subspace import plane_distance_deg
from processes import mess3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--n_eval", type=int, default=4000)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    A = mess3(0.85, 0.05, name="A")
    eval_seqs = np.random.default_rng(1234).integers(0, 3, size=(a.n_eval, 10))

    models, mses = {}, {}
    for s in a.seeds:
        print(f"Training A (seed {s}) ...")
        m = train_from_scratch(A, a.steps, 64, 10, 1e-3, seed=s, device=a.device)
        W, mse, _ = probe_plane(m, A, eval_seqs, a.device)
        models[s], mses[s] = W, mse

    d_model = 64
    print("\nProbe MSEs (all should be similar & small, ~0.003):")
    for s in a.seeds:
        print(f"  seed {s}: {mses[s]:.5f}")

    print("\nPairwise plane angles (deg) -- SAME process A, different init:")
    print("        " + "".join(f"{s:>8}" for s in a.seeds))
    for i in a.seeds:
        row = "".join(f"{plane_distance_deg(models[i], models[j], d_model):>8.1f}"
                      for j in a.seeds)
        print(f"  seed {i:>2}" + row)
    print("\nDiagonal is 0 (a plane vs itself). Read the OFF-diagonal:")
    print("  ~80-90 deg -> INIT-determined  -> cold-B is a confounded Q2 control")
    print("  ~0 deg     -> PROCESS-determined -> cold-B control is fine")


if __name__ == "__main__":
    main()