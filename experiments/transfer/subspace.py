"""Compare *where* the belief triangle is painted in activation space.

The affine probe returns W of shape (d_model+1, |S|): the first d_model rows are
the linear part W_lin (d_model x |S|), the last row is the bias. The plane the
triangle lives on is the column space of W_lin. Because belief coords sum to 1,
those |S| columns are linearly dependent (they sum to ~0), so the column space is
(|S|-1)-dimensional -- a 2-plane for a 3-state process. We orthonormalize that
plane and compare two of them via principal angles.

principal angle 0 deg  -> identical plane (same directions used)
principal angle 90 deg -> orthogonal planes (completely different real estate)
"""
import numpy as np


def plane_basis(W, d_model, k=None):
    """Orthonormal basis (d_model x k) for the plane the probe paints on.

    W: (d_model+1, n_states) affine probe (incl. bias row). Uses the linear part.
    k: subspace dim to keep; default n_states-1 (the true simplex dimension).
    """
    W_lin = np.asarray(W)[:d_model]                 # (d_model, n_states)
    if k is None:
        k = W_lin.shape[1] - 1                      # simplex is (n_states-1)-dim
    U, S, _ = np.linalg.svd(W_lin, full_matrices=False)
    return U[:, :k], S                              # columns of U span the plane


def principal_angles_deg(Qa, Qb):
    """Principal angles (degrees, ascending) between two orthonormal bases."""
    M = Qa.T @ Qb
    sv = np.linalg.svd(M, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def plane_distance_deg(Wa, Wb, d_model, k=None):
    """Single scalar summary: the largest principal angle between two probe planes.

    Largest (not mean) is the conservative choice -- two planes are 'the same' only
    if they agree in *every* direction, so the worst-aligned direction is the honest
    headline number.
    """
    Qa, _ = plane_basis(Wa, d_model, k)
    Qb, _ = plane_basis(Wb, d_model, k)
    return float(principal_angles_deg(Qa, Qb).max())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 64
    Wl = rng.standard_normal((d, 3)); Wl -= Wl.mean(1, keepdims=True)
    W = np.vstack([Wl, rng.standard_normal((1, 3))])
    print("same plane, max angle:", round(plane_distance_deg(W, W, d), 3), "deg  (expect ~0)")
    Wl2 = rng.standard_normal((d, 3)); Wl2 -= Wl2.mean(1, keepdims=True)
    W2 = np.vstack([Wl2, rng.standard_normal((1, 3))])
    print("random plane, max angle:", round(plane_distance_deg(W, W2, d), 1),
          "deg  (expect large, ~80-90 for d=64)")