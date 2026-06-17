"""
Mixed-state presentation (MSP): the metadynamics of optimal belief updating.

Belief update (paper Eq. 1):   eta' = (eta @ T[x]) / (eta @ T[x]).sum()
The denominator is exactly Pr(next token = x | current belief eta).
"""
import numpy as np


def belief_update(eta, Tx, eps=1e-12):
    num = eta @ Tx
    p = num.sum()
    if p < eps:
        return None, 0.0
    return num / p, p


def belief_for_sequence(process, tokens, eta0=None):
    """Belief state after observing a token sequence (used to label activations)."""
    eta = process.stationary.copy() if eta0 is None else np.asarray(eta0, float)
    for x in tokens:
        eta, _ = belief_update(eta, process.T[x])
        if eta is None:
            raise ValueError("zero-probability sequence under this process")
    return eta


def belief_cloud(process, depth):
    """All belief states reached by every length-`depth` sequence, with path probs.

    Good for processes that never synchronize (e.g. Mess3) -> dense fractal.
    Returns (beliefs[N, n_states], probs[N]).
    """
    beliefs = [process.stationary.copy()]
    probs = [1.0]
    for _ in range(depth):
        nb, npr = [], []
        for eta, p in zip(beliefs, probs):
            for x in process.tokens:
                eta2, pe = belief_update(eta, process.T[x])
                if eta2 is not None and pe > 0:
                    nb.append(eta2)
                    npr.append(p * pe)
        beliefs, probs = nb, npr
    return np.array(beliefs), np.array(probs)


def distinct_belief_states(process, max_depth, tol=1e-6):
    """BFS over *distinct* belief states (the MSP state set).

    Good for processes that synchronize (e.g. RRXOR) -> finite set of points.
    """
    def key(eta):
        return tuple(np.round(eta / tol).astype(np.int64))

    start = process.stationary.copy()
    seen = {key(start): start}
    frontier = [start]
    for _ in range(max_depth):
        nf = []
        for eta in frontier:
            for x in process.tokens:
                eta2, pe = belief_update(eta, process.T[x])
                if eta2 is not None and pe > 1e-12:
                    k = key(eta2)
                    if k not in seen:
                        seen[k] = eta2
                        nf.append(eta2)
        frontier = nf
        if not frontier:
            break
    return np.array(list(seen.values()))


if __name__ == "__main__":
    from processes import MESS3, RRXOR

    b, pr = belief_cloud(MESS3, depth=10)
    print(f"mess3 cloud at depth 10: {len(b)} points (fractal, never synchronizes)")

    states = distinct_belief_states(RRXOR, max_depth=20)
    print(f"rrxor distinct belief states: {len(states)}  (paper says 36)")

    eta = belief_for_sequence(MESS3, [0, 1, 2, 0])
    print("mess3 belief after 'ABCA':", np.round(eta, 4), "sums to", round(eta.sum(), 6))
