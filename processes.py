"""
Data-generating processes (edge-emitting HMMs) from Shai et al. 2024.

Convention (from the paper):
    T[x][i, j] = Pr(emit token x AND move to state j | currently in state i)
The marginal state-to-state matrix  T_state = sum_x T[x]  is row-stochastic.
"""
import numpy as np


class Process:
    def __init__(self, name, transition_matrices, token_names=None):
        self.name = name
        self.T = {x: np.asarray(M, dtype=float) for x, M in transition_matrices.items()}
        self.tokens = sorted(self.T.keys())
        self.n_states = next(iter(self.T.values())).shape[0]
        self.n_tokens = len(self.tokens)
        self.token_names = token_names or {x: str(x) for x in self.tokens}
        self.T_state = sum(self.T.values())
        rs = self.T_state.sum(axis=1)
        assert np.allclose(rs, 1.0), f"{name}: state rows must sum to 1, got {rs}"
        self.stationary = self._stationary()

    def _stationary(self):
        vals, vecs = np.linalg.eig(self.T_state.T)
        pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return pi / pi.sum()

    def sample(self, length, rng=None):
        """Return a list of token ids of the given length."""
        rng = rng or np.random.default_rng()
        s = rng.choice(self.n_states, p=self.stationary)
        out = []
        for _ in range(length):
            joint = np.stack([self.T[x][s] for x in self.tokens])  # (n_tokens, n_states)
            flat = joint.ravel()
            idx = rng.choice(flat.size, p=flat)
            xi, sj = divmod(idx, self.n_states)
            out.append(self.tokens[xi])
            s = sj
        return out


MESS3 = Process(
    "mess3",
    {
        0: [[0.765,  0.00375, 0.00375],
            [0.0425, 0.0675,  0.00375],
            [0.0425, 0.00375, 0.0675]],
        1: [[0.0675,  0.0425, 0.00375],
            [0.00375, 0.765,  0.00375],
            [0.00375, 0.0425, 0.0675]],
        2: [[0.0675,  0.00375, 0.0425],
            [0.00375, 0.0675,  0.0425],
            [0.00375, 0.00375, 0.765]],
    },
    token_names={0: "A", 1: "B", 2: "C"},
)

RRXOR = Process(
    "rrxor",
    {
        0: [[0, 0.5, 0,   0,   0  ],
            [0, 0,   0,   0,   0.5],
            [0, 0,   0,   0.5, 0  ],
            [0, 0,   0,   0,   0  ],
            [1, 0,   0,   0,   0  ]],
        1: [[0, 0,   0.5, 0,   0  ],
            [0, 0,   0,   0.5, 0  ],
            [0, 0,   0,   0,   0.5],
            [1, 0,   0,   0,   0  ],
            [0, 0,   0,   0,   0  ]],
    },
    token_names={0: "0", 1: "1"},
)

PROCESSES = {"mess3": MESS3, "rrxor": RRXOR}

if __name__ == "__main__":
    for p in (MESS3, RRXOR):
        print(f"{p.name}: {p.n_states} states, {p.n_tokens} tokens, "
              f"stationary={np.round(p.stationary, 4)}")
        rng = np.random.default_rng(0)
        seq = p.sample(20, rng)
        print("  sample:", "".join(p.token_names[t] for t in seq))
