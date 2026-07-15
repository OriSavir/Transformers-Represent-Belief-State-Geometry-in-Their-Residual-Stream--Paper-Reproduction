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


def mess3(a=0.85, x=0.05, name=None):
    """Parameterized symmetric Mess3 process (3 states, 3 tokens).

    Two knobs, both preserving the 3-state/3-token structure, the uniform
    stationary distribution, and the three simplex vertices:
      x : prob of switching to EACH other state (stay-prob = 1 - 2x). 0 < x < 0.5
      a : emission fidelity -- P(emit symbol aligned with DESTINATION state) = a;
          the other two symbols share (1 - a)/2 each. 1/3 < a < 1

    mess3(0.85, 0.05) reproduces the paper's hardcoded matrices exactly.
    """
    T = {0: np.zeros((3, 3)), 1: np.zeros((3, 3)), 2: np.zeros((3, 3))}
    for i in range(3):
        for j in range(3):
            p_trans = (1 - 2 * x) if i == j else x
            for s in range(3):
                p_emit = a if s == j else (1 - a) / 2
                T[s][i, j] = p_trans * p_emit
    return Process(
        name or f"mess3_a{a}_x{x}",
        T,
        token_names={0: "A", 1: "B", 2: "C"},
    )


# The paper's Mess3 == mess3(0.85, 0.05). Kept under the original name so the
# Fig 5/6 reproduction is unaffected.
# Self-note: I verified this with a print statement earlier
MESS3 = mess3(0.85, 0.05, name="mess3")

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
