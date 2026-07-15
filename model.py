"""The transformer used to reproduce Shai et al. (2024).

Use TransformerLens' HookedTransformer so that residual-stream activations at
every layer/position are available via `run_with_cache` — that cache is the entire
input to the probe later.

Confirmed from the paper:
  - d_model = 64           (probe regresses these 64-dim vectors onto beliefs)
  - context window = 10
  - probe target = final block's `resid_post`, i.e. before ln_final + unembed

Marked (verify A.6): values reconstructed pending a check of Appendix A.6. They
affect fidelity, not the correctness of the pipeline, and are one-line changes.
"""
import torch


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(
    process,
    n_layers=4,        # (verify A.6)
    d_model=64,        # confirmed
    n_heads=1,         # (verify A.6)
    d_head=8,         # (verify A.6) single head spanning the width
    d_mlp=256,         # (verify A.6)
    n_ctx=10,          # confirmed
    act_fn="relu",     # (verify A.6)
    seed=0,
    device=None,
):
    device = device or get_device()
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_mlp=d_mlp,
        n_ctx=n_ctx,
        d_vocab=process.n_tokens,
        act_fn=act_fn,
        normalization_type="LN",
        seed=seed,
        device=device,
    )
    model = HookedTransformer(cfg)
    return model


# the hook used to extract final resid stream, this can be edited without affecting the rest of the codebase, but must be consistent with the probe's target layer
def final_resid_hook(model):
    return f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"


if __name__ == "__main__":
    from processes import MESS3
    m = build_model(MESS3, device="cpu")
    print(m.cfg)
    print("final residual hook:", final_resid_hook(m))
    print("param count:", sum(p.numel() for p in m.parameters()))