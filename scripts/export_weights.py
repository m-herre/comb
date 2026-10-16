"""Export a training checkpoint (``.ckpt``) as release weights: ``config.json`` + ``model.safetensors``.

    python scripts/export_weights.py runs/comb/stage3/step-10000.ckpt weights/

Drops the optimizer and scheduler state and checks that the export gives the checkpoint's outputs. The released
weights were trained with the research code (nanotabiclv2), whose config also names the path it took
(``RESEARCH_FIXED``); those keys must hold the values below and are dropped.
"""
import argparse
import hashlib
import json
from pathlib import Path

import inspect

import torch
from safetensors.torch import save_file

from comb.checkpoint import load_model
from comb.model import TabICL

SSMAX = "qassmax-mlp-elementwise"
RESEARCH_FIXED = dict(icl_type="cell", col_affine=False, col_feature_group="same", col_target_aware=True,
                      col_ssmax=SSMAX, icl_ssmax=SSMAX, row_rope_interleaved=False, norm_first=True, recompute=False)
RESEARCH_UNUSED = ("num_quantiles", "induct_layers", "induct_slots")  # regression / other summaries: no weights


def comb_config(config: dict) -> dict:
    """The research config as ``TabICL``'s arguments (identity for a checkpoint from ``comb.train``)."""
    wrong = {k: config[k] for k, v in RESEARCH_FIXED.items() if k in config and config[k] != v}
    if wrong:
        raise SystemExit(f"not the Comb architecture: {wrong}")
    out = {k: v for k, v in config.items() if k not in RESEARCH_FIXED and k not in RESEARCH_UNUSED}
    unknown = set(out) - set(inspect.signature(TabICL).parameters)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("ckpt")
    p.add_argument("out")
    args = p.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    config = comb_config(ck["config"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    state = {k: v.contiguous() for k, v in ck["state_dict"].items()}
    save_file(state, str(out / "model.safetensors"), metadata={"source_step": str(ck.get("curr_step"))})

    with torch.device("meta"):
        a = TabICL(**config)
    a.load_state_dict(ck["state_dict"], strict=True, assign=True)
    a, b = a.eval(), load_model(str(out)).eval()
    g = torch.Generator().manual_seed(0)
    X, y = torch.randn(1, 300, 7, generator=g), (torch.rand(1, 200, generator=g) * 3).floor()
    assert torch.equal(a(X, y), b(X, y)), "export changed the outputs"
    n = sum(v.numel() for v in state.values())
    digest = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    print(f"{n:,} parameters, step {ck.get('curr_step')}, sha256 {digest} -> {out}")


if __name__ == "__main__":
    main()
