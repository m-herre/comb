"""Released weights: where they live and how they load.

``checkpoint=None`` resolves, in order, to the ``COMB_CHECKPOINT`` environment variable or the released weights on
Hugging Face (``REPO_ID`` at ``REVISION``, downloaded once and cached by ``huggingface_hub``; the download must match
``SHA256``, so a package version always runs the weights it was released with). A path may
be a folder with ``config.json`` + ``model.safetensors``, a ``.safetensors`` file next to its ``config.json``, or a
training checkpoint (``.ckpt``, ``{"config", "state_dict", ...}``).
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .model import TabICL

REPO_ID = "m-herre/Comb"
REVISION = "ccf447da404f8506a752d7adf77686e20a6aac6e"  # m-herre/Comb at tag v1.0.0
SHA256 = "33ea632c217a7f21c5b5f1608ebb0e4fca7034b2671437cc839651fedb59aaec"  # weights_digest of the released weights (cell-clf-v3-s4)
FILES = ("config.json", "model.safetensors")

_LOADED: dict = {}  # (resolved path, mtime, size, device) -> (config, state_dict on device)
_BUILT: dict = {}  # same key -> the network built on those weights (callers get a view: ``_view``)


def resolve(checkpoint: Optional[str] = None) -> Path:
    """The local file or folder that holds the weights (downloads the released weights if needed)."""
    checkpoint = checkpoint or os.environ.get("COMB_CHECKPOINT")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {checkpoint!r} does not exist")
        return path
    return _released()


@functools.lru_cache(maxsize=None)
def _released() -> Path:
    """The released weights' local folder; asked of the Hub once per process, not on every ``fit``."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(REPO_ID, revision=REVISION, allow_patterns=list(FILES)))
    if weights_digest(path) != SHA256:
        raise ValueError(f"the weights downloaded from {REPO_ID}@{REVISION} are not this version's "
                         f"(sha256 {weights_digest(path)[:12]}... instead of {SHA256[:12]}...)")
    return path


_DIGESTS: dict = {}  # (resolved path, mtime, size) -> sha256


def _weight_files(path: Path) -> list:
    if path.is_dir():
        return [path / "config.json", path / "model.safetensors"]
    return [path.parent / "config.json", path] if path.suffix == ".safetensors" else [path]


def weights_digest(checkpoint) -> str:
    """sha256 of the weights (and their config) at ``checkpoint``; computed once per file and process."""
    path = Path(checkpoint).expanduser().resolve()
    files = _weight_files(path)
    key = (str(path), *[(f.stat().st_mtime_ns, f.stat().st_size) for f in files])
    if key not in _DIGESTS:
        h = hashlib.sha256()
        for f in files:
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
        _DIGESTS[key] = h.hexdigest()
    return _DIGESTS[key]


def find_weights(path: str, digest: str) -> str:
    """The weights a pickled model was fitted with: ``path`` if it still holds them, else ``COMB_CHECKPOINT`` or the
    released weights if they do. Raises instead of silently attaching different weights."""
    candidates = [Path(path)] if Path(path).exists() else []
    try:
        candidates.append(resolve(None))
    except Exception:  # noqa: BLE001 -- no COMB_CHECKPOINT and no Hub access: only the stored path counts
        pass
    for c in candidates:
        if weights_digest(c) == digest:
            return str(c.resolve())
    if candidates and Path(path).exists():
        raise ValueError(f"the weights at {path} changed since this model was saved (sha256 {digest[:12]}...); "
                         "point COMB_CHECKPOINT to the weights it was fitted with")
    raise FileNotFoundError(f"the weights this model was fitted with (sha256 {digest[:12]}...) are not at {path}; "
                            "set COMB_CHECKPOINT to a copy of them")


def _read(path: Path, device: torch.device) -> Tuple[dict, dict]:
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        config = json.loads((path.parent / "config.json").read_text())
        return config, load_file(str(path), device=str(device))
    ck = torch.load(path, map_location=device, weights_only=True)
    return ck["config"], ck["state_dict"]


def load_model(checkpoint: Optional[str] = None, device="cpu") -> TabICL:
    """The network in eval mode on ``device``. Each file is read once per process and device, and every network
    built from it shares those weights (inference never writes them): no copy per model, in time or memory."""
    device = torch.device(device)
    path = resolve(checkpoint).resolve()
    stat = (path / "model.safetensors" if path.is_dir() else path).stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, str(device))
    if key not in _LOADED:
        _LOADED[key] = _read(path, device)
    if key not in _BUILT:
        _BUILT[key] = build(*_LOADED[key])
    return _view(_BUILT[key])


def _view(model: TabICL) -> TabICL:
    """A network that shares ``model``'s submodules and weights but has its own cell budget ``num_cells`` (the only
    attribute a classifier sets). Building a new network costs ~40 ms of module construction per ``fit``."""
    view = copy.copy(model)
    view._modules = dict(model._modules)
    view._modules["icl_predictor"] = copy.copy(model._modules["icl_predictor"])
    return view


def build(config: dict, state_dict: dict) -> TabICL:
    """The network in eval mode on the tensors of ``state_dict`` (shared, not copied)."""
    with torch.device("meta"):
        model = TabICL(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()
