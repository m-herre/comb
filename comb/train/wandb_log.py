"""Optional Weights & Biases logging for ``train.py``: fail-safe and offline by default.

* ``import wandb`` happens lazily; ``mode="disabled"`` or a missing package gives a no-op logger.
* Every call into wandb is guarded: a logging failure is printed once and never kills a multi-day job.
* Offline by default: the compute nodes have no outbound internet, so runs are written locally and
  ``wandb sync`` uploads them from a machine with internet access. An explicit ``mode`` wins, then
  ``WANDB_MODE``, then ``offline``.
* One run spans all three stages and every requeue: a fixed ``run_id`` with ``resume="allow"``, and every
  metric is plotted against ``global_step`` (``define_metric``), never wandb's internal step, so a resume
  that rewinds to a checkpoint still appends cleanly.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional

STEP_METRIC = "global_step"


def resolve_mode(mode: Optional[str]) -> str:
    mode = mode or os.environ.get("WANDB_MODE") or "offline"
    if mode not in ("online", "offline", "disabled"):
        raise ValueError(f"wandb mode must be online, offline or disabled; got {mode!r}")
    return mode


def numeric(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """The finite-or-not float metrics of ``metrics`` (bools and ints become floats; strings are dropped).
    Non-finite values are kept: a NaN loss is exactly what the stability panels must show."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif hasattr(v, "item") and getattr(v, "numel", lambda: 0)() == 1:
            out[k] = float(v.item())
    return out


class WandbLogger:
    """``log(metrics, global_step)`` / ``finish()``; a no-op when disabled or when wandb is unavailable."""

    def __init__(self, project: str, run_id: Optional[str], name: Optional[str] = None, group: Optional[str] = None,
                 config: Optional[dict] = None, mode: Optional[str] = None, directory: Optional[str] = None,
                 step_metric: str = STEP_METRIC):
        self.mode = resolve_mode(mode)
        self.step_metric = step_metric
        self.run = None
        self._warned = False
        if self.mode == "disabled":
            return
        try:
            import wandb

            if directory:
                os.makedirs(directory, exist_ok=True)
            self.run = wandb.init(project=project, id=run_id, name=name or run_id, group=group, config=config or {},
                                  mode=self.mode, dir=directory, resume="allow" if run_id else None)
            self.run.define_metric(step_metric)
            self.run.define_metric("*", step_metric=step_metric)
        except Exception as e:  # noqa: BLE001 -- logging must never stop training
            print(f"[wandb] disabled: init failed: {e!r}", flush=True)
            self.run = None

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, metrics: Mapping[str, Any], global_step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log({**numeric(metrics), self.step_metric: int(global_step)})
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                print(f"[wandb] log failed (further failures are silent): {e!r}", flush=True)
                self._warned = True

    def summary(self, values: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.summary.update(dict(values))
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:  # noqa: BLE001
            pass
        self.run = None


def finite_or_nan(x: float) -> float:
    return x if math.isfinite(x) else float("nan")
