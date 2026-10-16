"""The pretraining code (source checkout, ``pip install -e .[train]``): tiny CPU runs of the trainer."""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

pytest.importorskip("threadpoolctl")
train = pytest.importorskip("comb.train", reason="training code is in the source checkout, not the wheel")
SOURCE = str(pathlib.Path(train.__file__).parents[2])  # the subprocesses import this comb, not an installed one


def _train(tmp_path, out, *extra, nproc=None):
    cmd = [sys.executable, "-m", "comb.train.train"]
    if nproc:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
               "-m", "comb.train.train"]
    cmd += ["--tiny", "--device", "cpu", "--batch-size", "4", "--micro-batch-size", "2",
            "--workers", "0", "--wandb-mode", "disabled", "--out", str(tmp_path / out), *extra]
    path = os.pathsep.join([SOURCE, os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": path}
    subprocess.run(cmd, check=True, env=env)
    return [json.loads(line) for line in open(tmp_path / out / "metrics.jsonl")]


def test_tiny_cell_run_logs_and_resumes(tmp_path):
    """Per-step M from (seed, step): a resumed run draws the same M; the checkpoint keeps the inference M."""
    import numpy as np
    common = ["--cell-m", "48", "--cell-m-range", "16,64", "--log-every", "2",
              "--diag-every", "2", "--val-every", "2", "--val-groups", "2", "--save-every", "2"]
    rows = _train(tmp_path, "a", "--stage", "1", "--max-steps", "4", *common)
    assert [r["step"] for r in rows] == [1, 2, 3, 4] and all(np.isfinite(r["loss"]) for r in rows)
    logged = rows[1]
    for key in ("train/loss", "val/loss", "cell/m_mean", "cell/exact_micro_frac", "cell/cells", "cell/m",
                "optim/grad_norm_icl_blocks"):
        assert key in logged, key
    assert logged["cell/exact"] == 0.0 and logged["cell/cells"] <= logged["cell/m"]
    ck = torch.load(tmp_path / "a" / "step-4.ckpt", weights_only=True)
    assert ck["config"]["cell_m"] == 48
    # M is drawn from (seed, step) only: the logged M of a resumed run is the formula's
    import math
    def m_at(step):  # step = 0-based index of the optimizer step
        r = np.random.default_rng([42, 7349, step]).random()
        return int(round(math.exp(math.log(16) + r * (math.log(64) - math.log(16)))))
    assert logged["cell/m"] == m_at(1)
    resumed = _train(tmp_path, "a", "--stage", "1", "--max-steps", "6", *common)
    assert [r["step"] for r in resumed] == [1, 2, 3, 4, 5, 6] and resumed[5]["cell/m"] == m_at(5)


def test_ddp_cell_runs(tmp_path):
    import numpy as np
    rows = _train(tmp_path, "d", "--stage", "1", "--max-steps", "3", "--cell-m-range", "16,64",
                  "--log-every", "3", "--val-every", "3", "--val-groups", "2", nproc=2)
    assert [r["step"] for r in rows] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in rows)


def test_next_stage_initialises_from_the_previous(tmp_path):
    """The stage chain: stage 2 takes stage 1's weights with --init."""
    _train(tmp_path, "s1", "--stage", "1", "--max-steps", "2", "--val-every", "0", "--cell-m-range", "16,64")
    rows = _train(tmp_path, "s2", "--stage", "2", "--max-steps", "1", "--init", str(tmp_path / "s1" / "step-2.ckpt"),
                  "--cell-m-range", "16,64", "--val-every", "0", "--micro-batch-size", "1")
    assert rows[0]["step"] == 1
