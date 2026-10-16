# Ported from tabicl 2.2.0 (BSD-3-Clause, Copyright (c) 2025, Soda team @ Inria); see LICENSE-tabicl.
"""Pretraining of Comb: TabICLv2's pretraining (``tabicl/train/_run.py`` at the v2 stage-script settings) with the
in-context stage over cells.

    python -m comb.train.train --stage {1,2,3} --out DIR [--init CKPT] [--max-steps N] ...
    torchrun --standalone --nproc_per_node=4 -m comb.train.train ...   # DDP

Seeding, the DataLoader worker seeding and round-robin order, micro-batching, loss, clipping, Muon and the
warmup-cosine schedule follow upstream's order. Defaults are the values of upstream tabicl's ``scripts/train_v2_clf_stage{1,2,3}.sh``.

  * Cells: every step runs the in-context stage over <= M cells (``cell.py``), with M drawn log-uniformly from
    ``--cell-m-range`` (from the seed and the step, so every rank and every resume draws the same M);
    ``--cell-m`` is used at validation and saved in the checkpoints.
  * fp32 with TF32; no GradScaler, FlashAttention-3 or recompute (stages 2/3 run fp32 SDPA).
  * a CUDA OOM in a micro-batch is re-raised (upstream skips it, which would change the trajectory).
  * per-step ``{"step", "loss", "lr"}`` go to ``DIR/metrics.jsonl``; every ``--log-every`` steps also the
    instrumentation (``train/ optim/ stability/ data/ perf/ cell/``), every ``--diag-every`` steps per-module
    diagnostics, every ``--val-every`` steps the loss on a fixed held-out prior set (``val/``); all of it also
    goes to W&B (``--wandb-mode``, offline by default). None of it changes the numerics of a step.
  * a step whose gradient norm is not finite is skipped (``--skip-nonfinite 1``, the default).
  * a checkpoint is also written at the final step; resume truncates ``metrics.jsonl`` to the checkpoint step.
  * DDP (``--balance 1``, the default): micro-batches move between ranks so that every rank has about the same
    work (``Trainer._balance``); only the gradient's summation order differs from upstream.
  * the CUDA cache is emptied after a step only when it holds most of the GPU (``Trainer._release_cache``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..model import TabICL
from .optim import Muon, cosine_with_warmup
from .prior import STAGE_CONFIG, STAGES, PriorDataset
from .wandb_log import WandbLogger

TINY = dict(embed_dim=16, icl_num_blocks=1, col_num_blocks=1, row_num_blocks=1, col_num_inds=8)  # tests only

# Per-stage values of the upstream classification stage scripts.
STAGE_DEFAULTS = {
    1: dict(max_steps=500_000, lr=8e-4, clip=10.0, micro_batch_size=4, save_every=500, save_perm_every=5000),
    2: dict(max_steps=40_000, lr=1e-4, clip=10.0, micro_batch_size=1, save_every=50, save_perm_every=1000),
    3: dict(max_steps=10_000, lr=2e-5, clip=1.0, micro_batch_size=1, save_every=50, save_perm_every=1000),
}
SEED = 42  # --np_seed / --torch_seed
MAX_CLASSES = 10
WEIGHT_DECAY, MOMENTUM, WARMUP_PROPORTION, LR_END = 0.01, 0.9, 0.01, 1e-7
MAX_TEMP_CHECKPOINTS = 5  # upstream --max_checkpoints
EMPTY_CACHE_ABOVE = 0.75  # empty the CUDA cache after a step only above this fraction of GPU memory reserved
# micro_batch_cost coefficients, fitted to RTX PRO 6000 micro-batch times (R2 0.996, median error 6%)
COST_ROWS, COST_CELLS, COST_ICL, COST_MICRO = 1.0, 0.26, 0.0055, 20_800.0

# Instrumentation
STAGE_OFFSETS = {1: 0, 2: 500_000, 3: 540_000}  # global step at the start of each stage (upstream step counts)
VAL_TABLE_GROUPS = {1: 64, 2: 64, 3: 64}  # held-out prior groups per stage (stage 1 groups hold 4 tables)
VAL_SEED = 1_000_000
CLASS_BUCKETS = ((2, 2), (3, 5), (6, 10))  # inclusive class-count ranges
LEN_EDGES = (256, 1024, 4096, 16384)  # sequence-length bucket edges
MODULE_GROUPS = (("col", "col_embedder."), ("row", "row_interactor."), ("icl_blocks", "icl_predictor.tf_icl."),
                 ("label", "icl_predictor.y_encoder."), ("head", "icl_predictor.decoder."), ("icl_ln", "icl_predictor.ln."))
SPIKE_SIGMAS = 4.0


def module_group(name: str) -> str:
    for group, prefix in MODULE_GROUPS:
        if name.startswith(prefix):
            return group
    return "other"


def class_bucket(n_classes: int) -> int:
    for i, (lo, hi) in enumerate(CLASS_BUCKETS):
        if lo <= n_classes <= hi:
            return i
    return len(CLASS_BUCKETS) - 1


def len_bucket(seq_len: int) -> int:
    for i, edge in enumerate(LEN_EDGES):
        if seq_len <= edge:
            return i
    return len(LEN_EDGES)


def bucket_names():
    cls = [f"cls{lo}" if lo == hi else f"cls{lo}_{hi}" for lo, hi in CLASS_BUCKETS]
    lo = (0,) + LEN_EDGES
    lens = [f"len{a + 1}_{b}" for a, b in zip(lo, LEN_EDGES)] + [f"len{LEN_EDGES[-1] + 1}_up"]
    return cls, lens


def git_sha() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def seed(s: int) -> None:
    """Global seeding as in upstream ``Trainer.configure_ddp`` / ``Trainer.seed`` (``_run.py:142-146, 520-528``)."""
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


_seed = seed  # make_stream's ``seed`` argument shadows the function


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` (upstream ``prior/_genload.py:50-55``): numpy follows the worker's torch seed."""
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


def make_stream(prior_kwargs: Dict[str, Any], seed: int, workers: int, start_step: int = 0, rank: int = 0,
                pin_memory: bool = False) -> Iterator:
    """Infinite batch stream in upstream's order: re-seed with ``seed + rank + start_step`` (``_run.py:520-528``),
    then iterate ``DataLoader(PriorDataset(...))``; worker ``w`` is seeded ``base_seed + w`` from the main torch
    RNG and batches arrive round-robin. ``workers == 0`` generates in-process (not upstream's order)."""
    _seed(seed + rank + start_step)
    dataset = PriorDataset(**prior_kwargs)
    if workers == 0:
        return iter(dataset)
    loader = DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=2, pin_memory=pin_memory,
        worker_init_fn=seed_worker, persistent_workers=True, multiprocessing_context="spawn",
    )  # fmt: skip
    return iter(loader)


def _ckpt_step(name: str) -> Optional[int]:
    m = re.fullmatch(r"step-(\d+)\.ckpt", name)
    return int(m.group(1)) if m else None


def latest_checkpoint(out: str) -> Optional[str]:
    if not os.path.isdir(out):
        return None
    steps = [s for s in map(_ckpt_step, os.listdir(out)) if s is not None]
    return os.path.join(out, f"step-{max(steps)}.ckpt") if steps else None


def micro_batch_cost(tables: int, seq_len: int, features: int, train_size: int, cell_m: int) -> float:
    """Relative forward/backward time of a micro-batch, for balancing ranks (fitted to measured step times). The
    in-context stage attends over <= cell_m cells (plus a k-means term, rows x cells, counted like it)."""
    if train_size > cell_m:
        test = seq_len - train_size
        icl = (test + cell_m) * cell_m + train_size * cell_m
        return tables * (COST_ROWS * seq_len + COST_CELLS * seq_len * features + COST_ICL * icl) + COST_MICRO
    test = seq_len - train_size
    return tables * (COST_ROWS * seq_len + COST_CELLS * seq_len * features + COST_ICL * test * train_size) + COST_MICRO


def balanced_assignment(costs: List[List[float]]) -> List[List[int]]:
    """Owner rank of each rank's micro-batches: longest first onto the least-loaded rank (its own on ties).
    Deterministic, so every rank computes the same assignment."""
    world = len(costs)
    load = [0.0] * world
    owner = [[r] * len(c) for r, c in enumerate(costs)]
    for c, r, i in sorted(((c, r, i) for r, cs in enumerate(costs) for i, c in enumerate(cs)), key=lambda t: (-t[0], t[1], t[2])):
        least = min(load)
        dst = r if load[r] == least else load.index(least)
        owner[r][i] = dst
        load[dst] += c
    return owner


class Trainer:
    """Single-process or torchrun DDP trainer. ``run()`` trains to ``max_steps`` and returns the new steps' metrics."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # DDP (upstream configure_ddp): per-rank batch, rank-offset seed, device per local rank.
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:  # --device cpu runs DDP over gloo (for smoke tests); otherwise NCCL, one GPU per rank
            on_gpu = args.device != "cpu"
            torch.distributed.init_process_group(backend="nccl" if on_gpu else "gloo")
            self.rank, self.world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = f"cuda:{self.local_rank}" if on_gpu else "cpu"
            if on_gpu:
                torch.cuda.set_device(self.device)
        else:
            self.rank, self.world, self.device = 0, 1, args.device
        if args.batch_size % self.world:
            raise ValueError(f"--batch-size {args.batch_size} is not divisible by the world size {self.world}")
        self.batch_size = args.batch_size // self.world
        self.master = self.rank == 0
        self.cuda = "cuda" in self.device
        self.device_memory = torch.cuda.get_device_properties(self.device).total_memory if self.cuda else 0
        self.curr_step = 0
        seed(args.seed + self.rank)

        # Model (upstream build_model): constructed on CPU right after seeding, then moved.
        self.cell_range = None
        if args.cell_m_range:
            lo, hi = (int(v) for v in args.cell_m_range.split(","))
            if not MAX_CLASSES <= lo <= hi:
                raise ValueError(f"--cell-m-range {args.cell_m_range!r}: need {MAX_CLASSES} <= lo <= hi")
            self.cell_range = (lo, hi)
        self.cell_m_now = args.cell_m
        model = TabICL(**dict(TINY if args.tiny else {}, max_classes=MAX_CLASSES, cell_m=args.cell_m))
        self.model_config = model.config
        model.to(device=self.device)
        self.raw_model = model

        # Optimizer and schedule (upstream configure_optimizer, get_scheduler).
        self.optimizer = Muon(list(model.parameters()), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM,
                              nesterov=True, ns_steps=5, matched_adamw_rms=0.2)
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_steps=args.max_steps * WARMUP_PROPORTION,
                                            total_steps=args.max_steps, lr_end=LR_END)

        # Checkpoint: resume from DIR/step-*.ckpt, else optionally take weights from --init.
        resume = latest_checkpoint(args.out)
        if resume is not None and args.init is not None:
            raise ValueError(f"--init given but {args.out} already holds checkpoints; resume needs no --init")
        if resume is not None:
            ck = torch.load(resume, map_location=self.device, weights_only=True)
            self._check_config(ck, resume)
            model.load_state_dict(ck["state_dict"])
            self.optimizer.load_state_dict(ck["optimizer_state"])
            self.scheduler.load_state_dict(ck["scheduler_state"])
            self.curr_step = ck["curr_step"]
            self._log(f"resuming from {resume} at step {self.curr_step}")
        elif args.init is not None:
            ck = torch.load(args.init, map_location=self.device, weights_only=True)
            self._check_config(ck, args.init)
            model.load_state_dict(ck["state_dict"])
            self._log(f"initialised weights from {args.init}")

        if self.ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(model, device_ids=[self.local_rank] if self.cuda else None, broadcast_buffers=False)
        else:
            self.model = model

        # Upstream runs the forward under torch.autocast(cuda, float32) (--amp True is the default and the
        # scripts pass --dtype float32), with the cuDNN SDPA backend excluded.
        self.amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float32)) if self.cuda else nullcontext
        if self.cuda and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            self.sdpa_ctx = lambda: sdpa_kernel(backends)
        else:
            self.sdpa_ctx = nullcontext

        self.prior_kwargs = dict(STAGES[f"stage{args.stage}"], batch_size=self.batch_size, max_classes=MAX_CLASSES,
                                 config=STAGE_CONFIG)

        # Instrumentation state (none of it touches the numerics of a step)
        self.offset = args.global_step_offset if args.global_step_offset is not None else STAGE_OFFSETS[args.stage]
        n_cls, n_len = len(CLASS_BUCKETS), len(LEN_EDGES) + 1
        self.n_buckets = n_cls + n_len
        self._window = self._new_window()
        self.loss_ema, self.loss_var = None, 0.0
        self.skipped_total, self.nonfinite_total, self.cache_releases = 0, 0, 0
        self.imbalance = 1.0
        self.val_set = None
        self.logger = None
        if self.master:
            params = {g: 0 for g, _ in MODULE_GROUPS}
            for name, p in model.named_parameters():
                params[module_group(name)] = params.get(module_group(name), 0) + p.numel()
            config = dict(args=vars(args), model=self.model_config, git_sha=git_sha(), world=self.world,
                          params_total=sum(p.numel() for p in model.parameters()),
                          params={g: n for g, n in params.items() if n})
            self.logger = WandbLogger(project=args.wandb_project, run_id=args.wandb_run_id, name=args.wandb_run_id,
                                      group=args.wandb_group, config=config, mode=args.wandb_mode,
                                      directory=args.wandb_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                                               "wandb"))

    def _check_config(self, ck, path) -> None:
        """A checkpoint must hold this architecture (the cell budget may differ: it is only the inference M)."""
        have = {k: v for k, v in ck.get("config", {}).items() if k != "cell_m"}
        want = {k: v for k, v in self.model_config.items() if k != "cell_m"}
        if have != want:
            raise ValueError(f"{path} holds a different model: {have} vs {want}")

    def _new_window(self) -> Dict[str, Any]:
        z = torch.zeros(self.n_buckets, 3, dtype=torch.float64, device=self.device)  # (loss sum, acc sum, tables)
        return dict(buckets=z, max_logit=0.0, tables=0.0, cells=0.0, rows=0.0, micro=0.0, steps=0, data_time=0.0,
                    step_time=0.0, loss_min=float("inf"), loss_max=float("-inf"), grad_norm_sum=0.0, clipped=0,
                    train_rows=0.0, features=0.0, classes=0.0, max_seq=0.0, max_features=0.0, imbalance=0.0,
                    cell_m=0.0, cell_exact=0.0, t0=time.time())

    def _release_cache(self) -> bool:
        """Upstream empties the CUDA cache after every step. The next step then allocates all its memory again,
        which costs its first micro-batch 25-40 ms under DDP (~15-25% of a stage-1 step); an allocation that
        fails frees the cache and retries anyway. So the cache is emptied only when it holds most of the GPU,
        where fragmentation across the varying shapes of stages 2/3 could otherwise tighten memory."""
        if self.cuda and torch.cuda.memory_reserved(self.device) > EMPTY_CACHE_ABOVE * self.device_memory:
            torch.cuda.empty_cache()
            return True
        return False

    def _log(self, msg: str) -> None:
        if self.master:
            print(msg, flush=True)

    # -------------------------------------------------------------------------------------------- step

    def _prepare(self, micro_batch) -> Tuple[Tensor, Tensor, int]:
        """A micro-batch's (X, y, train_size) as run by upstream ``run_micro_batch``: cut to its sequence length
        and largest feature count. Still on the CPU: it moves to the device only right before it runs."""
        X, y, d, seq_lens, train_sizes = micro_batch
        if len(torch.unique(seq_lens)) > 1:
            raise ValueError("All datasets in the micro batch must have the same sequence length.")
        if len(torch.unique(train_sizes)) > 1:
            raise ValueError("All datasets in the micro batch must have the same training size.")
        seq_len, train_size = seq_lens[0].item(), train_sizes[0].item()
        X, y = X[:, :seq_len], y[:, :seq_len]
        X = X[..., : d.max().item()]
        return X, y, train_size

    def run_micro_batch(self, X: Tensor, y: Tensor, train_size: int, last: bool, num_micro: int) -> float:
        """Forward/backward of one prepared micro-batch (upstream ``run_micro_batch``); returns the scaled loss."""
        y_train, y_test = y[:, :train_size], y[:, train_size:]
        self._window["cell_exact"] += float(train_size <= self.cell_m_now)

        # DDP gradient sync on the last micro-batch only; the per-table count d is ignored (--ignore_d True).
        sync = nullcontext() if not self.ddp or last else self.model.no_sync()
        with sync:
            with self.amp_ctx(), self.sdpa_ctx():
                pred = self.model(X, y_train)
                loss = F.cross_entropy(pred.flatten(end_dim=-2), y_test.long().flatten())
            scaled = loss / num_micro
            scaled.backward()
        self._record_micro(pred.detach(), y, train_size, X.shape[-1], loss.detach())
        return scaled.item()

    @torch.no_grad()
    def _record_micro(self, pred: Tensor, y: Tensor, train_size: int, features: int, loss: Tensor) -> None:
        """Adds one micro-batch to the logging window: per-table loss and accuracy by class-count and length
        bucket, the largest logit, and the data shape counters."""
        w = self._window
        B, T = y.shape
        w["tables"] += B
        w["cells"] += B * T * features
        w["rows"] += B * T
        w["micro"] += 1
        w["train_rows"] += B * train_size
        w["features"] += B * features
        w["max_seq"] = max(w["max_seq"], T)
        w["max_features"] = max(w["max_features"], features)
        w["max_logit"] = max(w["max_logit"], float(pred.abs().max()))
        lv = float(loss)
        w["loss_min"], w["loss_max"] = min(w["loss_min"], lv), max(w["loss_max"], lv)
        lb = len(CLASS_BUCKETS) + len_bucket(T)
        y_test = y[:, train_size:].long()
        ce = F.cross_entropy(pred.float().flatten(end_dim=-2), y_test.flatten(), reduction="none").view(B, -1).mean(1)
        acc = (pred.argmax(-1) == y_test).float().mean(1)
        n_classes = (y.long().amax(1) + 1).tolist()
        w["classes"] += sum(n_classes)
        rows = torch.stack([ce.double(), acc.double(), torch.ones_like(ce, dtype=torch.float64)], 1)  # (B, 3)
        idx = torch.tensor([class_bucket(int(c)) for c in n_classes], device=rows.device)
        w["buckets"].index_add_(0, idx, rows)
        w["buckets"][lb] += rows.sum(0)

    def _balance(self, prepared: List[Tuple[Tensor, Tensor, int]]) -> List[Tuple[int, int, Tuple[Tensor, Tensor, int]]]:
        """Redistributes the ranks' micro-batches so that every rank has about the same work: tables are drawn
        per rank (upstream's stream), and their sizes vary widely in stages 2/3, so without this each step waits
        for the rank with the largest tables. Every micro-batch still enters the averaged gradient with weight
        1 / (world * num_micro), whichever rank runs it, so only the summation order changes. Returns this
        rank's (source rank, index, micro-batch) in (source rank, index) order; its own micro-batches stay on the
        CPU, received ones are on the device. Sends and receives are issued in the same global (source rank,
        index) order on every rank, so each peer pair matches its operations in order."""
        shapes = torch.tensor([[X.shape[0], X.shape[1], X.shape[2], t] for X, _, t in prepared],
                              dtype=torch.long, device=self.device)
        every = [torch.empty_like(shapes) for _ in range(self.world)]
        torch.distributed.all_gather(every, shapes)
        every = [e.tolist() for e in every]
        costs = [[micro_batch_cost(*m, cell_m=self.cell_m_now) for m in e] for e in every]
        owner = balanced_assignment(costs)
        load = [0.0] * self.world
        for r in range(self.world):
            for i, c in enumerate(costs[r]):
                load[owner[r][i]] += c
        self.imbalance = max(load) / max(sum(load) / self.world, 1e-9)

        ops, received = [], {}
        for r in range(self.world):
            for i, (b, seq_len, feats, _) in enumerate(every[r]):
                dst = owner[r][i]
                if dst == r:
                    continue
                if r == self.rank:
                    X, y, _ = prepared[i]  # to the device only to be sent; freed after the exchange
                    ops += [torch.distributed.P2POp(torch.distributed.isend, X.to(self.device).contiguous(), dst),
                            torch.distributed.P2POp(torch.distributed.isend, y.to(self.device).contiguous(), dst)]
                elif dst == self.rank:
                    X = torch.empty((b, seq_len, feats), dtype=prepared[0][0].dtype, device=self.device)
                    y = torch.empty((b, seq_len), dtype=prepared[0][1].dtype, device=self.device)
                    ops += [torch.distributed.P2POp(torch.distributed.irecv, X, r),
                            torch.distributed.P2POp(torch.distributed.irecv, y, r)]
                    received[r, i] = X, y
        if ops:
            for work in torch.distributed.batch_isend_irecv(ops):
                work.wait()
            del ops
        mine = []
        for r in range(self.world):
            for i, m in enumerate(every[r]):
                if owner[r][i] == self.rank:
                    mine.append((r, i, prepared[i] if r == self.rank else (*received[r, i], m[3])))
        return mine

    def run_batch(self, batch) -> Dict[str, float]:
        """One optimizer step over a batch (upstream ``run_batch``). A CUDA OOM propagates."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        icl = self.raw_model.icl_predictor
        self.cell_m_now = self._cell_m(self.curr_step)
        icl.num_cells = self.cell_m_now
        self._window["cell_m"] += self.cell_m_now
        if self._diag_due() and self.master:
            icl.diag = {}  # filled by this step's forwards (the last micro-batch's stats remain)
        try:
            return self._run_batch(batch)
        finally:  # validation, diagnostics and checkpoints use the inference M
            icl.num_cells = self.args.cell_m
            icl.diag = None

    def _cell_m(self, step: int) -> int:
        """The step's cell budget: log-uniform in --cell-m-range, from (seed, step) only."""
        if self.cell_range is None:
            return self.args.cell_m
        lo, hi = self.cell_range
        r = np.random.default_rng([self.args.seed, 7349, step]).random()
        return int(round(math.exp(math.log(lo) + r * (math.log(hi) - math.log(lo)))))

    def _run_batch(self, batch) -> Dict[str, float]:
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        mbs = self.args.micro_batch_size
        num_micro = math.ceil(self.batch_size / mbs)
        micro_batches = list(zip(*[torch.split(t, mbs, dim=0) for t in batch]))
        prepared = [self._prepare(mb) for mb in micro_batches]

        if self.ddp and self.args.balance:
            work = self._balance(prepared)
            scaled = torch.zeros(self.world, num_micro, dtype=torch.float64, device=self.device)
            for j in range(len(work)):
                r, i, (X, y, train_size) = work[j]
                work[j] = None  # a micro-batch's inputs are freed once it has run
                scaled[r, i] = self.run_micro_batch(X.to(self.device), y.to(self.device), train_size,
                                                    j == len(work) - 1, num_micro)
                del X, y
            torch.distributed.all_reduce(scaled)  # one nonzero entry per micro-batch: exact
            loss = 0.0
            for v in scaled[self.rank].tolist():  # the logged loss is this rank's own batch, as upstream
                loss += v
        else:
            loss = 0.0
            for idx, (X, y, train_size) in enumerate(prepared):
                loss += self.run_micro_batch(X.to(self.device), y.to(self.device), train_size, idx == num_micro - 1,
                                             num_micro)

        diag = self._diag_due()
        group_grad = self._group_norms(grads=True) if diag else None
        if self.args.clip > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip))
        else:
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")))
        lr = self.optimizer.param_groups[0]["lr"]
        skip = self.args.skip_nonfinite and not math.isfinite(grad_norm)  # identical on every rank (synced grads)
        before = [p.detach().clone() for p in self.raw_model.parameters()] if diag and not skip else None
        if skip:
            self.skipped_total += 1
            self._log(f"step {self.curr_step + 1}: non-finite gradient norm; optimizer step skipped")
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        if not math.isfinite(loss):
            self.nonfinite_total += 1
        out = {"loss": loss, "lr": lr, "grad_norm": grad_norm, "skipped": float(skip)}
        if diag:
            out["diag"] = self._diagnostics(group_grad, before)
            icl_diag = getattr(self.raw_model.icl_predictor, "diag", None)
            if icl_diag:
                out["diag"].update({f"cell/{k}": v for k, v in icl_diag.items()})
                out["diag"]["cell/m"] = float(self.cell_m_now)
        return out

    # -------------------------------------------------------------------------------------------- instrumentation

    def _diag_due(self) -> bool:
        return self.args.diag_every > 0 and (self.curr_step + 1) % self.args.diag_every == 0

    @torch.no_grad()
    def _group_norms(self, grads: bool) -> Dict[str, float]:
        sq: Dict[str, float] = {}
        for name, p in self.raw_model.named_parameters():
            t = p.grad if grads else p
            if t is None:
                continue
            g = module_group(name)
            sq[g] = sq.get(g, 0.0) + float(t.detach().double().pow(2).sum())
        return {g: math.sqrt(v) for g, v in sq.items()}

    @torch.no_grad()
    def _diagnostics(self, group_grad, before) -> Dict[str, float]:
        """Per-module gradient norm (before clipping), parameter norm and update/parameter ratio."""
        out = {f"optim/grad_norm_{g}": v for g, v in group_grad.items()}
        out.update({f"optim/param_norm_{g}": v for g, v in self._group_norms(grads=False).items()})
        if before is not None:
            upd: Dict[str, float] = {}
            ref: Dict[str, float] = {}
            for (name, p), b in zip(self.raw_model.named_parameters(), before):
                g = module_group(name)
                upd[g] = upd.get(g, 0.0) + float((p.detach() - b).double().pow(2).sum())
                ref[g] = ref.get(g, 0.0) + float(b.double().pow(2).sum())
            out.update({f"optim/update_ratio_{g}": math.sqrt(upd[g] / max(ref[g], 1e-30)) for g in upd})
        return out

    # -------------------------------------------------------------------------------------------- loop

    def save_checkpoint(self) -> None:
        os.makedirs(self.args.out, exist_ok=True)
        path = os.path.join(self.args.out, f"step-{self.curr_step}.ckpt")
        torch.save(
            {
                "config": self.model_config,
                "state_dict": self.raw_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "curr_step": self.curr_step,
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)  # atomic, so a requeue never resumes from a partial file
        # Keep the newest MAX_TEMP_CHECKPOINTS temporary ones (upstream manage_checkpoint).
        temp = sorted(s for s in map(_ckpt_step, os.listdir(self.args.out))
                      if s is not None and s % self.args.save_perm_every != 0)
        for s in temp[:-MAX_TEMP_CHECKPOINTS]:
            os.remove(os.path.join(self.args.out, f"step-{s}.ckpt"))

    def _open_metrics(self):
        os.makedirs(self.args.out, exist_ok=True)
        path = os.path.join(self.args.out, "metrics.jsonl")
        keep = []
        if os.path.exists(path):  # drop rows past the checkpoint we resumed from
            with open(path) as f:
                keep = [line for line in f if line.strip() and json.loads(line)["step"] <= self.curr_step]
        with open(path, "w") as f:
            f.writelines(keep)
        return open(path, "a")

    def run(self) -> List[Dict[str, float]]:
        try:
            return self._run()
        finally:
            if self.ddp:
                torch.distributed.destroy_process_group()

    def _run(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        if self.curr_step >= self.args.max_steps:
            self._log(f"already at step {self.curr_step} >= --max-steps {self.args.max_steps}; nothing to do")
            return history
        metrics = self._open_metrics() if self.master else None
        if self.args.val_every > 0:
            self.val_set = self._load_val_set()
        stream = make_stream(self.prior_kwargs, self.args.seed, self.args.workers, start_step=self.curr_step,
                             rank=self.rank, pin_memory=self.cuda)
        try:
            t_prev = time.time()
            for step in range(self.curr_step, self.args.max_steps):
                t_data = time.time()
                batch = next(stream)
                self._window["data_time"] += time.time() - t_data
                results = self.run_batch(batch)
                if self._release_cache():
                    self.cache_releases += 1
                self.curr_step = step + 1
                now = time.time()
                self._window["step_time"] += now - t_prev
                t_prev = now
                self._window["steps"] += 1
                self._window["grad_norm_sum"] += results["grad_norm"] if math.isfinite(results["grad_norm"]) else 0.0
                self._window["clipped"] += int(self.args.clip > 0 and results["grad_norm"] > self.args.clip)
                self._window["imbalance"] += self.imbalance
                diag = results.pop("diag", None)
                row = {"step": self.curr_step, "loss": results["loss"], "lr": results["lr"]}
                logged: Dict[str, float] = {}
                if diag is not None:
                    logged.update(diag)
                if self.curr_step % self.args.log_every == 0:
                    logged.update(self._flush_window(results))  # collective: every rank
                if self.val_set is not None and (self.curr_step % self.args.val_every == 0
                                                 or self.curr_step == self.args.max_steps):
                    logged.update(self._validate())  # collective: every rank
                history.append(row)
                if self.master:
                    last = self.curr_step == self.args.max_steps
                    if last or self.curr_step % self.args.save_every == 0 or self.curr_step % self.args.save_perm_every == 0:
                        self.save_checkpoint()
                        logged["ckpt/saved"] = 1.0
                        logged["ckpt/permanent"] = float(self.curr_step % self.args.save_perm_every == 0 or last)
                    if logged:
                        row = {**row, **logged}
                        self.logger.log({"stage": self.args.stage, "step_in_stage": self.curr_step, **logged},
                                        self.offset + self.curr_step)
                    metrics.write(json.dumps(row) + "\n")
                    metrics.flush()
                    if self.curr_step % self.args.log_every == 0:
                        print(f"step {self.curr_step} loss {results['loss']:.5f} lr {results['lr']:.3e} "
                              f"gnorm {results['grad_norm']:.3f} {logged.get('perf/step_time', 0.0):.3f}s/step",
                              flush=True)
        finally:
            if metrics is not None:
                metrics.close()
            if self.logger is not None:
                self.logger.finish()
        return history

    # -------------------------------------------------------------------------------------------- logging window

    def _flush_window(self, results: Dict[str, float]) -> Dict[str, float]:
        """The logging window's metrics, summed / maxed over ranks (a collective: every rank calls it at the
        same steps), then a fresh window."""
        w = self._window
        steps = max(w["steps"], 1)
        sums = torch.tensor([w["tables"], w["cells"], w["rows"], w["train_rows"], w["features"], w["classes"],
                             w["data_time"], w["step_time"]], dtype=torch.float64, device=self.device)
        maxes = torch.tensor([w["max_logit"], w["max_seq"], w["max_features"], w["loss_max"], -w["loss_min"],
                              w["data_time"], self._mem("max_memory_allocated"), self._mem("memory_reserved")],
                             dtype=torch.float64, device=self.device)
        buckets = w["buckets"]
        if self.ddp:
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(buckets)
            torch.distributed.all_reduce(maxes, op=torch.distributed.ReduceOp.MAX)
        sums, maxes = sums.tolist(), maxes.tolist()
        tables, cells, rows, train_rows, features, classes, data_time, step_time = sums
        step_time /= self.world
        out: Dict[str, float] = {}
        b = buckets.tolist()
        cls_names, len_names = bucket_names()
        total = [sum(r[0] for r in b[len(CLASS_BUCKETS):]), sum(r[1] for r in b[len(CLASS_BUCKETS):]),
                 sum(r[2] for r in b[len(CLASS_BUCKETS):])]
        if total[2] > 0:
            out["train/loss"] = total[0] / total[2]  # all ranks, all tables of the window
            out["train/acc"] = total[1] / total[2]
        for name, r in zip(cls_names + len_names, b):
            if r[2] > 0:
                out[f"train/loss_{name}"] = r[0] / r[2]
                out[f"train/acc_{name}"] = r[1] / r[2]
                out[f"data/tables_{name}"] = r[2]
        out["train/loss_rank0_step"] = results["loss"]
        out["train/micro_loss_max"] = maxes[3]
        out["train/micro_loss_min"] = -maxes[4]
        if "train/loss" in out:
            x = out["train/loss"]
            if self.loss_ema is None:
                self.loss_ema = x
            spike = math.isfinite(x) and self.loss_var > 0 and x > self.loss_ema + SPIKE_SIGMAS * math.sqrt(self.loss_var)
            out["stability/loss_spike"] = float(spike)
            out["train/loss_ema"] = self.loss_ema
            if math.isfinite(x):
                d = x - self.loss_ema
                self.loss_ema += 0.05 * d
                self.loss_var = 0.95 * (self.loss_var + 0.05 * d * d)
        out["optim/lr"] = results["lr"]
        out["optim/grad_norm"] = results["grad_norm"]
        out["optim/grad_norm_mean"] = w["grad_norm_sum"] / steps
        out["optim/clip_frac"] = w["clipped"] / steps
        out["stability/skipped_steps"] = float(self.skipped_total)
        out["stability/nonfinite_loss_steps"] = float(self.nonfinite_total)
        out["stability/max_abs_logit"] = maxes[0]
        out["data/tables_per_step"] = tables / steps
        out["data/seq_len_mean"] = rows / max(tables, 1)
        out["data/train_rows_mean"] = train_rows / max(tables, 1)
        out["data/features_mean"] = features / max(tables, 1)
        out["data/classes_mean"] = classes / max(tables, 1)
        out["data/seq_len_max"] = maxes[1]
        out["data/features_max"] = maxes[2]
        out["perf/step_time"] = step_time / steps
        out["perf/data_wait_frac"] = data_time / max(self.world * w["step_time"], 1e-9) if w["step_time"] else 0.0
        out["perf/data_wait_max"] = maxes[5] / steps
        out["perf/tables_per_s"] = tables / max(step_time, 1e-9)
        out["perf/cells_per_s"] = cells / max(step_time, 1e-9)
        out["perf/rank_imbalance"] = w["imbalance"] / steps
        out["cell/m_mean"] = w["cell_m"] / steps  # rank 0's window (M is the same on every rank)
        out["cell/exact_micro_frac"] = w["cell_exact"] / max(w["micro"], 1)
        out["perf/cache_releases"] = float(self.cache_releases)
        if self.cuda:
            out["perf/gpu_max_allocated_gb"] = maxes[6] / 2**30
            out["perf/gpu_reserved_gb"] = maxes[7] / 2**30
            torch.cuda.reset_peak_memory_stats(self.device)
        self._window = self._new_window()
        return out

    def _mem(self, fn: str) -> float:
        return float(getattr(torch.cuda, fn)(self.device)) if self.cuda else 0.0

    # -------------------------------------------------------------------------------------------- held-out prior

    def _val_path(self) -> str:
        return os.path.join(self.args.out, "val_prior.pt")

    def _load_val_set(self):
        """The stage's fixed held-out prior tables: generated once by rank 0 from a seed no training stream
        uses (the global RNG states are restored afterwards, so the training stream is unchanged), saved to
        ``OUT/val_prior.pt`` and reloaded on resume, so every evaluation of a run sees the same tables."""
        path = self._val_path()
        if self.master and not os.path.exists(path):
            states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            try:
                seed(VAL_SEED + self.args.stage)
                random.seed(VAL_SEED + self.args.stage)
                kw = dict(STAGES[f"stage{self.args.stage}"], max_classes=MAX_CLASSES, config=STAGE_CONFIG)
                ds = PriorDataset(**kw)
                groups = []
                n_groups = self.args.val_groups if self.args.val_groups is not None else VAL_TABLE_GROUPS[self.args.stage]
                t0 = time.time()
                for _ in range(n_groups):
                    batch = ds.get_batch(batch_size=ds.batch_size_per_gp)
                    batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
                    groups.append(self._prepare(batch))
                self._log(f"held-out prior: {sum(g[0].shape[0] for g in groups)} tables in {time.time() - t0:.0f}s")
            finally:
                random.setstate(states[0])
                np.random.set_state(states[1])
                torch.set_rng_state(states[2])
            os.makedirs(self.args.out, exist_ok=True)
            torch.save(groups, path + ".tmp")
            os.replace(path + ".tmp", path)
        if self.ddp:
            torch.distributed.barrier()
        return torch.load(path, weights_only=True)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Loss / accuracy on the held-out prior tables, sharded over ranks (a collective). Train mode (the
        training forward, all ``max_classes`` logits, as the training loss) under no_grad."""
        m = self.raw_model
        acc = torch.zeros(self.n_buckets, 3, dtype=torch.float64, device=self.device)
        t0 = time.time()
        for i, (X, y, train_size) in enumerate(self.val_set):
            if i % self.world != self.rank:
                continue
            X, y = X.to(self.device), y.to(self.device)
            with self.amp_ctx(), self.sdpa_ctx():
                pred = m(X, y[:, :train_size]).float()
            B, T = y.shape
            lb = len(CLASS_BUCKETS) + len_bucket(T)
            y_test = y[:, train_size:].long()
            ce = F.cross_entropy(pred.flatten(end_dim=-2), y_test.flatten(), reduction="none").view(B, -1).mean(1)
            hit = (pred.argmax(-1) == y_test).float().mean(1)
            rows = torch.stack([ce.double(), hit.double(), torch.ones_like(ce, dtype=torch.float64)], 1)
            idx = torch.tensor([class_bucket(int(c)) for c in (y.long().amax(1) + 1).tolist()], device=self.device)
            acc.index_add_(0, idx, rows)
            acc[lb] += rows.sum(0)
        if self.ddp:
            torch.distributed.all_reduce(acc)
        b = acc.tolist()
        lens = b[len(CLASS_BUCKETS):]
        n = sum(r[2] for r in lens)
        out = {"val/loss": sum(r[0] for r in lens) / max(n, 1), "val/acc": sum(r[1] for r in lens) / max(n, 1),
               "val/tables": n, "val/time_s": time.time() - t0}
        cls_names, len_names = bucket_names()
        for name, r in zip(cls_names + len_names, b):
            if r[2] > 0:
                out[f"val/loss_{name}"] = r[0] / r[2]
                out[f"val/acc_{name}"] = r[1] / r[2]
        return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m comb.train.train", description=__doc__.split("\n")[0])
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--out", required=True, help="checkpoint + metrics directory; existing step-*.ckpt resume")
    p.add_argument("--init", default=None, help="checkpoint whose state_dict initialises the model (weights only)")
    p.add_argument("--max-steps", type=int, default=None, help="default: the stage's step count")
    p.add_argument("--batch-size", type=int, default=64, help="global batch (split across DDP ranks)")
    p.add_argument("--micro-batch-size", type=int, default=None, help="default: 4 (stage 1) / 1 (stages 2, 3)")
    p.add_argument("--lr", type=float, default=None, help="default: 8e-4 / 1e-4 / 2e-5")
    p.add_argument("--clip", type=float, default=None, help="gradient-norm clipping; default: 10 / 10 / 1")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers (upstream --n_jobs)")
    p.add_argument("--balance", type=int, default=1, help="DDP: even out the ranks' micro-batches (1) or not (0)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-every", type=int, default=None, help="temporary checkpoint period; default 500 / 50 / 50")
    p.add_argument("--save-perm-every", type=int, default=None, help="permanent checkpoint period; 5000 / 1000 / 1000")
    p.add_argument("--log-every", type=int, default=10, help="stdout + instrumentation window (steps)")
    p.add_argument("--diag-every", type=int, default=500, help="per-module and cell diagnostics period (0: off)")
    p.add_argument("--val-every", type=int, default=1000, help="held-out prior evaluation period (0: off)")
    p.add_argument("--val-groups", type=int, default=None, help="held-out prior groups; default 64 per stage")
    p.add_argument("--skip-nonfinite", type=int, default=1, help="skip the optimizer step on a non-finite grad norm")
    p.add_argument("--global-step-offset", type=int, default=None,
                   help="global step of this stage's step 0 (W&B x-axis); default 0 / 500000 / 540000")
    p.add_argument("--cell-m", type=int, default=1024, help="cell budget at validation and inference")
    p.add_argument("--cell-m-range", default="128,2048",
                   help="per-step cell budget, log-uniform in 'lo,hi' ('' = always --cell-m)")
    p.add_argument("--wandb-project", default="comb")
    p.add_argument("--wandb-run-id", default=None, help="fixed id: all stages and requeues append to one run")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None,
                   help="default: $WANDB_MODE, else offline (compute nodes have no internet)")
    p.add_argument("--wandb-dir", default=None, help="default: <parent of --out>/wandb")
    p.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)  # test-only small model
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for k, v in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    torch.multiprocessing.set_start_method("spawn", force=True)  # as upstream tabicl/train/__main__.py
    Trainer(args).run()


if __name__ == "__main__":
    main(sys.argv[1:])
