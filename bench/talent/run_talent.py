"""TALENT held-out read (pre-registered: bench/talent/PREREGISTRATION.md).

    TALENT_DATA=... python bench/talent/run_talent.py --model comb_s4 --out OUT [--shard 0/4]

Tasks: the frozen manifest (bench/talent/manifest.csv, ``include == True``, 106 classification tasks), its sha256
and every task's y_test hash checked before any fit. Protocol: TALENT's fixed splits; context = train + val rows,
evaluation = test rows; one fit per task; every model through its own sklearn wrapper. The same preprocessing for
all: TALENT's N_ / C_ arrays concatenated, categoricals ordinal-encoded on the context rows (unseen / missing -> -1),
numeric NaN left to the model. Labels are encoded on the context rows; test rows of a class absent from the context
are dropped (counted in ``n_test_dropped``).

Every Comb setting is written out below (not left to the package defaults, which changed after the read), and the
weights are checked against their sha256 before the first fit. ``comb_s4`` is the pre-registered read (the released
weights at the defaults of the time: 8 members, fp32, 8 k-means steps); ``posthoc_comb_s4_defaults`` (today's
defaults) came after it and is not part of the pre-registration. Data: TALENT's dataset-250115 release, unpacked to
``TALENT_DATA``.

Writes one JSON line per task to OUT/<model>.jsonl (resumable: tasks already written are skipped).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path(__file__).with_name("manifest.csv")
MANIFEST_SHA = "71121b30d1ebbdad59be7f2ff2aa6078b35f4d7a7fc91ac274e4292b7bf8d9dc"
DATA = os.environ.get("TALENT_DATA", os.path.expanduser("~/.cache/talent/v250115/data"))  # dataset-250115.zip
WEIGHTS_SHA = "d00355574c90c6c9f93b4b1f5221e21e7c27d70ca3c8707e07b69992a9791573"  # released model.safetensors
READ = dict(n_estimators=8, use_amp=False, kmeans_iters=8, softmax_temperature=0.9, random_state=42, batch_size=8)
DEFAULTS = dict(n_estimators=4, use_amp="auto", kmeans_iters=4, softmax_temperature=0.9, random_state=42, batch_size=8)
COMB = {  # model name -> CombClassifier settings (cell_m None = the weights' config, 4096)
    "comb_s4": dict(READ, cell_m=None),
    "posthoc_comb_s4_defaults": dict(DEFAULTS, cell_m=None),  # the released defaults (fp16 on CUDA)
}
TABICL_CKPT = "tabicl-classifier-v2-20260212.ckpt"


@functools.lru_cache(maxsize=None)
def weights() -> str:
    """The released weights folder (COMB_CHECKPOINT or the Hub), checked against WEIGHTS_SHA."""
    from comb.checkpoint import resolve

    path = resolve(None)
    got = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    if got != WEIGHTS_SHA:
        raise SystemExit(f"{path}: model.safetensors sha256 {got[:12]} is not {WEIGHTS_SHA[:12]}")
    return str(path)


def tasks() -> pd.DataFrame:
    raw = Path(MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise SystemExit("manifest sha256 differs from the pre-registration")
    m = pd.read_csv(MANIFEST)
    m = m[m["include"] == True].reset_index(drop=True)  # noqa: E712
    if len(m) != 106:
        raise SystemExit(f"expected 106 included tasks, got {len(m)}")
    return m


def load(name: str, sha_y_test: str):
    d = Path(DATA) / name
    part = {}
    for s in ("train", "val", "test"):
        cols = []
        for kind in ("N", "C"):
            f = d / f"{kind}_{s}.npy"
            if f.exists():
                a = np.load(f, allow_pickle=True)
                cols.append((kind, a.reshape(len(a), -1)))
        part[s] = (cols, np.load(d / f"y_{s}.npy", allow_pickle=True))
    yt = part["test"][1]
    if hashlib.sha256((d / "y_test.npy").read_bytes()).hexdigest() != sha_y_test:  # the manifest hashes the file
        raise SystemExit(f"{name}: y_test hash differs from the manifest")
    ctx = {k: np.concatenate([dict(part["train"][0])[k], dict(part["val"][0])[k]]) for k, _ in part["train"][0]}
    tst = dict(part["test"][0])
    y_ctx = np.concatenate([part["train"][1], part["val"][1]])
    Xc, Xt = [], []
    if "N" in ctx:
        Xc.append(ctx["N"].astype(np.float64)); Xt.append(tst["N"].astype(np.float64))
    if "C" in ctx:
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1,
                             dtype=np.float64)
        Xc.append(enc.fit_transform(ctx["C"].astype(str))); Xt.append(enc.transform(tst["C"].astype(str)))
    Xc, Xt = np.concatenate(Xc, axis=1), np.concatenate(Xt, axis=1)
    classes = np.unique(y_ctx.astype(str))
    lut = {c: i for i, c in enumerate(classes)}
    yc = np.array([lut[v] for v in y_ctx.astype(str)])
    yt_enc = np.array([lut.get(v, -1) for v in yt.astype(str)])
    keep = yt_enc >= 0
    return Xc, yc, Xt[keep], yt_enc[keep], int((~keep).sum()), len(classes)


def make(model: str, seed: int = 0):
    if model in COMB:
        from comb import CombClassifier
        return CombClassifier(checkpoint=weights(), **COMB[model])
    if model == "tabiclv2":
        from tabicl import TabICLClassifier
        return TabICLClassifier(checkpoint_version=TABICL_CKPT, use_amp=False)
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=16)
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=16)
    raise SystemExit(f"unknown model {model}")


def score(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score
    p = np.asarray(p, dtype=np.float64)
    out = {"acc": float(accuracy_score(y, p.argmax(1)))}
    present = np.unique(y)
    if len(present) < 2:
        out["auc"] = float("nan")
    elif k == 2:
        out["auc"] = float(roc_auc_score(y, p[:, 1]))
    else:  # one-vs-rest macro over the classes present in the test rows
        pp = p[:, present] / np.clip(p[:, present].sum(1, keepdims=True), 1e-12, None)
        out["auc"] = float(roc_auc_score(y, pp, multi_class="ovr", average="macro", labels=present))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*COMB, "tabiclv2", "xgboost", "catboost"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th task from i")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.model}.jsonl"
    done = {json.loads(l)["task"] for l in f.read_text().splitlines()} if f.exists() else set()
    m = tasks()
    if a.model in COMB:
        weights()  # fail before any task
    for j, r in m.iterrows():
        if j % n != i or r.talent_name in done:
            continue
        Xc, yc, Xt, yt, dropped, k = load(r.talent_name, r.sha256_y_test)
        clf = make(a.model)
        t0 = time.time(); clf.fit(Xc, yc); t1 = time.time(); p = clf.predict_proba(Xt); t2 = time.time()
        if p.shape[1] != k:  # a wrapper that saw fewer classes (cannot happen with context-encoded labels)
            raise SystemExit(f"{r.talent_name}: {p.shape[1]} probability columns for {k} classes")
        rec = dict(task=r.talent_name, model=a.model, n_ctx=len(yc), n_test=len(yt), n_test_dropped=dropped,
                   n_features=Xc.shape[1], n_classes=k, fit_s=t1 - t0, predict_s=t2 - t1, **score(yt, p, k))
        if a.model in COMB:
            rec["settings"] = COMB[a.model]
        with open(f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
