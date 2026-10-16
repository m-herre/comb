"""Fit and predict time per 1,000 rows: Comb vs LightPFN vs TabICLv2, one GPU, the same tables.

    python bench/timing.py --out bench/results/timing.csv [--sizes 1000,5000,20000,100000] [--repeats 3]

Tables are synthetic (``make_classification``, 20 features, 2 classes, 1,000 test rows). Each (model, size) is
warmed up once, then timed ``--repeats`` times with CUDA synchronised; the median is reported. LightPFN
(``lightpfn``) and TabICLv2 (``tabicl``) run when their packages are installed.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


def models():
    from comb import CombClassifier

    out = {
        "Comb (8 members)": lambda: CombClassifier(n_estimators=8, device="cuda"),
        "Comb (4 members)": lambda: CombClassifier(n_estimators=4, device="cuda"),
        "Comb (2 members)": lambda: CombClassifier(n_estimators=2, device="cuda"),
    }
    try:
        from lightpfn import LightPFNClassifier

        out["LightPFN (4 members)"] = lambda: LightPFNClassifier(n_estimators=4, device="cuda", random_state=0)
    except ImportError:
        print("lightpfn not installed: skipped")
    try:
        from tabicl import TabICLClassifier

        out["TabICLv2 (8 members)"] = lambda: TabICLClassifier(n_estimators=8, device="cuda")
    except ImportError:
        print("tabicl not installed: skipped")
    return out


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="bench/results/timing.csv")
    p.add_argument("--sizes", default="1000,5000,20000,100000")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-test", type=int, default=1000)
    args = p.parse_args()
    rows = []
    for n in (int(s) for s in args.sizes.split(",")):
        X, y = make_classification(n + args.n_test, 20, n_informative=10, random_state=n)
        X, Xt, y, yt = X[:n], X[n:], y[:n], y[n:]
        for name, make in models().items():
            make().fit(X[:200], y[:200]).predict_proba(Xt[:50])  # warm-up: weights, kernels, allocator
            fits, preds = [], []
            for _ in range(args.repeats):
                t0 = sync()
                clf = make().fit(X, y)
                t1 = sync()
                proba = clf.predict_proba(Xt)
                t2 = sync()
                fits.append(t1 - t0)
                preds.append(t2 - t1)
            row = dict(model=name, n_train=n, n_test=args.n_test, fit_s=np.median(fits), predict_s=np.median(preds),
                       fit_s_per_1k=np.median(fits) / n * 1000, predict_s_per_1k=np.median(preds) / args.n_test * 1000,
                       auc=roc_auc_score(yt, proba[:, 1]), gpu=torch.cuda.get_device_name())
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            rows.append(row)
            del clf
            torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
