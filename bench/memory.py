"""Peak GPU memory of fit and predict at the defaults, on the full GPU and with the budget of a smaller one.

    python bench/memory.py --out bench/results/memory.csv [--cap-gb 8]

Synthetic tables (``make_classification``, 2 classes), 10,000 test rows. Peaks are ``torch.cuda.max_memory_allocated``
above what is allocated before the call (the loaded network, the fitted state for predict). ``--cap-gb`` simulates a
smaller GPU on the same card: the reported device memory (from which Comb sizes its passes) is set to the cap and the
process may allocate at most that much (``set_per_process_memory_fraction``).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.datasets import make_classification

GB = 2**30


class _Props:  # a device-properties view that reports a smaller GPU
    def __init__(self, props, total):
        self._props, self.total_memory = props, total

    def __getattr__(self, name):
        return getattr(self._props, name)


def cap(gb: float):
    real = torch.cuda.get_device_properties
    torch.cuda.set_per_process_memory_fraction(gb * GB / real(0).total_memory)
    torch.cuda.get_device_properties = lambda d=None: _Props(real(d), int(gb * GB))


def peak(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / GB, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--cap-gb", type=float, default=None)
    p.add_argument("--tables", default="300000x50,1000000x20", help="rows x features, comma separated")
    a = p.parse_args()
    from comb import CombClassifier

    if a.cap_gb:
        cap(a.cap_gb)
    rows = []
    for spec in a.tables.split(","):
        n, f = map(int, spec.split("x"))
        X, y = make_classification(n + 10_000, f, n_informative=min(f, 10), random_state=0)
        X, Xt, y = X.astype(np.float32)[:n], X.astype(np.float32)[n:], y[:n]
        clf = CombClassifier(device="cuda")
        clf.fit(X[:1000], y[:1000])  # warm-up: loads the network outside the measurement
        clf, fit_gb, fit_s = peak(lambda: CombClassifier(device="cuda").fit(X, y))
        _, pred_gb, pred_s = peak(lambda: clf.predict_proba(Xt))
        row = dict(gpu=torch.cuda.get_device_name(0), cap_gb=a.cap_gb or "", rows=n, features=f, cell_m=clf.cell_m_,
                   n_estimators=clf.n_estimators, fit_peak_gb=round(fit_gb, 2), fit_s=round(fit_s, 1),
                   predict_10k_peak_gb=round(pred_gb, 2), predict_s=round(pred_s, 2))
        print(row, flush=True)
        rows.append(row)
        del clf
        torch.cuda.empty_cache()
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
