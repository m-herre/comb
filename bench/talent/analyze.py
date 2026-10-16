"""Pre-registered analysis of the TALENT read (bench/talent/PREREGISTRATION.md).

    python bench/talent/analyze.py bench/results/talent

Primary: mean over tasks of the paired AUC delta (comb_s4 - tabiclv2), 95 % bootstrap CI over tasks (10 000
resamples, seed 0). Secondary: accuracy delta, win / loss count, size slices (< 5k, 5k-50k, > 50k context rows).
Pass rules: TabICLv2-level = CI lower bound > -0.25 pp; beats TabICLv2 = lower bound > 0.

Post-hoc (not pre-registered, written to posthoc.csv when the run exists): today's defaults (4 members, fp16)
against the read's configuration and TabICLv2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["comb_s4", "tabiclv2", "xgboost", "catboost"]
POSTHOC = [("posthoc_comb_s4_defaults", "comb_s4"), ("posthoc_comb_s4_defaults", "tabiclv2")]


def boot(d: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return d.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def paired(df: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    x = df.pivot(index="task", columns="model", values=metric)[[a, b]].dropna()
    d = (x[a] - x[b]).to_numpy() * 100  # pp
    m, lo, hi = boot(d)
    return dict(pair=f"{a} - {b}", metric=metric, n=len(d), mean_pp=m, ci_lo=lo, ci_hi=hi,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()))


def main():
    out = Path(sys.argv[1])
    df = pd.concat([pd.read_json(out / f"{m}.jsonl", lines=True) for m in MODELS if (out / f"{m}.jsonl").exists()])
    counts = df.groupby("model").task.nunique()
    print("tasks per model:", counts.to_dict())
    rows = [paired(df, "comb_s4", "tabiclv2", "auc"), paired(df, "comb_s4", "tabiclv2", "acc")]
    for g in ("xgboost", "catboost"):
        rows.append(paired(df, "comb_s4", g, "auc"))
    size = df[df.model == "tabiclv2"].set_index("task").n_ctx
    for lab, lo, hi in (("< 5k", 0, 5000), ("5k-50k", 5000, 50000), ("> 50k", 50000, 10**9)):
        tasks = size[(size >= lo) & (size < hi)].index
        r = paired(df[df.task.isin(tasks)], "comb_s4", "tabiclv2", "auc"); r["pair"] += f" [{lab}]"; rows.append(r)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    prim = res.iloc[0]
    print(f"\nTabICLv2-level (lo > -0.25 pp): {prim.ci_lo > -0.25}   beats TabICLv2 (lo > 0): {prim.ci_lo > 0}")
    print("mean AUC / acc:", df.groupby("model")[["auc", "acc"]].mean().round(4).to_dict())
    res.to_csv(out / "summary.csv", index=False)
    extra = [m for m in {m for pair in POSTHOC for m in pair} - set(MODELS) if (out / f"{m}.jsonl").exists()]
    if extra:
        df = pd.concat([df, *[pd.read_json(out / f"{m}.jsonl", lines=True) for m in extra]])
        post = pd.DataFrame([paired(df, a, b, metric) for a, b in POSTHOC for metric in ("auc", "acc")
                             if {a, b} <= set(df.model)])
        print("\npost-hoc (not pre-registered):")
        print(post.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("mean AUC:", df[df.model.isin(extra)].groupby("model").auc.mean().round(4).to_dict())
        post.to_csv(out / "posthoc.csv", index=False)


if __name__ == "__main__":
    main()
