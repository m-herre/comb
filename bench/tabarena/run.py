"""Comb vs LightPFN (vs TabICLv2) through TabArena's own pipeline: TabArena-Lite and BeyondArena core2k, classification.

    python bench/tabarena/run.py --out D --worker N --arms A,B       # fit (arm, suite, dataset) items from a shared queue
    python bench/tabarena/run.py --out D --compare --arms A,B        # one leaderboard per slice with every arm

Protocol: single-fit by default (one fit on each split's full training data, no bagging, so arms compare like for
like and fast); ``COMB_OFFICIAL=1`` runs TabArena's official validation protocol (8-fold bagging; BeyondArena's own
protocol on core2k), the one published entries use. ``--ref-dir`` adds a single-fit eval of the released TabICLv2.
Arms: ``lightpfn`` (TabArena's LightPFN model, default config) and parametric Comb arms
``comb-<weights>-<M>-e<members>-amp<auto|1>[-t<temperature>][-k<k-means steps>]`` with ``<M>`` one of ``m4096``,
``auto<lo>x<hi>`` (clip(N, lo, hi)) or ``sqrt<c>`` (clip(c sqrt(N), 4096, 16384)); ``<weights>`` names a folder in
``WEIGHTS`` below. Workers on any number of GPUs and nodes share the queue
through atomic claim files.

Needs the TabArena environment with ``comb`` and ``lightpfn`` importable, and ``TABARENA_CACHE`` set. Compare writes
``result.json``, ``slices.csv`` (Elo per slice and method) and ``paired_<suite>.csv`` (per-task error of every arm).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = {"lightpfn": "LightPFN"}
# Parametric Comb arms for the speed / quality sweep: comb-<weights>-m<M | auto<lo>x<hi>>-e<members>-amp<auto|1>,
# e.g. comb-s4-m4096-e4-amp1 or comb-s4-auto4096x16384-e8-ampauto. ``s4`` is the released weights (COMB_CHECKPOINT
# or the Hub); other weights folders are named by COMB_WEIGHTS_<NAME> (e.g. COMB_WEIGHTS_MINE=/path -> comb-mine-...).
WEIGHTS = {"s4": None, **{k[len("COMB_WEIGHTS_"):].lower(): v for k, v in os.environ.items() if k.startswith("COMB_WEIGHTS_")}}
ARM_RE = re.compile(r"comb-(?P<w>[a-z0-9]+)-(?:m(?P<m>\d+)|auto(?P<lo>\d+)x(?P<hi>\d+)|sqrt(?P<c>\d+))-e(?P<e>\d+)-amp(?P<amp>auto|1)"
                    r"(?:-t(?P<t>[0-9.]+))?(?:-k(?P<k>\d+))?$")  # optional softmax temperature, k-means iterations


def config_of(arm):
    """CombClassifier arguments of a parametric arm (``cell_m_lo`` / ``cell_m_hi`` are resolved per table in ``comb_model``)."""
    g = ARM_RE.match(arm)
    if g is None:
        raise ValueError(f"unknown arm {arm!r}")
    cfg = {"checkpoint": WEIGHTS[g["w"]], "n_estimators": int(g["e"]), "use_amp": True if g["amp"] == "1" else "auto"}
    if g["t"]:
        cfg["softmax_temperature"] = float(g["t"])
    if g["k"]:
        cfg["kmeans_iters"] = int(g["k"])
    if g["m"]:
        cfg["cell_m"] = int(g["m"])
    elif g["c"]:  # M = clip(c * sqrt(N), 4096, 16384)
        cfg["cell_m_sqrt"], cfg["cell_m_lo"], cfg["cell_m_hi"] = int(g["c"]), 4096, 16384
    else:
        cfg["cell_m_lo"], cfg["cell_m_hi"] = int(g["lo"]), int(g["hi"])
    return cfg


def label_of(arm):
    return ARMS.get(arm) or arm
SINGLE = " (single-fit)"
OFFICIAL = os.environ.get("COMB_OFFICIAL") == "1"  # 8-fold bagging under the official validation protocol
if OFFICIAL:
    SINGLE = " (official)"
REF_NAME = "TabICLv2" + SINGLE
NEW_PREFIX = "[New] "
SUITES = {"ta": ["lite", "classification"], "ba": ["core2k", "lite", "classification"]}
SLICES = {"ta": [("all", None)],
          "ba": [("all", None), ("task_iid", "iid"), ("task_grouped", "grouped"), ("task_temporal", "temporal"),
                 ("scale_tiny", "tiny"), ("scale_small", "small"), ("scale_medium", "medium"),
                 ("scale_large", "large"), ("dim_low", "low-dim"), ("dim_high", "high-dim"),
                 ("feat_text", "text"), ("feat_highcard", "high-cardinality")]}
PUBLISHED = ("TABICLV2 (default)", "LIGHTPFN (default)")  # the bagged leaderboard entries, where in the pool


def gen_of(arm):
    if arm == "lightpfn":
        from tabarena.models.lightpfn.hpo import gen_lightpfn

        return gen_lightpfn
    from comb_model import gen

    return gen(**config_of(arm))


def context_of(suite):
    from tabarena.contexts import BeyondArenaContext, TabArenaContext

    return TabArenaContext() if suite == "ta" else BeyondArenaContext()


def run(context, arm, suite, out, datasets):
    from tabarena.benchmark.experiment import BeyondArenaExperimentBundle, TabArenaV0pt1ExperimentBundle

    bundle = TabArenaV0pt1ExperimentBundle if suite == "ta" else BeyondArenaExperimentBundle
    kw = dict(models=[(gen_of(arm), 0)], model_verbosity=2, outer_experiments=not OFFICIAL)
    if os.environ.get("TABARENA_MODEL_ARTIFACTS_BASE_PATH"):
        kw["model_artifacts_base_path"] = os.environ["TABARENA_MODEL_ARTIFACTS_BASE_PATH"]
    return context.build_and_run_jobs(bundle(**kw).build_experiments(), expname=str(out / arm / f"experiments_{suite}"),
                                      subset=SUITES[suite], build_kwargs={"dataset_names": datasets},
                                      new_result_prefix=NEW_PREFIX, debug_mode=True)


def datasets_of(context, suite):
    """The suite's datasets, largest first (split 0 is the lite split)."""
    tm = context.task_metadata
    tm = tm.assign(split=0) if "split" not in tm.columns else tm[tm["split"] == 0]
    mask = None
    for name in SUITES[suite]:
        m = context.subset_predicates[name].predicate(tm)
        mask = m if mask is None else mask & m
    sub = tm[mask].drop_duplicates("dataset")
    name = "dataset_name" if "dataset_name" in sub.columns else "dataset"
    return list(sub.assign(_s=sub["num_instances"] * sub["n_features"]).sort_values("_s", ascending=False)[name])


def worker(arms, suites, out, wid, only=None):
    """Claims items largest first, arms interleaved, so the huge tables start early on every GPU."""
    queue = [(arm, suite, d) for suite in suites for d in datasets_of(context_of(suite), suite) for arm in arms
             if only is None or d in only]
    (out / "claims").mkdir(parents=True, exist_ok=True)
    for arm, suite, name in queue:
        path = out / arm / f"status_{suite}" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(out / "claims" / f"{arm}__{suite}__{name}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            run(context_of(suite), arm, suite, out, [name])
            status = {"ok": True}
        except Exception as e:  # noqa: BLE001 -- a failed fit costs only its item
            traceback.print_exc()
            status = {"ok": False, "error": repr(e)[:2000]}
        status.update(time_s=time.time() - t0, worker=wid, host=os.uname().nodename)
        path.write_text(json.dumps(status))
        print(f"[{arm}/{suite}] {name}: {'ok' if status['ok'] else 'FAILED'} in {status['time_s']:.0f}s", flush=True)
        gc.collect()
        import torch

        torch.cuda.empty_cache()


def results_of(arm, suite, out):
    """The arm's non-imputed per-task results under its display name, and its failed datasets."""
    context = context_of(suite)
    ok, failed = [], {}
    for name in datasets_of(context, suite):
        p = out / arm / f"status_{suite}" / f"{name}.json"
        st = json.loads(p.read_text()) if p.exists() else {"ok": False, "error": "not run"}
        (ok.append(name) if st["ok"] else failed.__setitem__(name, st.get("error")))
    if not ok:
        return None, failed
    run(context, arm, suite, out, ok)  # cached results only: registers them
    _, res = context.leaderboard(subset=SUITES[suite], return_results=True)
    res = res[res["method"].astype(str).str.startswith(NEW_PREFIX)].copy()
    if "imputed" in res.columns:
        res = res[~res["imputed"].fillna(False).astype(bool)]
    return rename(res, label_of(arm) + SINGLE), failed


def rename(res, name):
    for col in ("method", "ta_name", "ta_suite"):
        if col in res.columns:
            res[col] = name
    if OFFICIAL:  # bagged runs come back as a tuning config; rank them as one fixed method, as single-fit rows are
        res["method_type"] = "baseline"
        for col in ("method_subtype", "config_type"):
            if col in res.columns:
                res[col] = None
    return res


def compare(arms, suites, out, ref_dir, listed_only=False):
    import pandas as pd

    from listing import NOT_LISTED

    result, rows = {}, []
    for suite in suites:
        parts, failed = [], {}
        for arm in arms:
            res, failed[arm] = results_of(arm, suite, out)
            if res is not None:
                parts.append(res)
        if ref_dir is not None and (ref_dir / f"new_results_{suite}.csv").exists():
            parts.append(rename(pd.read_csv(ref_dir / f"new_results_{suite}.csv"), REF_NAME))
        new = pd.concat(parts, ignore_index=True)
        new.to_csv(out / f"paired_{suite}.csv", index=False)
        ours = list(new["method"].unique())
        context = context_of(suite)
        base = context.load_results(download_results="auto")
        if listed_only:  # score against the public leaderboard's entries only (listing.py)
            base = base[~base["method"].isin(NOT_LISTED)]
        result[suite] = {"failed": failed}
        (out / "leaderboards").mkdir(exist_ok=True)
        for key, pred in SLICES[suite]:
            subset = SUITES[suite] + ([pred] if pred else [])
            try:
                lb, res = context.leaderboard(subset=subset, new_results=new, ta_results=base, return_results=True)
            except Exception as e:  # noqa: BLE001 -- one bad slice must not lose the others
                result[suite][key] = {"error": repr(e)}
                continue
            lb = lb.reset_index() if "method" not in lb.columns else lb
            lb.to_csv(out / "leaderboards" / f"{suite}_{key}.csv", index=False)
            imputed = res[res["imputed"].fillna(False).astype(bool)] if "imputed" in res.columns else res.iloc[:0]
            n_tasks = int(res[res["method"] == ours[0]]["dataset"].nunique())
            for _, r in lb[lb["method"].isin(ours + list(PUBLISHED))].iterrows():
                row = {"suite": suite, "slice": key, "method": r["method"], "elo": float(r["elo"]),
                       "elo_plus": float(r["elo+"]), "elo_minus": float(r["elo-"]), "rank": float(r["rank"]),
                       "n_tasks": n_tasks, "n_methods": int(len(lb)),
                       "imputed": int((imputed["method"] == r["method"]).sum())}
                rows.append(row)
                result[suite].setdefault(key, {})[r["method"]] = row
            print(suite, key, {m: v["elo"] for m, v in result[suite][key].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "slices.csv", index=False)
    (out / "result.json").write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--suites", default="ta,ba")
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ref-dir", type=Path, default=None)
    ap.add_argument("--listed-only", action="store_true", help="leave out entries not on the public leaderboard")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (a smoke test)")
    args = ap.parse_args()
    if not os.environ.get("TABARENA_CACHE"):
        raise RuntimeError("set TABARENA_CACHE")
    arms, suites = args.arms.split(","), args.suites.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        worker(arms, suites, args.out, args.worker, args.datasets and args.datasets.split(","))
    if args.compare:
        compare(arms, suites, args.out, args.ref_dir, args.listed_only)


if __name__ == "__main__":
    main()
