"""Elo vs train and inference time on TabArena-Lite and BeyondArena (core2k), with the Pareto front.

    python bench/tabarena/plot_pareto.py      # writes docs/assets/pareto_{tabarena,beyondarena}{,-dark}.svg

Reads the leaderboards in bench/results/tabarena/leaderboard_*.csv: TabArena's published entries plus our runs of
Comb and LightPFN, all under the official protocol (``run.py --compare`` on sweepO3). Entries that are not on the
public leaderboards (hosted APIs under review, unverified methods) are left out: listing.py. Times are TabArena's medians
per 1,000 rows. Needs matplotlib and pandas.
"""
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from listing import NOT_LISTED  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "bench/results/tabarena", ROOT / "docs/assets"
SUITES = {"tabarena": ("leaderboard_tabarena_lite.csv", "TabArena-Lite (38 classification datasets)"),
          "beyondarena": ("leaderboard_beyondarena_core2k.csv", "BeyondArena core2k (98 datasets)")}
COMB = {"comb-s4-m4096-e4-amp1-k4 (official)": ("Comb (defaults)", "o", 70),  # legend name, marker, size
        "comb-s4-m4096-e2-amp1 (official)": ("Comb, n_estimators=2", "v", 55),
        "comb-s4-m16384-e4-amp1 (official)": ("Comb, cell_m=16384", "D", 42)}
DROP = {"comb-s4-m4096-e4-amp1 (official)", "LIGHTPFN (default)"}  # 8-step k-means twin; published twin of our run
LABEL = {"LightPFN (official)": "LightPFN", "TABICLV2 (default)": "TabICLv2", "TA-TABICLv2 (default)": "TabICLv2",
         "TA-KUMO-TABULAR-SMALL (default)": "Kumo-Tabular-Small", "TA-TABPFN-3.5 (default)": "TabPFN-3.5",
         "TABDPT-1.3 (default)": "TabDPT-1.3", "TA-TABDPT-1.3 (default)": "TabDPT-1.3",
         "comb-s4-m4096-e4-amp1-k4 (official)": "Comb"}
# label offsets in points (dx, dy) and alignment; default: right of the point
PLACE = {"Comb": (-9, 6, "right"), "LightPFN": (7, -11, "left"), "TabDPT-1.3": (7, -4, "left"), "Kumo-Tabular-Small": (7, -10, "left"), "TabPFN-3.5": (-7, 4, "right")}
AXES = (("median_time_train_s_per_1K", "train time per 1,000 rows (s)"),
        ("median_time_infer_s_per_1K", "inference time per 1,000 rows (s)"))
THEMES = {"": dict(bg="#ffffff", fg="#1c2333", grid="#e3e5ea", other="#b8bcc6", front="#1c2333", comb="#e8930c"),
          "-dark": dict(bg="#0d1117", fg="#e6edf3", grid="#262c36", other="#5b6270", front="#e6edf3", comb="#f2a516")}


def front(df, col):
    """Pareto front (lower time, higher Elo) as the rows on it, by ascending time."""
    rows, best = [], -float("inf")
    for _, r in df.sort_values([col, "elo"], ascending=[True, False]).iterrows():
        if r["elo"] > best:
            rows.append(r)
            best = r["elo"]
    return pd.DataFrame(rows)


def plot(key, theme, c):
    file, title = SUITES[key]
    df = pd.read_csv(RES / file)
    df = df[~df["method"].isin(DROP | NOT_LISTED)].reset_index(drop=True)
    is_comb = df["method"].isin(COMB)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=c["bg"])
    for ax, (col, xlabel) in zip(axes, AXES):
        ax.set_facecolor(c["bg"])
        f = front(df, col)
        lo, hi = np.log10(df[col].min()), np.log10(df[col].max())
        x0, x1 = 10 ** (lo - 0.06 * (hi - lo)), 10 ** (hi + 0.06 * (hi - lo))
        ax.set_xlim(x0, x1)
        ax.step([x0, *f[col], x1], [f["elo"].iloc[0], *f["elo"], f["elo"].iloc[-1]], where="post", color=c["front"],
                lw=1.3, ls=":", alpha=0.8, zorder=2, label="Pareto front")
        on = df["method"].isin(set(f["method"]))
        ax.scatter(df.loc[~is_comb, col], df.loc[~is_comb, "elo"], s=18, color=c["other"], zorder=3, lw=0)
        ax.scatter(df.loc[~is_comb & on, col], df.loc[~is_comb & on, "elo"], s=22, color=c["fg"], zorder=4, lw=0)
        for m, (name, marker, size) in COMB.items():
            r = df[df["method"] == m]
            ax.scatter(r[col], r["elo"], s=size, marker=marker, color=c["comb"], zorder=5, edgecolor=c["bg"], lw=0.8,
                       label=name)
        for _, r in df.iterrows():
            name = LABEL.get(r["method"])
            if not name:
                continue
            comb = r["method"] in COMB
            dx, dy, ha = PLACE.get(name, (6, 3, "left"))
            ax.annotate(name, (r[col], r["elo"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha=ha,
                        color=c["comb"] if comb else c["fg"], fontweight="bold" if comb else "normal",
                        zorder=6)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=c["fg"])
        ax.grid(True, color=c["grid"], lw=0.6, zorder=0)
        ax.tick_params(colors=c["fg"], labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(c["grid"])
    axes[0].set_ylabel("Elo", color=c["fg"])
    leg = axes[1].legend(loc="lower right", fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(c["fg"])
    fig.suptitle(f"{title}, official protocol: higher and further left is better", color=c["fg"], fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"pareto_{key}{theme}.svg", facecolor=c["bg"], metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    for key in SUITES:
        for theme, colors in THEMES.items():
            plot(key, theme, colors)
