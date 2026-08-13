"""Regenerate the five paper figures as PDF and PNG."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PANEL, RAW, FIGS

PARTS = ["Client", "FII", "DII", "Pro"]
COLORS = {"Client": "#c1440e", "FII": "#1f4e79", "DII": "#4a7c59", "Pro": "#7d6608"}
NAVY = "#16324f"
LIGHT = "#c3cfdd"
RED = "#b03a2e"

SHARES_SMOOTH_WINDOW = 20
SHOCK_BASELINE_OFFSET = -1

K3 = 3 * np.sqrt(2)
K6 = 6 * np.sqrt(2)

BOUND_MPOR = pd.Timestamp("2019-01-21")
BOUND_6S = pd.Timestamp("2020-06-01")
BOUND_2024 = pd.Timestamp("2024-11-20")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
})


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS}/{name}.pdf")


#figures

def fig_multiplier(d):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fb = d[d.vol_bound == False]
    vb = d[d.vol_bound == True]
    ax.scatter(fb.date, fb.k, s=2.5, color=LIGHT, label="Floor-bound")
    ax.scatter(vb.date, vb.k, s=2.5, color=NAVY, label="Volatility-bound")

    segs = [
        (d.date.min(), BOUND_MPOR, K3, "--", "Pre-MPOR\n$k$ = 4.243", 3.35),
        (BOUND_MPOR, BOUND_6S, K3, "-", "MPOR\n$k$ = 4.243", 3.35),
        (BOUND_6S, BOUND_2024, K6, "-", "6-sigma\n$k$ = 8.485", 7.55),
        (BOUND_2024, d.date.max(), K6, "-", "Post-2024\n$k$ = 8.485", 7.55),
    ]
    for lo, hi, k, ls, lab, ty in segs:
        ax.plot([lo, hi], [k, k], color=RED, ls=ls, lw=2.2, zorder=5)
        mid = lo + (hi - lo) / 2
        ax.text(mid, ty, lab, color=RED, ha="center", va="top", fontsize=10.5)
    for b in (BOUND_MPOR, BOUND_6S, BOUND_2024):
        ax.axvline(b, color="grey", ls="--", lw=1.0, alpha=0.8)

    ax.set_ylabel("Implied multiplier $k$ = PSR/$\\sigma$")
    ax.set_ylim(2.4, 16.8)
    ax.legend(loc="upper left", frameon=False, fontsize=11, markerscale=5)
    fig.tight_layout()
    save(fig, "multiplier")


def fig_volbound_by_year(d):
    cc = d.dropna(subset=["k"])
    yr = cc.groupby(cc.date.dt.year)["vol_bound"].mean() * 100
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(yr.index, yr.values, width=0.62, color=NAVY)
    for x, v in yr.items():
        if v > 0:
            ax.text(x, v + 2.0, f"{v:.1f}", ha="center", fontsize=10.5, color=NAVY)
    # bars sit on integer years, so a date maps to year - 0.5 + fraction
    for dstr, lab in (("2019-01-21", "Jan 2019"), ("2020-06-01", "Jun 2020"),
                      ("2024-11-20", "Nov 2024")):
        ts = pd.Timestamp(dstr)
        xpos = ts.year - 0.5 + (ts.dayofyear / (366 if ts.is_leap_year else 365))
        ax.axvline(xpos, color="grey", ls="--", lw=1.2, alpha=0.8)
        ax.text(xpos + 0.07, 113, lab, fontsize=10.5, color="#555555")
    ax.set_ylabel("Volatility-bound days (%)")
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(yr.index)
    fig.tight_layout()
    save(fig, "volbound_by_year")


def fig_discretionary_floor(d):
    s = d.set_index("date")["psr_spot_nifty"].dropna()
    mmin = s.resample("MS").min().reindex(
        pd.date_range(s.index.min().replace(day=1), s.index.max(), freq="MS"))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 9.4))

    ax1.plot(s.index, s.values, lw=0.7, color=LIGHT, label="Daily price scan range")
    ax1.step(mmin.index, mmin.values, where="post", lw=2.2, color=NAVY,
             label="Monthly minimum PSR")
    ax1.axvspan(pd.Timestamp("2018-09-01"), pd.Timestamp("2019-01-01"),
                color="#e8c4ae", alpha=0.45, zorder=0)
    ax1.axvline(BOUND_MPOR, color="#555555", ls="--", lw=1.4)
    ax1.text(pd.Timestamp("2017-06-01"), 13.4, "floor raised\nSep--Dec 2018",
             color=RED, fontsize=11, ha="center")
    ax1.set_ylabel("NIFTY price scan range (% of index close)")
    ax1.set_title("A. Full sample", loc="left", color="#444444")
    ax1.legend(loc="upper left", frameon=False, fontsize=10.5)

    w = s.loc["2018-06-01":"2019-04-30"]
    ax2.plot(w.index, w.values, lw=1.2, color=NAVY, marker=".", ms=2.5)
    for step in ("2018-09-21", "2018-09-28", "2018-10-26", "2018-11-30"):
        ax2.axvline(pd.Timestamp(step), color=RED, ls="--", lw=1.3)
    ax2.axvline(BOUND_MPOR, color="#555555", ls="-", lw=1.4)
    ax2.set_ylabel("NIFTY price scan range (% of index close)")
    ax2.set_title("B. June 2018 to April 2019", loc="left", color="#444444")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.tight_layout()
    save(fig, "discretionary_floor")


def fig_participant_shares(l):
    wide = l.pivot(index="date", columns="client_type", values="share")[PARTS] * 100
    wide = wide.rolling(SHARES_SMOOTH_WINDOW, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    for p in PARTS:
        ax.plot(wide.index, wide[p], lw=1.3, label=p, color=COLORS[p])
    ax.set_ylabel("Share of gross index open interest (%)")
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=11)
    ax.set_ylim(-2, 78)
    fig.tight_layout()
    save(fig, "participant_shares")


def fig_shock_windows():
    valid = set(pd.read_csv(PANEL / "long.csv", parse_dates=["date"])["date"])
    oi = pd.read_csv(RAW / "participant_oi.csv", low_memory=False)
    oi["date"] = pd.to_datetime(oi["date"])
    oi["client_type"] = oi["client_type"].astype(str).str.strip()
    oi = oi[oi.client_type.isin(PARTS) & oi.date.isin(valid)]
    oi["fut_gross"] = oi["Future Index Long"] + oi["Future Index Short"]
    wide = np.log(oi.pivot_table(index="date", columns="client_type",
                                 values="fut_gross", aggfunc="first")[PARTS]
                  .sort_index().where(lambda x: x > 0))
    shocks = [("2018-09-21", "A. First 2018 floor increase"),
              ("2019-01-21", "B. January 2019 premium withdrawal"),
              ("2020-06-01", "C. June 2020 methodology change")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    idx = list(wide.index)
    for ax, (dt, title) in zip(axes, shocks):
        ts = pd.Timestamp(dt)
        if ts not in idx:
            ts = next(x for x in idx if x >= ts)
        i = idx.index(ts)
        lo, hi = i - 10, i + 20
        base = wide.iloc[i + SHOCK_BASELINE_OFFSET]
        rel = (wide.iloc[lo:hi + 1] - base) * 100
        rel.index = range(-10, hi - i + 1)
        for p in PARTS:
            ax.plot(rel.index, rel[p], lw=1.4, color=COLORS[p], label=p)
        ax.axvline(0, color="black", ls="--", lw=1.2)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_title(title, loc="left", fontsize=11, color="#444444")
        ax.set_xlabel("Trading days from event")
    axes[0].set_ylabel("Change in log gross futures open interest (%)")
    axes[0].legend(ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, "shock_windows")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PANEL / "daily_binding.csv", parse_dates=["date"]).sort_values("date")
    l = pd.read_csv(PANEL / "long.csv", parse_dates=["date"]).sort_values(["date", "client_type"])

    fig_multiplier(d)
    fig_volbound_by_year(d)
    fig_discretionary_floor(d)
    fig_participant_shares(l)
    fig_shock_windows()


if __name__ == "__main__":
    main()
