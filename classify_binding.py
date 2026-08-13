"""Classify each trading day as volatility bound or floor bound.

PSR = max(k * sigma(t-1), F), everything on the previous index close, which is
the base on which the implemented floor is exact. Where a plateau identifies
the floor the comparison is exact, where none does, the implied multiplier is
compared against theory instead.
"""
import numpy as np
import pandas as pd

from paths import PANEL

TOL_BAND = 0.02     # k band where no floor is identified
TOL_FLOOR = 0.005   # pp tolerance on floor equality

# formula and volatility estimator changed on the same date
TRANSITIONS = ["2020-06-01"]

K3 = 3 * np.sqrt(2)
K6 = 6 * np.sqrt(2)

REGIMES = [
    ("2012-07-05", "2019-01-20", "Pre-MPOR",  K3),
    ("2019-01-21", "2020-05-31", "MPOR",      K3),
    ("2020-06-01", "2024-11-19", "6sigma",    K6),
    ("2024-11-20", "2026-07-31", "Post-2024", K6),
]

ORDER = ["Pre-MPOR", "MPOR", "6sigma", "Post-2024"]

FLOORS = {
    "nifty": [
        ("2019-01-21", "2020-05-31", 5.0 * np.sqrt(2)),
        ("2020-06-01", "2026-07-31", 9.30),
    ],
    "banknifty": [
        ("2019-01-21", "2019-04-04", 5.0 * np.sqrt(2)),
        ("2019-04-05", "2019-04-11", 5.5 * np.sqrt(2)),
        ("2019-04-12", "2019-04-18", 6.0 * np.sqrt(2)),
        ("2019-04-22", "2020-03-12", 6.5 * np.sqrt(2)),
        ("2023-01-19", "2026-02-26", 11.30),
        ("2026-02-27", "2026-07-31", 9.30),
    ],
}


def classify(d: pd.DataFrame, sym: str, band: float = TOL_BAND) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    out["psr_spot"] = d[f"psr_spot_{sym}"]
    out["sig_lag"] = d[f"sigma_{sym}"].shift(1)
    out["k"] = out["psr_spot"] / 100 / out["sig_lag"]

    out["regime"] = pd.NA
    out["k_theory"] = np.nan
    for lo, hi, name, kt in REGIMES:
        m = (d.date >= lo) & (d.date <= hi)
        out.loc[m, "regime"] = name
        out.loc[m, "k_theory"] = kt

    out["floor"] = np.nan
    for lo, hi, f in FLOORS[sym]:
        out.loc[(d.date >= lo) & (d.date <= hi), "floor"] = f

    # on a transition date the lag convention pairs a new scan range with an
    # old-vintage sigma, so classify off the same-day value there
    k_cls = out["k"].copy()
    tr = d.date.isin([pd.Timestamp(x) for x in TRANSITIONS]).values
    k_cls[tr] = out.loc[tr, "psr_spot"] / 100 / d.loc[tr, f"sigma_{sym}"]

    has_floor = out["floor"].notna()
    ok = out["k"].notna()
    out["vol_bound"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[has_floor & ok, "vol_bound"] = (
        out.loc[has_floor, "psr_spot"] > out.loc[has_floor, "floor"] + TOL_FLOOR)
    out.loc[~has_floor & ok, "vol_bound"] = (
        (k_cls[~has_floor] / out.loc[~has_floor, "k_theory"] - 1).abs() <= band)

    out["unclassified_reason"] = ""
    out.loc[out.psr_spot.isna(), "unclassified_reason"] = "missing_index_close_psr"
    out.loc[out.psr_spot.notna() & out.sig_lag.isna(),
            "unclassified_reason"] = "missing_lagged_volatility"
    out.loc[out.psr_spot.notna() & out.sig_lag.notna() & out.k_theory.isna(),
            "unclassified_reason"] = "outside_defined_regime"

    out["slack"] = out["k"] / out["k_theory"]
    return out


def report(d, res, label):
    cc = res[res.k.notna()].copy()
    cc["date"] = d.loc[cc.index, "date"]
    tab = cc.groupby("regime")["vol_bound"].agg(days="count", vb="sum", share="mean")
    tab["share%"] = (tab["share"] * 100).round(1)
    print(f"\n{label}")
    print(tab.reindex(ORDER)[["days", "vb", "share%"]].to_string())
    yr = (cc.groupby(cc.date.dt.year)["vol_bound"].mean() * 100).round(1)
    print("by year: " + ", ".join(f"{y}:{v}" for y, v in yr.items()))
    return cc


def main():
    d = (pd.read_csv(PANEL / "daily.csv", parse_dates=["date"])
           .sort_values("date").reset_index(drop=True))

    rn = classify(d, "nifty")
    rb = classify(d, "banknifty")
    cn = report(d, rn, "NIFTY")
    report(d, rb, "BANKNIFTY")

    excluded = sorted(set(d.loc[d.psr_nifty.notna(), "date"]) - set(cn["date"]))
    print(f"\nclassified {len(cn)} of {int(d.psr_nifty.notna().sum())} scan-range days; "
          f"excluded {[str(x.date()) for x in excluded]}")

    out = d.copy()
    for col, src in [("psr_spot_nifty", "psr_spot"), ("sig_lag", "sig_lag"),
                     ("k", "k"), ("regime", "regime"), ("k_theory", "k_theory"),
                     ("floor_nifty", "floor"), ("vol_bound", "vol_bound"),
                     ("unclassified_reason", "unclassified_reason"),
                     ("slack", "slack")]:
        out[col] = rn[src]
    for col, src in [("psr_spot_banknifty", "psr_spot"), ("k_banknifty", "k"),
                     ("floor_banknifty", "floor"),
                     ("vol_bound_banknifty", "vol_bound"),
                     ("unclassified_reason_banknifty", "unclassified_reason")]:
        out[col] = rb[src]

    out.to_csv(PANEL / "daily_binding.csv", index=False)
    print(f"wrote {PANEL / 'daily_binding.csv'}")


if __name__ == "__main__":
    main()
