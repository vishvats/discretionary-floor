"""Participant incidence regressions.

Shock: the NIFTY price scan range on the previous-index-close base, purged of
the change in lagged applicable volatility. DV: forward five-day change in log
gross index futures open interest by participant category. Inference is
Newey-West per participant and Driscoll-Kraay on the stacked interacted model.
"""
import sys
from pathlib import Path
from math import exp, sqrt, erfc

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PANEL, RAW

PARTS = ["Client", "FII", "DII", "Pro"]
CTRL = ["dlog_sigma", "ret_nifty", "abs_ret", "vix", "near_expiry"]
CTRL_LAG = ["ret_nifty_lag", "abs_ret_lag", "vix_lag"]
LAGS = 10


# inference

def chi2_p(stat, df):
    if df == 1:
        return erfc(sqrt(stat / 2.0))
    if df == 2:
        return exp(-stat / 2.0)
    if df == 3:
        return erfc(sqrt(stat / 2)) + sqrt(2 * stat / np.pi) * exp(-stat / 2)
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
    return 0.5 * erfc(z / sqrt(2))


def _score_calendar(scores, dates, calendar):
    """Sum scores by date, zero-filling missing trading days."""
    if dates is None or calendar is None:
        return scores
    return (pd.DataFrame(scores, index=pd.DatetimeIndex(dates))
            .groupby(level=0).sum()
            .reindex(pd.DatetimeIndex(calendar), fill_value=0).values)


def ols_hac(y, X, lags=LAGS, dates=None, calendar=None):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    dates = None if dates is None else np.asarray(dates)[ok]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    scores = _score_calendar(X * resid[:, None], dates, calendar)
    S = scores.T @ scores
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = scores[L:].T @ scores[:-L]
        S += w * (G + G.T)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return beta, se, vcov, t, len(y), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def ols_dk(y, X, dates, lags=LAGS, calendar=None):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, dates = y[ok], X[ok], dates[ok]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    H = _score_calendar(X * resid[:, None], dates, calendar)
    if calendar is None:
        df = pd.DataFrame(H)
        df["d"] = dates
        H = df.groupby("d").sum().sort_index().values
    S = H.T @ H
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = H[L:].T @ H[:-L]
        S += w * (G + G.T)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return beta, se, vcov, t, len(y), H.shape[0]


def wald(beta, vcov, R):
    R = np.atleast_2d(R)
    d = R @ beta
    V = R @ vcov @ R.T
    return float(d @ np.linalg.pinv(V) @ d), R.shape[0]


# data 

def load():
    d = pd.read_csv(PANEL / "daily_binding.csv", parse_dates=["date"]).sort_values("date")
    d["abs_ret"] = d["ret_nifty"].abs()

    b = pd.read_csv(RAW / "bhav_index.csv", low_memory=False)
    b = b[b.symbol == "NIFTY"].copy()
    b["date"] = pd.to_datetime(b["date"], format="%Y-%m-%d", errors="coerce")
    b["expiry"] = pd.to_datetime(b["expiry"], format="mixed", dayfirst=True,
                                 errors="coerce")
    nearest = (b.dropna(subset=["date", "expiry"]).query("expiry >= date")
               .groupby("date")["expiry"].min())
    # expiry is static metadata, so the two missing bhavcopy dates can be
    # filled from either adjacent contract list
    cal = pd.DatetimeIndex(d["date"])
    nearest = nearest.reindex(cal)
    prev = nearest.ffill().where(lambda s: s >= s.index)
    nxt = nearest.bfill().where(lambda s: s >= s.index)
    nearest = nearest.fillna(prev).fillna(nxt)
    d["dte"] = ((nearest - nearest.index).dt.days).reindex(
        pd.DatetimeIndex(d["date"])).values
    d["near_expiry"] = (d["dte"] <= 3).astype(float).where(d["dte"].notna())
    for c in ("ret_nifty", "abs_ret", "vix"):
        d[f"{c}_lag"] = d[c].shift(1)
    return d


def residualise(d, margin_col, label):
    dlog_m = np.log(d[margin_col].where(d[margin_col] > 0)).diff()
    dlog_s = np.log(d["sigma_nifty"].where(d["sigma_nifty"] > 0)).diff().shift(1)
    m = pd.DataFrame({"date": d["date"], "m": dlog_m, "s": dlog_s}).dropna()
    bt, *_ = ols_hac(m["m"].values,
                     np.column_stack([np.ones(len(m)), m["s"].values]),
                     dates=m["date"].values, calendar=d["date"].values)
    print(f"purge regression ({label}): intercept {bt[0]:.6f}, slope {bt[1]:.4f}, n {len(m)}")
    d = d.copy()
    d["dlog_sigma"] = dlog_s
    d["margin_exog"] = dlog_m - (bt[0] + bt[1] * dlog_s)
    d["exog_up"] = d["margin_exog"].clip(lower=0)
    d["exog_dn"] = d["margin_exog"].clip(upper=0)
    return d


def participant_wides(calendar):
    """Log gross index OI per category, on the full trading-day calendar.

    The reindex matters: participant OI is absent on three trading days, and
    without it a shift(-5) would silently span six. Dates failing the TOTAL-row
    check in build_panel.py are excluded here too.
    """
    valid = set(pd.read_csv(PANEL / "long.csv", parse_dates=["date"])["date"])
    oi = pd.read_csv(RAW / "participant_oi.csv", low_memory=False)
    oi["date"] = pd.to_datetime(oi["date"])
    oi["client_type"] = oi["client_type"].astype(str).str.strip()
    oi = oi[oi.client_type.isin(PARTS) & oi.date.isin(valid)].copy()
    fut = ["Future Index Long", "Future Index Short"]
    idx_cols = fut + ["Option Index Call Long", "Option Index Put Long",
                      "Option Index Call Short", "Option Index Put Short"]
    oi["fut_gross"] = oi[fut].sum(axis=1, min_count=len(fut))
    oi["comb_gross"] = oi[idx_cols].sum(axis=1, min_count=len(idx_cols))

    def wide(col):
        w = oi.pivot_table(index="date", columns="client_type",
                           values=col, aggfunc="first")[PARTS].sort_index()
        return np.log(w.reindex(calendar).where(lambda x: x > 0))

    return wide("fut_gross"), wide("comb_gross")


def lot_transition_exposure_dates():
    """Dates on which a benchmark board-lot revision is measurable.

    A phased revision shows up as positive OI at more than one lot; an
    all-contract revision as a one-day step in the single lot in force.
    """
    import build_panel as bp
    b = bp.add_lot_size(pd.read_csv(RAW / "bhav_index.csv", low_memory=False,
                                    parse_dates=["date"]))
    b = b[b.symbol.isin(["NIFTY", "BANKNIFTY"]) & b.open_interest.gt(0)]
    sets = b.groupby(["symbol", "date"])["lot"].apply(
        lambda s: tuple(sorted(s.dropna().unique())))
    affected = set(sets[sets.map(len) > 1].index.get_level_values("date"))
    for _, s in sets.groupby(level=0):
        s = s.droplevel(0).sort_index()
        prev = s.shift()
        step = (s.map(len) == 1) & (prev.map(lambda x: len(x) if isinstance(x, tuple) else 0) == 1) & (s != prev)
        affected.update(s.index[step])
    return pd.DatetimeIndex(sorted(affected))


def build_frame(wide, d, end=None, window="event", drop_lot_windows=False,
                calendar=None):
    """Five-trading-day change in log OI aligned to day t.

    The scan range is a begin-of-day parameter and participant OI an
    end-of-day report, so the window runs from the last snapshot before the
    change, OI(t-1), to OI(t+4). window="post" excludes the event day.
    """
    if window == "event":
        fwd = (wide.shift(-4) - wide.shift(1)) * 100
    elif window == "post":
        fwd = (wide.shift(-5) - wide) * 100
    else:
        raise ValueError(window)
    fr = fwd.reset_index().melt(id_vars="index", var_name="client_type",
                                value_name="dOI").rename(columns={"index": "date"})
    X = d[["date", "margin_exog", "exog_up", "exog_dn"] + CTRL + CTRL_LAG]
    fr = fr.merge(X, on="date", how="inner").dropna(subset=["dOI"])
    if end is not None:
        fr = fr[fr.date <= end]
    if drop_lot_windows:
        cal = pd.DatetimeIndex(sorted(wide.index)) if calendar is None else calendar
        pos = {d_: i for i, d_ in enumerate(cal)}
        bad = set()
        affected = set(lot_transition_exposure_dates())
        for tdt, i in pos.items():
            lo, hi = (i - 1, i + 4) if window == "event" else (i, i + 5)
            if any(cal[j] in affected for j in range(max(0, lo), min(len(cal), hi + 1))):
                bad.add(tdt)
        fr = fr[~fr.date.isin(bad)]
    return fr


#stacked tests 

def stacked_walds(frame, ctrl=None, calendar=None):
    """4-way and Client=Pro Driscoll-Kraay Wald statistics."""
    ctrl = CTRL if ctrl is None else ctrl
    f = frame.reset_index(drop=True)
    D = pd.get_dummies(f["client_type"])[PARTS].astype(float).values
    dt = f["date"].values
    cols, names = [], []
    for j, p in enumerate(PARTS):
        cols += [D[:, j], D[:, j] * f["margin_exog"].values]
        names += [f"c_{p}", f"mx_{p}"]
        for c in ctrl:
            cols.append(D[:, j] * f[c].values); names.append(f"{c}_{p}")
    Xc = np.column_stack(cols)
    bb, se, V, t, n, T = ols_dk(f["dOI"].values, Xc, dt, calendar=calendar)
    ix = {nm: i for i, nm in enumerate(names)}
    k4 = [ix[f"mx_{p}"] for p in PARTS]
    R = np.zeros((3, Xc.shape[1]))
    for i in range(3):
        R[i, k4[i]] = 1; R[i, k4[i + 1]] = -1
    st, _ = wald(bb, V, R)
    Rcp = np.zeros(Xc.shape[1]); Rcp[ix["mx_Client"]] = 1; Rcp[ix["mx_Pro"]] = -1
    st2, _ = wald(bb, V, Rcp)
    return st, st2


def stacked_asym_p(frame, ctrl=None, calendar=None):
    """Driscoll-Kraay p-value for beta_up == beta_dn, per participant."""
    ctrl = CTRL if ctrl is None else ctrl
    f = frame.reset_index(drop=True)
    D = pd.get_dummies(f["client_type"])[PARTS].astype(float).values
    dt = f["date"].values
    cols, names = [], []
    for j, p in enumerate(PARTS):
        cols += [D[:, j], D[:, j] * f["exog_up"].values, D[:, j] * f["exog_dn"].values]
        names += [f"c_{p}", f"up_{p}", f"dn_{p}"]
        for c in ctrl:
            cols.append(D[:, j] * f[c].values); names.append(f"{c}_{p}")
    Xa = np.column_stack(cols)
    bb, se, V, t, n, T = ols_dk(f["dOI"].values, Xa, dt, calendar=calendar)
    ix = {nm: i for i, nm in enumerate(names)}
    out = {}
    for p in PARTS:
        R = np.zeros(Xa.shape[1]); R[ix[f"up_{p}"]] = 1; R[ix[f"dn_{p}"]] = -1
        st, _ = wald(bb, V, R)
        out[p] = chi2_p(st, 1)
    return out


# reporting 

def run(frame, label, ctrl=None, calendar=None):
    ctrl = CTRL if ctrl is None else ctrl
    print(f"\n===== {label} =====")
    print("--- per-participant OLS, Newey-West (10 lags)")
    for p in PARTS:
        s = frame[frame.client_type == p]
        Xc = np.column_stack([np.ones(len(s)), s["margin_exog"].values,
                              *[s[c].values for c in ctrl]])
        bb, se, V, t, n, r2 = ols_hac(s["dOI"].values, Xc,
                                      dates=s["date"].values, calendar=calendar)
        pv = chi2_p(t[1] ** 2, 1)
        print(f"  {p:6s} beta {bb[1]:8.2f}  SE {se[1]:6.2f}  t {t[1]:5.2f}  "
              f"p {pv:.3f}  n {n}  R2 {r2:.3f}")

    f = frame.reset_index(drop=True)
    D = pd.get_dummies(f["client_type"])[PARTS].astype(float).values
    dt = f["date"].values

    cols, names = [], []
    for j, p in enumerate(PARTS):
        cols += [D[:, j], D[:, j] * f["margin_exog"].values]
        names += [f"c_{p}", f"mx_{p}"]
        for c in ctrl:
            cols.append(D[:, j] * f[c].values); names.append(f"{c}_{p}")
    Xc = np.column_stack(cols)
    bb, se, V, t, n, T = ols_dk(f["dOI"].values, Xc, dt, calendar=calendar)
    ix = {nm: i for i, nm in enumerate(names)}
    k4 = [ix[f"mx_{p}"] for p in PARTS]
    R = np.zeros((3, Xc.shape[1]))
    for i in range(3):
        R[i, k4[i]] = 1; R[i, k4[i + 1]] = -1
    st, dfr = wald(bb, V, R)
    Rcp = np.zeros(Xc.shape[1]); Rcp[ix["mx_Client"]] = 1; Rcp[ix["mx_Pro"]] = -1
    st2, _ = wald(bb, V, Rcp)
    print(f"--- stacked Driscoll-Kraay: 4-way chi2(3) = {st:.3f} p = {chi2_p(st, 3):.4f}; "
          f"Client=Pro chi2(1) = {st2:.3f} p = {chi2_p(st2, 1):.4f}")

    cols, names = [], []
    for j, p in enumerate(PARTS):
        cols += [D[:, j], D[:, j] * f["exog_up"].values, D[:, j] * f["exog_dn"].values]
        names += [f"c_{p}", f"up_{p}", f"dn_{p}"]
        for c in ctrl:
            cols.append(D[:, j] * f[c].values); names.append(f"{c}_{p}")
    Xa = np.column_stack(cols)
    bb, se, V, t, n, T = ols_dk(f["dOI"].values, Xa, dt, calendar=calendar)
    ix = {nm: i for i, nm in enumerate(names)}
    print("--- asymmetry (up = dn), Driscoll-Kraay:")
    for p in PARTS:
        R = np.zeros(Xa.shape[1]); R[ix[f"up_{p}"]] = 1; R[ix[f"dn_{p}"]] = -1
        st, _ = wald(bb, V, R)
        print(f"  {p:6s} up {bb[ix[f'up_{p}']]:8.2f}  dn {bb[ix[f'dn_{p}']]:8.2f}  "
              f"chi2 {st:.3f}  p {chi2_p(st, 1):.4f}")


def main():
    d = load()
    calendar = pd.DatetimeIndex(sorted(d["date"].unique()))
    wf, wc = participant_wides(calendar)

    dp = residualise(d, "psr_spot_nifty", "NIFTY PSR, index-close base")
    print("\n--- 20 largest |margin_exog|:")
    top = dp.reindex(dp["margin_exog"].abs().sort_values(ascending=False).index).head(20)
    top = top.assign(rank=range(1, len(top) + 1))
    print(top[["rank", "date", "regime", "psr_spot_nifty", "margin_exog"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    events = ["2018-09-21", "2018-09-28", "2018-10-26", "2018-11-30",
              "2019-01-21", "2020-06-01"]
    ranks = {e: int(top.loc[top.date == e, "rank"].iloc[0])
             for e in events if (top.date == e).any()}
    print(f"  ranks of the six dated institutional adjustments: {ranks}")

    run(build_frame(wf, dp), "PRIMARY: futures-only DV, NIFTY-PSR shock, OI(t+4)-OI(t-1)", calendar=calendar)
    run(build_frame(wf, dp, window="post"),
        "ROBUSTNESS: post-event-day window, OI(t+5)-OI(t)", calendar=calendar)
    run(build_frame(wc, dp), "ROBUSTNESS: combined futures+options DV", calendar=calendar)

    dm = residualise(d, "margin", "lag-weighted two-index margin")
    run(build_frame(wf, dm), "ROBUSTNESS: futures-only DV, two-index margin shock", calendar=calendar)

    run(build_frame(wf, dp, end="2024-11-19"),
        "ROBUSTNESS: sample ends 19 Nov 2024 (pre lot revision)", calendar=calendar)

    run(build_frame(wf, dp),
        "ROBUSTNESS: market controls lagged to t-1",
        ctrl=["dlog_sigma", "near_expiry"] + CTRL_LAG, calendar=calendar)

    run(build_frame(wf, dp, drop_lot_windows=True, calendar=calendar),
        "ROBUSTNESS: windows overlapping any mixed-lot phase or all-contract lot step dropped",
        calendar=calendar)


if __name__ == "__main__":
    main()
