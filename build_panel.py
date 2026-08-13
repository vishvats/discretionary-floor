"""Merge the parsed NSE sources into a daily panel and a participant panel."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import PANEL, RAW, VIX

PARTICIPANTS = ["Client", "DII", "FII", "Pro"]

IDX_LONG = ["Future Index Long", "Option Index Call Long", "Option Index Put Long"]
IDX_SHORT = ["Future Index Short", "Option Index Call Short", "Option Index Put Short"]

BENCHMARKS = ["NIFTY", "BANKNIFTY"]

# tolerance, in contracts, on the participant file's own TOTAL row
TOTAL_TOL = 10


# io

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\t", " ")).strip()


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.normalize()


def to_date_flex(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed").dt.normalize()


def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [norm(c) for c in df.columns]
    df["date"] = to_date(df["date"] if "date" in df.columns else df.iloc[:, 0])
    return df.dropna(subset=["date"]).drop_duplicates()


def load_vix(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        print(f"  ! VIX not found at {path}, continuing without it")
        return None
    df = pd.read_csv(path)
    df.columns = [norm(c) for c in df.columns]
    dcol = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    ccol = next((c for c in df.columns if "close" in c.lower()), None)
    if ccol is None:
        print(f"  ! no close column in VIX file: {list(df.columns)}")
        return None
    out = pd.DataFrame({"date": to_date_flex(df[dcol]),
                        "vix": pd.to_numeric(df[ccol].astype(str).str.replace(",", ""),
                                             errors="coerce")})
    return out.dropna(subset=["date"]).drop_duplicates("date")


# participant shares 

def build_shares(oi: pd.DataFrame) -> pd.DataFrame:
    have_long = [c for c in IDX_LONG if c in oi.columns]
    have_short = [c for c in IDX_SHORT if c in oi.columns]
    if not have_long or not have_short:
        raise SystemExit(f"index OI columns not found. Present: {list(oi.columns)}")

    oi = oi.copy()
    oi["client_type"] = oi["client_type"].astype(str).str.strip()
    # min_count=1 keeps an all-blank record missing instead of summing it to zero
    oi["gross"] = (oi[have_long].sum(axis=1, min_count=1) +
                   oi[have_short].sum(axis=1, min_count=1))

    p = oi[oi.client_type.isin(PARTICIPANTS)].copy()

    bad = set(pd.Timestamp(x) for x in p.loc[p["gross"].isna(), "date"].unique())
    tot = oi[oi.client_type.str.upper() == "TOTAL"]
    if len(tot):
        cat = p.groupby("date")[have_long + have_short].sum(min_count=1)
        ref = tot.set_index("date")[have_long + have_short]
        for dt in cat.index:
            if dt not in ref.index or ref.loc[dt].isna().all():
                bad.add(dt)
                continue
            gap = (cat.loc[dt] - ref.loc[dt]).abs().max()
            if not np.isfinite(gap) or gap > TOTAL_TOL:
                bad.add(dt)
    if bad:
        print(f"  ! dropping {len(bad)} date(s) failing the TOTAL check: "
              f"{sorted(str(x.date()) for x in bad)}")
        p = p[~p.date.isin(bad)]

    p = p.merge(p.groupby("date")["gross"].sum().rename("gross_total"), on="date")
    p = p[p.gross_total > 0].copy()
    p["share"] = p["gross"] / p["gross_total"]
    return p[["date", "client_type", "gross", "gross_total", "share"]]


# lot sizes

def add_lot_size(bhav: pd.DataFrame) -> pd.DataFrame:
    """Board lot and open interest in contracts.

    Bhavcopy open interest is in underlying units, so it needs the lot. The lot
    is not fixed for the life of a contract: some revisions re-specify every
    outstanding contract on one date (NIFTY November 2014 goes 50 to 25 on
    31 October). A centred rolling median tracks that step while ignoring
    single-day distortions in traded value.
    """
    b = bhav.copy()
    b["expiry_dt"] = pd.to_datetime(b["expiry"], format="mixed", dayfirst=True,
                                    errors="coerce")
    b = b.sort_values(["symbol", "expiry_dt", "date"])
    med = (b.groupby(["symbol", "expiry_dt"])["implied_lot"]
             .transform(lambda s: s.rolling(7, center=True, min_periods=1).median()))
    med = med.groupby([b.symbol, b.expiry_dt]).transform(lambda s: s.ffill().bfill())
    est = ((med / 5).round() * 5)
    # kept separately from stated_lot so the overlap check is not tautological
    b["lot_inferred"] = est.where(est > 0)
    b["lot"] = b["stated_lot"].where(b["stated_lot"].notna(), b["lot_inferred"])
    b["oi_contracts"] = b["open_interest"] / b["lot"]
    b["oi_notional"] = b["open_interest"] * b["close"]
    return b


def add_effective_expiry(bhav: pd.DataFrame, calendar) -> pd.DataFrame:
    """Last exchange trading day on or before nominal expiry."""
    b = bhav.copy()
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    exp = pd.DatetimeIndex(b["expiry_dt"])
    pos = cal.searchsorted(exp, side="right") - 1
    eff = pd.Series(pd.NaT, index=b.index, dtype="datetime64[ns]")
    inside = (pos >= 0) & exp.notna() & (exp <= cal.max())
    eff.loc[inside] = cal[pos[inside]].values
    eff.loc[~inside & exp.notna()] = exp[~inside & exp.notna()].values
    b["effective_expiry_dt"] = eff
    return b


# benchmark weights 

def build_weights(bhav: pd.DataFrame, calendar) -> pd.DataFrame:
    """Previous-day OI weights, counting only contracts still open."""
    bidx = add_lot_size(bhav[bhav.symbol.isin(BENCHMARKS)])
    if bidx.empty:
        return pd.DataFrame(index=calendar)
    bidx = add_effective_expiry(bidx, calendar)
    src = {d: g for d, g in bidx.groupby("date")}

    rows = {}
    for i, t in enumerate(calendar):
        if i == 0:
            continue
        prev = calendar[i - 1]      
        if prev not in src:
            continue
        g = src[prev]
        s = g[g.effective_expiry_dt >= t].groupby("symbol")["oi_contracts"].sum()
        if s.sum() > 0:
            rows[t] = s / s.sum()
    w = pd.DataFrame(rows).T.reindex(calendar)
    w.columns = [f"w_{c.lower()}" for c in w.columns]
    return w


# daily panel

def build_daily(span: pd.DataFrame, vol: pd.DataFrame,
                bhav: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    span = span[span.symbol.isin(BENCHMARKS)].copy()
    wide = span.pivot_table(index="date", columns="symbol",
                            values="price_scan_pct", aggfunc="first")
    wide.columns = [f"psr_{c.lower()}" for c in wide.columns]

    scan = span.pivot_table(index="date", columns="symbol",
                            values="price_scan", aggfunc="first")
    scan.columns = [f"scan_{c.lower()}" for c in scan.columns]
    wide = wide.join(scan, how="left")
    cal = pd.DatetimeIndex(sorted(set(wide.index) | set(bhav.date)))
    wide = wide.join(build_weights(bhav, cal), how="left").reset_index(names="date")

    v = vol[vol.symbol.isin(BENCHMARKS)].copy()

    def pivot(col, prefix):
        t = v.pivot_table(index="date", columns="symbol", values=col, aggfunc="first")
        t.columns = [f"{prefix}_{c.lower()}" for c in t.columns]
        return t.reset_index()

    daily = (wide.merge(pivot("sigma_applicable", "sigma"), on="date", how="outer")
                 .merge(pivot("fut_close", "px"), on="date", how="outer")
                 .merge(pivot("spot_prev", "spot_prev"), on="date", how="outer")
                 .sort_values("date"))


    for s in ("nifty", "banknifty"):
        daily[f"psr_spot_{s}"] = 100 * daily[f"scan_{s}"] / daily[f"spot_prev_{s}"]

  
    daily["margin"] = (daily.psr_spot_nifty * daily.w_nifty +
                       daily.psr_spot_banknifty * daily.w_banknifty)

    daily["ret_nifty"] = np.log(daily.px_nifty / daily.px_nifty.shift(1))

    if vix is not None:
        daily = daily.merge(vix, on="date", how="left")
    return daily


def coverage(sources: dict, daily: pd.DataFrame, panel: pd.DataFrame) -> str:
    lines = []
    for label, df in sources.items():
        if df is None:
            continue
        lines.append(f"{label:<18} {len(df):>7,} rows   "
                     f"{df.date.min().date()} -> {df.date.max().date()}   "
                     f"{df.date.nunique():>5,} days")
    lines += ["",
              f"daily panel        {len(daily):,} days",
              f"participant panel  {len(panel):,} rows on {panel.date.nunique():,} dates",
              ""]
    miss = daily.loc[daily["margin"].isna(), "date"]
    lines.append(f"aggregate margin missing on {len(miss)} day(s): "
                 + ", ".join(str(x.date()) for x in miss))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=RAW)
    ap.add_argument("--vix", type=Path, default=VIX)
    ap.add_argument("--out", type=Path, default=PANEL)
    args = ap.parse_args()

    d = args.data_dir
    oi = load(d / "participant_oi.csv", "participant OI")
    vol = load(d / "volatility.csv", "volatility")
    bhav = load(d / "bhav_index.csv", "bhavcopy")
    span = load(d / "span_psr.csv", "SPAN")
    vix = load_vix(args.vix)

    shares = build_shares(oi)
    daily = build_daily(span, vol, bhav, vix)
    panel = shares.merge(daily, on="date", how="left").sort_values(["date", "client_type"])

    args.out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out / "daily.csv", index=False)
    panel.to_csv(args.out / "long.csv", index=False)

    txt = coverage({"participant_oi": oi, "volatility": vol, "bhav_index": bhav,
                    "span_psr": span, "india_vix": vix}, daily, panel)
    (args.out / "coverage.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
