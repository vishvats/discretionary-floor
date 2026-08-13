"""Download and parse the NSE archives used by the paper.

Four sources per trading day: the participant-wise open interest report, the
daily volatility file, the begin-of-day SPAN risk parameter file, and the
derivatives bhavcopy. Progress is kept per (date, source) in _state.json, so
an interrupted run resumes without refetching what it already has.

    python nse_pipeline.py --start 2012-07-05 --end 2026-07-31
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

BASE = "https://nsearchives.nseindia.com"

URL_TEMPLATES = {
    "participant_oi": [
        f"{BASE}/content/nsccl/fao_participant_oi_{{d}}{{m}}{{Y}}.csv"
    ],
    "fovolt": [
        f"{BASE}/archives/nsccl/volt/FOVOLT_{{d}}{{m}}{{Y}}.csv"
    ],
    "span": [
        "https://www.nseindia.com/api/reports?archives="
        "%5B%7B%22name%22%3A%22F%26O%20-%20Span%20Risk%20Parameter%20File%20"
        "(Begin%20of%20day)%22%2C%22type%22%3A%22archives%22%2C%22category%22%3A"
        "%22derivatives%22%2C%22section%22%3A%22equity%22%7D%5D"
        "&date={apidate}&type=equity&mode=single",
    ],
    "bhav_legacy": [
        f"{BASE}/content/historical/DERIVATIVES/{{Y}}/{{B}}/fo{{b}}bhav.csv.zip"
    ],
    "bhav_udiff": [
        f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{{iso}}_F_0000.csv.zip"
    ],
}

UDIFF_FROM = date(2024, 7, 8)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

SLEEP = 1.2
RETRIES = 3
TIMEOUT = 90

INDEX_FUT_CODES = {"FUTIDX", "IDF"}

SOURCES = ["participant_oi", "fovolt", "span", "bhav"]


# ---- http ----

def date_fields(d: date) -> dict:
    return dict(d=f"{d.day:02d}", m=f"{d.month:02d}", Y=f"{d.year}",
                iso=d.strftime("%Y%m%d"),
                b=d.strftime("%d%b%Y").upper(),
                B=d.strftime("%b").upper(),
                apidate=d.strftime("%d-%b-%Y"))


class Fetcher:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._bootstrap()
        self.working: dict[str, str] = {}

    def _bootstrap(self):
        for url in ("https://www.nseindia.com",
                    "https://www.nseindia.com/all-reports-derivatives"):
            try:
                self.s.get(url, timeout=TIMEOUT)
                time.sleep(0.8)
            except Exception as e:
                print(f"  ! cookie bootstrap failed for {url}: {e}", file=sys.stderr)

    def get(self, source: str, d: date) -> bytes | None:
        fields = date_fields(d)
        templates = URL_TEMPLATES[source]
        if source in self.working:
            templates = [self.working[source]] + [t for t in templates
                                                  if t != self.working[source]]

        for tmpl in templates:
            url = tmpl.format(**fields)
            for attempt in range(RETRIES):
                try:
                    r = self.s.get(url, timeout=TIMEOUT)
                    if r.status_code == 200 and len(r.content) > 200:
                        self.working[source] = tmpl
                        return r.content
                    if r.status_code in (403, 401):
                        self._bootstrap()
                        continue
                    break
                except requests.RequestException:
                    time.sleep(2 * (attempt + 1))
                finally:
                    time.sleep(SLEEP)
        return None


#helpers

def unzip_first(raw: bytes, want_ext: tuple[str, ...]) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(want_ext):
                    return zf.read(name)
    except zipfile.BadZipFile:
        return raw if not raw[:2] == b"PK" else None
    return None


def norm_cols(cols) -> list[str]:
    return [re.sub(r"\s+", " ", str(c).replace("\t", " ")).strip() for c in cols]


def read_csv_bytes(raw: bytes, **kw) -> pd.DataFrame | None:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False, **kw)
        except Exception:
            continue
    return None


# parsers

CANON_OI = ["Future Index Long", "Future Index Short",
            "Future Stock Long", "Future Stock Short",
            "Option Index Call Long", "Option Index Put Long",
            "Option Index Call Short", "Option Index Put Short",
            "Option Stock Call Long", "Option Stock Put Long",
            "Option Stock Call Short", "Option Stock Put Short",
            "Total Long Contracts", "Total Short Contracts"]


def parse_participant_oi(raw: bytes, d: date) -> pd.DataFrame | None:
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    hdr = None
    for i, ln in enumerate(lines[:5]):
        low = ln.lower()
        if "client type" in low or "future index long" in low:
            hdr = i
            break
    if hdr is None:
        return None

    df = read_csv_bytes(raw, skiprows=hdr)
    if df is None or df.empty:
        return None
    df.columns = norm_cols(df.columns)

    df = df.rename(columns={df.columns[0]: "client_type"})
    df["client_type"] = df["client_type"].astype(str).str.strip()
    df = df[df["client_type"].str.upper().isin(
        ["CLIENT", "DII", "FII", "PRO", "TOTAL"])]
    if df.empty:
        return None

    def key(c):
        return re.sub(r"[^a-z]", "", str(c).lower())
    lookup = {key(c): c for c in df.columns}
    out = pd.DataFrame({"date": d.isoformat(),
                        "client_type": df["client_type"].values})
    for canon in CANON_OI:
        src = lookup.get(key(canon))
        out[canon] = pd.to_numeric(df[src], errors="coerce").values if src else pd.NA
    return out


def parse_fovolt(raw: bytes, d: date) -> pd.DataFrame | None:
    df = read_csv_bytes(raw)
    if df is None or df.empty:
        return None
    df.columns = norm_cols(df.columns)
    symcol = next((c for c in df.columns if c.lower().startswith("symbol")), None)
    if symcol is None:
        return None
    df[symcol] = df[symcol].astype(str).str.strip().str.upper()
    df = df[df[symcol].isin(SYMBOLS)]
    if df.empty:
        return None

    def pick(*frags):
        for c in df.columns:
            lc = c.lower()
            if all(f in lc for f in frags):
                return c
        return None

    out = pd.DataFrame({"date": d.isoformat(), "symbol": df[symcol].values})
    for name, frags in {
        "spot_close":       ("underlying close",),
        "spot_prev":        ("underlying previous",),
        "fut_close":        ("futures close",),
        "fut_prev":         ("futures previous",),
        "sigma_underlying": ("current day underlying",),
        "sigma_futures":    ("current day futures",),
        "sigma_applicable": ("applicable daily",),
        "ann_applicable":   ("applicable annualised",),
    }.items():
        col = pick(*frags)
        out[name] = pd.to_numeric(df[col], errors="coerce").values if col else pd.NA
    return out


_FUTPF = re.compile(r"<futPf>(.*?)</futPf>", re.S)
_FIRSTFUT = re.compile(r"<fut>(.*?)</fut>", re.S)


def parse_span(raw: bytes, d: date) -> pd.DataFrame | None:
    """Nearest-expiry priceScan per index symbol, then drop the 15-60 MB file."""
    payload = unzip_first(raw, (".spn", ".xml")) or raw
    txt = payload.decode("latin-1", errors="replace").replace("\r", "")

    rows, seen = [], set()
    for blk in _FUTPF.findall(txt):
        m = re.search(r"<pfCode>([^<]+)</pfCode>", blk)
        if not m:
            continue
        sym = m.group(1).strip().upper()
        if sym not in SYMBOLS or sym in seen:
            continue
        fm = _FIRSTFUT.search(blk)
        if not fm:
            continue
        f = fm.group(1)
        p = re.search(r"<p>([\d.]+)</p>", f)
        ps = re.search(r"<priceScan>([\d.]+)</priceScan>", f)
        if not (p and ps):
            continue
        vs = re.search(r"<volScan>([\d.]+)</volScan>", f)
        pe = re.search(r"<pe>(\d+)</pe>", f)
        price, scan = float(p.group(1)), float(ps.group(1))
        seen.add(sym)
        rows.append(dict(date=d.isoformat(), symbol=sym,
                         expiry=pe.group(1) if pe else None,
                         futures_price=price, price_scan=scan,
                         price_scan_pct=round(scan / price * 100, 4) if price else None,
                         vol_scan=float(vs.group(1)) if vs else None))
    return pd.DataFrame(rows) if rows else None


def parse_bhav(raw: bytes, d: date, udiff: bool) -> pd.DataFrame | None:
    payload = unzip_first(raw, (".csv",)) or raw
    df = read_csv_bytes(payload)
    if df is None or df.empty:
        return None
    df.columns = norm_cols(df.columns)
    up = {c.upper(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.upper() in up:
                return up[n.upper()]
        return None

    c_sym = col("SYMBOL", "TckrSymb")
    c_ins = col("INSTRUMENT", "FinInstrmTp")
    c_exp = col("EXPIRY_DT", "XpryDt")
    c_cls = col("CLOSE", "ClsPric")
    c_con = col("CONTRACTS", "TtlTradgVol")
    c_val = col("VAL_INLAKH", "TtlTrfVal")
    c_oi = col("OPEN_INT", "OpnIntrst")
    c_lot = col("NewBrdLotQty", "LotSz", "MKT_LOT")
    if not (c_sym and c_cls):
        return None

    if c_ins:
        df = df[df[c_ins].astype(str).str.upper().str.strip().isin(INDEX_FUT_CODES)]
    df = df[df[c_sym].astype(str).str.upper().str.strip().isin(SYMBOLS)]
    if df.empty:
        return None

    out = pd.DataFrame({
        "date": d.isoformat(),
        "symbol": df[c_sym].astype(str).str.upper().str.strip().values,
        "expiry": df[c_exp].astype(str).values if c_exp else None,
        "close": pd.to_numeric(df[c_cls], errors="coerce").values,
        "contracts": pd.to_numeric(df[c_con], errors="coerce").values if c_con else pd.NA,
        "value": pd.to_numeric(df[c_val], errors="coerce").values if c_val else pd.NA,
        "open_interest": pd.to_numeric(df[c_oi], errors="coerce").values if c_oi else pd.NA,
        "stated_lot": pd.to_numeric(df[c_lot], errors="coerce").values if c_lot else pd.NA,
    })

    # notional per contract over price. the legacy field is in lakhs
    mult = 1.0 if udiff else 1e5
    with pd.option_context("mode.chained_assignment", None):
        ok = out["contracts"].gt(0) & out["close"].gt(0) & out["value"].notna()
        out["implied_lot"] = pd.NA
        out.loc[ok, "implied_lot"] = (
            out.loc[ok, "value"] * mult / out.loc[ok, "contracts"] / out.loc[ok, "close"]
        ).round(2)
    return out


# 
# orchestration

def trading_days(start: date, end: date, monthly: bool):
    d = start
    seen_months = set()
    while d <= end:
        if d.weekday() < 5:
            key = (d.year, d.month)
            if not monthly or key not in seen_months:
                seen_months.add(key)
                yield d
        d += timedelta(days=1)


class Store:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.buf: dict[str, list[pd.DataFrame]] = {k: [] for k in
                                                   ("participant_oi", "volatility",
                                                    "span_psr", "bhav_index")}

    def add(self, key: str, df: pd.DataFrame | None):
        if df is not None and not df.empty:
            self.buf[key].append(df)

    def flush(self):
        for key, frames in self.buf.items():
            if not frames:
                continue
            path = self.outdir / f"{key}.csv"
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(path, mode="a", header=not path.exists(), index=False)
            self.buf[key] = []


def probe(fetcher: Fetcher, d: date):
    print(f"\nProbing {d} - testing every URL template\n" + "=" * 60)
    fields = date_fields(d)
    for source, templates in URL_TEMPLATES.items():
        print(f"\n{source}")
        for tmpl in templates:
            url = tmpl.format(**fields)
            try:
                r = fetcher.s.get(url, timeout=TIMEOUT)
                mark = "OK " if (r.status_code == 200 and len(r.content) > 200) else "   "
                print(f"  {mark}[{r.status_code}] {len(r.content):>10,}b  {url}")
            except Exception as e:
                print(f"     [ERR] {url}\n           {e}")
            time.sleep(SLEEP)
    print("\nUse the OK lines. Edit URL_TEMPLATES to put working ones first.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2012-07-05")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--outdir", type=Path, default=Path("nse_data"))
    ap.add_argument("--monthly", action="store_true",
                    help="one trading day per month - cheap regime map")
    ap.add_argument("--probe", metavar="YYYY-MM-DD",
                    help="test URL templates on one date and exit")
    ap.add_argument("--skip", nargs="*", default=[], choices=SOURCES,
                    help="sources to skip (e.g. --skip span for a fast first pass)")
    ap.add_argument("--flush-every", type=int, default=20)
    args = ap.parse_args()

    fetcher = Fetcher()

    if args.probe:
        probe(fetcher, date.fromisoformat(args.probe))
        return

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    args.outdir.mkdir(parents=True, exist_ok=True)
    state_path = args.outdir / "_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "missing": []}
    done = set(state["done"])

    # state is per (date, source): marking a whole date done on the first
    # success would let a transient SPAN failure be skipped for ever
    wanted = [s for s in SOURCES if s not in args.skip]
    fetched = set(state.get("fetched", []))          # "iso|source"
    unavailable = set(state.get("unavailable", []))  # confirmed absent

    def settled(iso, src):
        return f"{iso}|{src}" in fetched or f"{iso}|{src}" in unavailable

    store = Store(args.outdir)
    days = [d for d in trading_days(start, end, args.monthly)
            if not all(settled(d.isoformat(), s) for s in wanted)]
    print(f"{len(days)} days to process -> {args.outdir}  "
          f"({len(fetched)} date-source pairs already fetched)")

    processed = 0
    try:
        for i, d in enumerate(days, 1):
            iso = d.isoformat()
            got, failed = [], []

            if "participant_oi" in wanted and not settled(iso, "participant_oi"):
                raw = fetcher.get("participant_oi", d)
                df = parse_participant_oi(raw, d) if raw else None
                if df is not None and not df.empty:
                    store.add("participant_oi", df)
                    fetched.add(f"{iso}|participant_oi"); got.append("oi")
                else:
                    failed.append("participant_oi")

            if "fovolt" in wanted and not settled(iso, "fovolt"):
                raw = fetcher.get("fovolt", d)
                df = parse_fovolt(raw, d) if raw else None
                if df is not None and not df.empty:
                    store.add("volatility", df)
                    fetched.add(f"{iso}|fovolt"); got.append("vol")
                else:
                    failed.append("fovolt")

            if "span" in wanted and not settled(iso, "span"):
                raw = fetcher.get("span", d)
                df = parse_span(raw, d) if raw else None
                del raw
                if df is not None and not df.empty:
                    store.add("span_psr", df)
                    fetched.add(f"{iso}|span"); got.append("span")
                else:
                    failed.append("span")

            if "bhav" in wanted and not settled(iso, "bhav"):
                udiff = d >= UDIFF_FROM
                raw = fetcher.get("bhav_udiff" if udiff else "bhav_legacy", d)
                if not raw:
                    udiff = not udiff
                    raw = fetcher.get("bhav_udiff" if udiff else "bhav_legacy", d)
                df = parse_bhav(raw, d, udiff) if raw else None
                if df is not None and not df.empty:
                    store.add("bhav_index", df)
                    fetched.add(f"{iso}|bhav"); got.append("bhav")
                else:
                    failed.append("bhav")

            if got and not failed:
                done.add(iso)
            if got:
                processed += 1
            note = f"  [retry next run: {'+'.join(failed)}]" if failed else ""
            if got or failed:
                print(f"[{i}/{len(days)}] {iso}  {'+'.join(got) or '--'}{note}")
            else:
                state["missing"].append(iso)
                print(f"[{i}/{len(days)}] {iso}  -- nothing (holiday?)")

            if processed and processed % args.flush_every == 0:
                store.flush()
                state["done"] = sorted(done)
                state["fetched"] = sorted(fetched)
                state["unavailable"] = sorted(unavailable)
                state_path.write_text(json.dumps(state, indent=1))
    except KeyboardInterrupt:
        print("\ninterrupted - saving progress")
    finally:
        store.flush()
        state["done"] = sorted(done)
        state["fetched"] = sorted(fetched)
        state["unavailable"] = sorted(unavailable)
        state["missing"] = sorted(set(state["missing"]))
        state_path.write_text(json.dumps(state, indent=1))
        print(f"\ncomplete days: {len(done)}   date-source pairs fetched: "
              f"{len(fetched)}   missing days: {len(state['missing'])}")
        pending = sum(1 for d in trading_days(start, end, args.monthly)
                      for s in wanted if not settled(d.isoformat(), s))
        if pending:
            print(f"{pending} date-source pairs still pending; rerun to retry them. "
                  f"Move confirmed-absent pairs into state['unavailable'] to stop retrying.")
        print(f"outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
