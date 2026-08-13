"""Adversarial tests for the failure modes this pipeline has actually hit."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_panel as bp
from paths import PANEL, RAW, ROOT

# only checked if the manuscript sits next to the code
PAPER = ROOT / "paper.tex"

passed = 0


def test(name, condition):
    global passed
    if not bool(condition):
        raise AssertionError(name)
    passed += 1
    print(f"PASS {passed:02d}: {name}")


d = pd.read_csv(PANEL / "daily_binding.csv", parse_dates=["date"])

#classifier edge cases
r = d.set_index("date")
test("6-Dec-2022 NIFTY is floor-bound", not bool(r.loc["2022-12-06", "vol_bound"]))
test("3-Feb-2023 BANKNIFTY is floor-bound", not bool(r.loc["2023-02-03", "vol_bound_banknifty"]))
bn = d[(d.date.between("2023-01-19", "2023-02-06")) &
       d.vol_bound_banknifty.eq(True)]
test("BANKNIFTY 2023 exceptions are the explicit five dates",
     list(bn.date.dt.strftime("%Y-%m-%d")) ==
     ["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2023-02-06"])

#participant records
oi = bp.load(RAW / "participant_oi.csv", "participant OI")
valid = bp.build_shares(oi)
valid_dates = set(valid.date)
test("22-Aug-2013 fails participant validation", pd.Timestamp("2013-08-22") not in valid_dates)
test("12-Dec-2019 fails participant validation", pd.Timestamp("2019-12-12") not in valid_dates)
blank = oi[(oi.date == "2019-12-12") & (oi.client_type.astype(str).str.strip() == "FII")]
idx = [c for c in bp.IDX_LONG + bp.IDX_SHORT if c in blank]
test("blank participant cells do not sum to zero",
     blank[idx].sum(axis=1, min_count=1).isna().all())

# price base and lot dating
basis = d.dropna(subset=["psr_nifty", "psr_spot_nifty"])
test("previous-close and futures-base PSR differ on basis-sensitive dates",
     (basis.psr_nifty - basis.psr_spot_nifty).abs().max() > .10)
plateau = r.loc["2022-12-06", "psr_spot_nifty"]
test("identified NIFTY floor matches 9.30 within source precision", abs(plateau - 9.30) <= .005)
bh = pd.read_csv(RAW / "bhav_index.csv", low_memory=False, parse_dates=["date"])
lots = bp.add_lot_size(bh)
nov = lots[(lots.symbol == "NIFTY") & (lots.expiry_dt == "2014-11-27")].set_index("date")
test("Nov-2014 NIFTY lot changes inside one contract",
     nov.loc["2014-10-30", "lot_inferred"] == 50 and nov.loc["2014-10-31", "lot_inferred"] == 25)

# weights and lags
cal = pd.DatetimeIndex(d.date)
lb = bp.add_effective_expiry(bp.add_lot_size(bh[bh.symbol.isin(["NIFTY", "BANKNIFTY"])]), cal)
src = lb[lb.date == "2020-01-30"]
alive = src[src.effective_expiry_dt >= pd.Timestamp("2020-01-31")]
expected = alive.groupby("symbol").oi_contracts.sum()
expected = expected / expected.sum()
test("expiry-survival weight excludes the just-expired contract",
     abs(r.loc["2020-01-31", "w_nifty"] - expected.NIFTY) < 1e-12)
test("first aggregate margin has no fabricated weights",
     pd.isna(r.loc["2012-07-05", "margin"]) and pd.isna(r.loc["2012-07-05", "w_nifty"]))
w = d.dropna(subset=["w_nifty", "w_banknifty"])
test("defined benchmark weights sum to one", np.allclose(w.w_nifty + w.w_banknifty, 1))

pos = pd.Series(np.arange(len(cal)), index=cal)
test("five-day outcome endpoints are five exchange trading days apart",
     (pos.shift(-4) - pos.shift(1)).dropna().eq(5).all())

# the verifier itself 
verifier = ROOT / "verify_paper_numbers.py"
runner = ("from pathlib import Path; p=Path(" + repr(str(verifier)) + "); "
          "s=p.read_text().replace('check(\"daily panel rows\", 3469, len(d))',"
          "'check(\"daily panel rows\", 9999, len(d))',1); "
          "g={'__file__':str(p),'__name__':'__main__'}; exec(compile(s,str(p),'exec'),g)")
bad = subprocess.run([sys.executable, "-c", runner], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
test("verifier exits non-zero after expected-value perturbation", bad.returncode != 0)

# manuscript, when present 
if PAPER.exists():
    tex = PAPER.read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:ref|pageref|autoref)\{([^}]+)\}", tex))
    test("all manuscript labels are unique", len(labels) == len(set(labels)))
    test("every manuscript label is referenced", set(labels) <= refs)
    test("no unresolved manuscript placeholder remains",
         re.search(r"\[RERUN\]|TODO|FIXME|\?\?\?|PLACEHOLDER", tex, re.I) is None)
else:
    print("SKIP    : manuscript checks (paper.tex not in this repository)")

# structural
test("aggregate margin is missing after the bhavcopy gap", pd.isna(r.loc["2013-10-10", "margin"]))
test("unclassifiable observations carry a reason",
     d.loc[d.vol_bound.isna(), "unclassified_reason"].astype(bool).all())
test("retained lot estimate independently matches published overlap",
     (lots.dropna(subset=["lot_inferred", "stated_lot"])
      .eval("lot_inferred == stated_lot").all()))
test("lot-transition exposure set contains 341 trading days",
     len(__import__("incidence_regressions").lot_transition_exposure_dates()) == 341)

print(f"\n{passed} ADVERSARIAL TESTS PASSED")
