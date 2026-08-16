Replication code for The Discretionary Floor: Regime-Switching Margin Procyclicality in Indian Index Derivatives, Prajeet Singh and Vatsal Vishwam (BITS Pilani).

The paper reconstructs the NSE Clearing SPAN price scan range for NIFTY and BANKNIFTY over 5 July 2012 to 31 July 2026 and classifies every trading day as
volatility bound or floor bound. Everything in the paper is rebuilt from the parsed exchange files in `data/` by the scripts here.

Language: Python 3.11
Libraries Required: `pandas`, `numpy`, `matplotlib`, and `requests`

Steps to follow: 
```bash
pip install -r requirements.txt
```

The Newey-West and Driscoll-Kraay estimators are written out in `incidence_regressions.py` rather than pulled from a library, so 
they can be read and checked. Developed on pandas 3.0 and numpy 2.5.

## Reproducing the paper

```bash
python build_panel.py
python classify_binding.py
python make_figures.py
```

`build_panel.py` and `classify_binding.py` write to `panel/`, `make_figures.py`
writes to `figures/`; both directories are generated and are not tracked here.
`incidence_regressions.py` prints the Section 8 tables and can be run on its
own once the panel exists.

## Files

| `nse_pipeline.py` | downloads and parses the NSE archives into `data/` |
| `build_panel.py` | merges the sources into a daily panel and a participant panel |
| `classify_binding.py` | classifies each day as volatility-bound or floor-bound |
| `incidence_regressions.py` | Section 8 participant incidence regressions |
| `make_figures.py` | the five figures |
| `paths.py` | one place for directory locations |

## Data

`data/` holds the parsed exchange datasets, one tidy CSV per source:
participant-wise open interest, FOVOLT volatility, the F&O bhavcopy, SPAN
price scan ranges, and the India VIX. The original downloads, in particular
the 15 to 60 MB SPAN XML archives, are discarded during parsing and are not
redistributed.

`nse_pipeline.py` documents and automates that acquisition, so the chain from
the exchange's published files to `data/` is reproducible. It does not need to
be run. It hits live NSE endpoints, and its output is already here. To collect
afresh, send it somewhere else:

```bash
python nse_pipeline.py --start 2012-07-05 --end 2026-07-31 --outdir nse_data
```

It keeps progress per (date, source) in `_state.json`, so a transient failure
on one endpoint is retried on the next run instead of being masked by the
other three succeeding that day.

It retains five index symbols from the bhavcopy, which is **not** the complete
set NSE has listed. Several products traded early in the sample and were later
discontinued. Coverage claims are therefore benchmarked against the
participant report's TOTAL row, not against the sum over these five.

## Choices that matter

**Price base.** Scan ranges are expressed as a percentage of the previous
index close, not the futures settlement price. That is the base on which the
implemented floor is exact.

**Open interest is in underlying units.** The bhavcopy field is contracts
times the board lot, not a contract count, so the weights divide it by the
lot. From 2021 on, OI/lot summed over the five collected symbols agrees with
the participant TOTAL on every timestamp-comparable date to source precision.

**The lot is not fixed per contract.** Some revisions are phased in by expiry;
others re-specify every outstanding contract on one date, as on 31 October
2014, when the NIFTY November contract went from 50 to 25 mid-life. The lot is
`NewBrdLotQty` where published and otherwise a centred rolling median of the
implied lot within each contract's series, snapped to the nearest multiple of
five. The independent estimate matches `NewBrdLotQty` on all 7,632 rows where
both exist.

**Lagged, expiry-surviving weights.** The two-index average uses the
immediately previous exchange trading day's open interest and counts only
contracts still open on the day the weight applies. Same-day weights would let
day *t* open interest, the outcome variable of Section 8, move the day *t*
explanatory series mechanically. No stale bhavcopy is carried across a gap, so
the first sample day and 10 October 2013 have no weight at all.

**Participant validation.** Every date is checked against the file's own TOTAL
row. Two records fail: 2019-12-12, whose FII row is blank, and 2013-08-22,
whose rows are displaced by one label so the market total sits in the
proprietary row. Both dates are dropped, leaving 13,856 participant-days on
3,464 dates.

**Two classification rules.** Where a plateau identifies the floor, the
comparison is exact: floor-bound where the scan range equals the prevailing
floor within 0.005 pp, volatility-bound where it exceeds it. Where no plateau
appears, no floor can be identified and the implied multiplier is compared
against theory with a two-sided 2% band instead. On 1 June 2020 the formula
and the volatility estimator changed together, so on that date alone the
classification uses same-day volatility; `k` keeps its lagged definition
everywhere.

Daily changes are computed only where day *t* and the immediately preceding
trading day both carry the relevant source, so no difference spans a gap. Of
the 3,466 dates carrying SPAN parameters, three cannot be classified:
2012-07-05 (first day, no lag), 2021-03-31 (no index close) and 2021-04-01 (no
lagged volatility). Regime counts sum to 3,463. The daily panel has 3,469
rows, the extra three being the 2025 dates with no SPAN file at all.
