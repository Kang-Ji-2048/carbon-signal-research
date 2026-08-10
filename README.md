# Does the carbon price predict green vs. brown equities?

A reproducible quantitative study testing one hypothesis on real market data:

> When carbon allowances get more expensive, do clean-energy equities subsequently
> outperform fossil-fuel equities?

**The answer, on this sample, is no** — and the more interesting result is *how
easy it would have been to claim otherwise*. See [Findings](#findings).

Live dashboard: `python app.py` · Static build: `python build_static.py`

---

## Findings

**1. The predictive relationship is not there.**
Regressing next-day green-minus-brown (GMB) returns on 60-day carbon momentum:

| | |
|---|---|
| Coefficient | −0.0050 |
| t-statistic (Newey–West) | **−0.74** |
| p-value | 0.46 |
| R² | 0.0006 |

Standard errors are Newey–West corrected, because overlapping momentum windows and
daily returns are autocorrelated — plain OLS errors would overstate significance.

**2. The apparent out-of-sample performance is a parameter artefact.**
This is the finding worth defending in an interview. Out-of-sample Sharpe across the
lookback × cost grid:

| Lookback | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| 20d | 0.43 | 0.28 | 0.12 | −0.19 |
| 40d | 0.49 | 0.34 | 0.19 | −0.11 |
| 60d | 0.28 | 0.19 | 0.11 | −0.06 |
| 90d | 0.07 | −0.05 | −0.17 | −0.41 |
| **120d** | **1.02** | **0.90** | **0.77** | **0.52** |

Reporting the 120-day row alone would have produced a headline **OOS Sharpe of 0.90**
— a genuinely impressive-looking number, obtained purely by choosing one of five
arbitrary lookbacks after seeing the results. The full-sample Sharpe at that same
setting is −0.08. A result that swings from −0.41 to +1.02 on a parameter choice is a
property of the parameter, not of the market.

Nothing survives 20 bps of cost except the one cherry-picked cell.

**3. In-sample and out-of-sample disagree**, which is what noise looks like:

| Period | Window | Obs | Ann. return | Ann. vol | Sharpe (net) | Max DD | Ann. turnover |
|---|---|---|---|---|---|---|---|
| In-sample | 2020-10-27 → 2023-02-15 | 580 | −6.12% | 10.42% | −0.55 | −25.6% | 3.7 |
| Out-of-sample | 2023-02-16 → 2026-08-10 | 872 | +1.47% | 10.47% | +0.19 | −18.4% | 8.9 |
| Full sample | 2020-10-27 → 2026-08-10 | 1,452 | −1.63% | 10.45% | −0.11 | −26.8% | 6.8 |
| Buy & hold GMB (OOS) | — | 872 | −18.66% | 35.56% | −0.40 | −62.4% | — |

**4. What did work.** Two components behaved exactly as designed, independent of the
signal's failure:

- **Volatility targeting.** Realised strategy volatility was **10.45%** against a 10%
  target, and held at ~10.4% in *both* sub-periods while the underlying factor ran at
  35.6% vol. Risk control worked even though the alpha did not.
- **The factor is genuinely distinct.** GMB's correlation to the S&P 500 is **0.067**
  (beta 0.12, R² 0.004) — it is not repackaged market beta.

For context, green has badly underperformed brown over the full 2010–2026 history:
GMB returned **−9.7% annualised** with a −92% maximum drawdown.

---

## Method

**Universe** — daily adjusted closes from Yahoo Finance:

| Role | Tickers |
|---|---|
| Green | ICLN, TAN, PBW |
| Brown | XLE, XOP |
| Carbon | KRBN (tracks EU ETS / CCA / RGGI allowance futures) |
| Benchmark | SPY |

**Factor** — `GMB = mean(green returns) − mean(brown returns)`, equal-weighted, daily
rebalanced. Simple (not log) returns, because portfolios aggregate linearly.

**Signal** — long GMB when 60-day carbon momentum is positive, short when negative,
scaled to a 10% annualised vol target using *trailing* realised volatility, capped at
2× leverage.

**Backtest** — a position formed at the close of day *t* earns the day *t+1* return.
Costs of 5 bps per leg are charged on turnover (the spread trades two legs). The
sample is split chronologically 40/60 into in-sample and out-of-sample.

### Guardrails

These are the substance of the project, not decoration:

- **No lookahead.** The one-day lag lives in exactly one place (`research/backtest.py`)
  so it can be tested directly. `tests/test_backtest.py::TestNoLookahead` overwrites
  all data after a cut date with garbage and asserts that P&L *before* the cut is
  bit-for-bit unchanged — the test that catches centred windows, backfills and missing
  shifts.
- **Forward-fill only, never backward.** Backfilling a price gap imports tomorrow's
  price into today. Gaps longer than 3 days are left as NaN rather than invented.
- **Costs and turnover always reported.** No gross-only results.
- **HAC standard errors** on every regression.
- **The whole parameter grid is published**, not the best cell.

### Honest limitations

- **Short sample.** KRBN launched 2020-07-31, so the carbon test has ~1,450 trading
  days. That is thin for any strategy claim, and is why the null result is stated as
  "not supported on this sample" rather than "does not exist."
- **KRBN is a proxy**, not a spot carbon price — it holds allowance futures and carries
  roll cost.
- **ETFs are a proxy** for green/brown exposure and carry sector and factor tilts that
  are not decomposed here.
- **One signal, one hypothesis.** No claim is made about carbon markets in general.

---

## Running it

```bash
pip install -r requirements.txt
python app.py --refresh        # fetch data, run ETL + research, launch dashboard
```

| Command | Effect |
|---|---|
| `python app.py` | Dashboard from cached results |
| `python app.py --refresh` | Re-download prices, rerun everything |
| `python app.py --research-only` | Print headline statistics, no dashboard |
| `python build_static.py` | Emit a static site to `dist/` |
| `python -m pytest tests/ -q` | Run the test suite (27 tests) |

Market data is cached to `data/raw/prices.csv`, so every run after the first is
deterministic and offline.

> **TLS note.** On machines with TLS interception (corporate AV/proxy), `yfinance`
> fails to verify certificates because it fetches via `curl_cffi`, which ignores
> `truststore`'s patch of Python's `ssl` module. `etl/ingest.py` handles this by
> exporting the Windows certificate store, combining it with certifi, and pointing
> `CURL_CA_BUNDLE` at the result.

## Layout

```
etl/         ingest (cached, TLS-aware) -> clean (missing-data policy) -> aggregate (factor)
research/    signal.py (causal by construction) | backtest.py (the lag + costs)
             stats.py (metrics, Newey-West)     | report.py (orchestration -> JSON)
dashboard/   Dash layer; figures shared with the static build
tests/       27 tests: P&L accounting, no-lookahead, stats, missing-data policy
docs/        design spec
```

## Tech

Python 3.13 · pandas · numpy · statsmodels (HAC) · Plotly / Dash · yfinance · pytest
