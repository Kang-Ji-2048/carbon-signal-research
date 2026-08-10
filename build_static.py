"""Export a self-contained static site for Netlify.

The figures are produced by the same functions the live Dash app uses, so the
static build cannot drift from the interactive one. Only the shell around them
differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import plotly.io as pio

import config
from dashboard.app import (
    drawdown_figure,
    equity_figure,
    position_figure,
    sensitivity_figure,
)
from research.report import load_report, save_report

log = logging.getLogger(__name__)

DIST = config.ROOT / "dist"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does the carbon price predict green vs. brown equities?</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="stylesheet" href="css/style.css">
</head>
<body>
<main>
  <header>
    <h1>Does the carbon price predict green vs. brown equities?</h1>
    <p class="sub">{subtitle}</p>
  </header>

  <section class="kpis">{kpis}</section>

  <section class="callout">
    <h2>What the data says</h2>
    <p><strong>The hypothesis is not supported.</strong> Regressing next-day
    green-minus-brown returns on {lookback}-day carbon momentum gives a
    t-statistic of {tstat:.2f} (p = {pval:.2f}) and an R&sup2; of {r2:.4f}.
    That is indistinguishable from no relationship.</p>
    <p>Out-of-sample Sharpe across the parameter grid ranges from
    <strong>{smin:.2f} to {smax:.2f}</strong>. A result that flips sign
    depending on an arbitrary lookback choice is a property of the parameter,
    not of the market. Reporting only the best cell would have produced a far
    more flattering &mdash; and entirely unjustified &mdash; number.</p>
    <p><strong>What did work:</strong> volatility targeting. Realised strategy
    volatility came in at {realvol:.2%} against a {targetvol:.0%} target, and the
    factor&rsquo;s correlation to the S&amp;P 500 is just {corr:.3f} &mdash; so
    green-minus-brown is a genuinely distinct exposure rather than repackaged
    market beta.</p>
  </section>

  {charts}

  <section>
    <h2>Performance detail</h2>
    {table}
  </section>

  <footer>
    <p>Generated {generated}. Positions are formed at the close of day <em>t</em>
    and earn day <em>t+1</em> returns; costs of {cost:.0f} bps per leg are charged
    on turnover. Carbon-allowance data begins at KRBN&rsquo;s 2020 inception, so
    the sample is short &mdash; roughly {nobs:,} trading days &mdash; and all
    conclusions are stated with that limitation in mind.</p>
  </footer>
</main>
</body>
</html>
"""

CSS = """:root {
  --ink: #1d2733; --muted: #6b7785; --line: #e3e8ee;
  --blue: #3d5a80; --red: #d1495b; --green: #2e9e6b;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: #f7f9fb; color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.55;
}
main { max-width: 1180px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.85rem; margin: 0 0 .4rem; line-height: 1.25; }
h2 { font-size: 1.1rem; margin: 0 0 .75rem; }
.sub { color: var(--muted); margin: 0 0 2rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; }
.kpi {
  background: #fff; border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem 1.15rem;
}
.kpi .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.kpi .value { font-size: 1.8rem; font-weight: 600; line-height: 1.2; }
.kpi .note { font-size: .78rem; color: var(--muted); }
.callout {
  background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--red);
  border-radius: 10px; padding: 1.25rem 1.4rem; margin: 1.75rem 0;
}
.callout p { margin: 0 0 .8rem; }
.callout p:last-child { margin-bottom: 0; }
.chart {
  background: #fff; border: 1px solid var(--line); border-radius: 10px;
  padding: .5rem; margin: 1.25rem 0;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }
.table-wrap { overflow-x: auto; background: #fff; border: 1px solid var(--line); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; min-width: 720px; }
th, td { padding: .6rem .8rem; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f1f3f5; font-weight: 600; }
tbody tr.highlight { background: #eef4ff; font-weight: 600; }
footer { margin-top: 2.5rem; color: var(--muted); font-size: .82rem; border-top: 1px solid var(--line); padding-top: 1rem; }
"""


def _kpi(label: str, value: str, note: str, colour: str) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value" style="color:{colour}">{value}</div>'
        f'<div class="note">{note}</div></div>'
    )


def _table(report: dict) -> str:
    labels = {"in_sample": "In-sample", "out_of_sample": "Out-of-sample", "full_sample": "Full sample"}
    head = ["Period", "Window", "Obs", "Ann. return", "Ann. vol", "Sharpe (net)",
            "Sharpe (gross)", "Max DD", "Ann. turnover"]
    body = ""
    for key, label in labels.items():
        d = report["performance"].get(key, {})
        if not d:
            continue
        cls = ' class="highlight"' if key == "out_of_sample" else ""
        body += (
            f"<tr{cls}><td>{label}</td><td>{d['start']} → {d['end']}</td><td>{d['n_obs']}</td>"
            f"<td>{d['ann_return']:.2%}</td><td>{d['ann_vol']:.2%}</td>"
            f"<td>{d['sharpe']:.3f}</td><td>{d['gross_sharpe']:.3f}</td>"
            f"<td>{d['max_drawdown']:.2%}</td><td>{d['ann_turnover']:.1f}</td></tr>"
        )
    bh = report["performance"]["buy_and_hold_gmb_oos"]
    body += (
        f"<tr><td>Buy &amp; hold GMB (OOS)</td><td>—</td><td>{bh['n_obs']}</td>"
        f"<td>{bh['ann_return']:.2%}</td><td>{bh['ann_vol']:.2%}</td>"
        f"<td>{bh['sharpe']:.3f}</td><td>—</td><td>{bh['max_drawdown']:.2%}</td><td>—</td></tr>"
    )
    header = "".join(f"<th>{h}</th>" for h in head)
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def build(refresh: bool = False, rerun: bool = False) -> Path:
    report = save_report(refresh=refresh) if (refresh or rerun) else load_report()

    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "css").mkdir(parents=True, exist_ok=True)
    (DIST / "data").mkdir(parents=True, exist_ok=True)

    def div(fig, first: bool) -> str:
        # Without autosize + responsive, the static export bakes in Plotly's
        # default 700px width and the page scrolls sideways on narrow screens.
        fig.update_layout(autosize=True, width=None)
        return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                           config={"displayModeBar": False, "responsive": True},
                           default_width="100%")

    charts = (
        f'<div class="chart">{div(sensitivity_figure(report), True)}</div>'
        f'<div class="chart">{div(equity_figure(report), False)}</div>'
        '<div class="grid2">'
        f'<div class="chart">{div(drawdown_figure(report), False)}</div>'
        f'<div class="chart">{div(position_figure(report), False)}</div>'
        '</div>'
    )

    perf, t = report["performance"], report["predictive_test"]
    sens = pd.DataFrame(report["sensitivity"])
    u, s = report["universe"], report["sample"]

    kpis = "".join([
        _kpi("Out-of-sample Sharpe", f"{perf['out_of_sample']['sharpe']:.2f}", "net of costs", "#6b7785"),
        _kpi("Predictive t-stat", f"{t['t_stat']:.2f}", f"p = {t['p_value']:.2f} — not significant", "#d1495b"),
        _kpi("Realised vol", f"{perf['full_sample']['ann_vol']:.1%}",
             f"vs {report['parameters']['target_vol']:.0%} target", "#2e9e6b"),
        _kpi("Correlation to S&P", f"{report['factor']['vs_market_full']['correlation']:.3f}",
             "a distinct factor", "#3d5a80"),
    ])

    html = PAGE.format(
        subtitle=(f"Daily data {s['backtest_start']} → {s['backtest_end']} "
                  f"({s['backtest_obs']:,} observations). Long/short "
                  f"{'/'.join(u['green'])} versus {'/'.join(u['brown'])}, "
                  f"signalled by {u['carbon']}."),
        kpis=kpis, charts=charts, table=_table(report),
        lookback=t["lookback"], tstat=t["t_stat"], pval=t["p_value"], r2=t["r_squared"],
        smin=sens.oos_sharpe.min(), smax=sens.oos_sharpe.max(),
        realvol=perf["full_sample"]["ann_vol"], targetvol=report["parameters"]["target_vol"],
        corr=report["factor"]["vs_market_full"]["correlation"],
        generated=report["generated"], cost=report["parameters"]["cost_bps"],
        nobs=s["backtest_obs"],
    )

    (DIST / "index.html").write_text(html, encoding="utf-8")
    (DIST / "css" / "style.css").write_text(CSS, encoding="utf-8")
    with open(DIST / "data" / "results.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log.info("Static site written to %s", DIST)
    return DIST


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ap = argparse.ArgumentParser(description="Build the static site")
    ap.add_argument("--refresh", action="store_true", help="Re-download data, rerun research")
    ap.add_argument("--rerun", action="store_true", help="Rerun research using cached prices")
    args = ap.parse_args()
    out = build(refresh=args.refresh, rerun=args.rerun)
    print(f"\nBuilt {out}\nPreview: cd dist && python -m http.server 8080")
