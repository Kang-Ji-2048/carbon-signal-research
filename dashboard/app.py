"""Dash presentation layer for the carbon-signal study.

The layout deliberately leads with the hypothesis test and the sensitivity grid
rather than the equity curve. An equity curve alone invites the reader to
believe a result the statistics do not support.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html

GREEN = "#2e9e6b"
RED = "#d1495b"
BLUE = "#3d5a80"
GREY = "#8d99ae"
TEMPLATE = "plotly_white"


def _fmt_pct(x, dp=2):
    return "n/a" if x is None else f"{x * 100:.{dp}f}%"


def _fmt_num(x, dp=2):
    return "n/a" if x is None else f"{x:.{dp}f}"


def kpi_card(title: str, value: str, subtitle: str, colour: str = BLUE) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted", style={"fontSize": "0.8rem", "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "1.9rem", "fontWeight": 600, "color": colour}),
            html.Div(subtitle, className="text-muted", style={"fontSize": "0.78rem"}),
        ]),
        className="shadow-sm h-100",
    )


def equity_figure(report: dict) -> go.Figure:
    s = report["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["dates"], y=s["strategy_equity"], name="Carbon-momentum strategy (net)",
        line=dict(color=BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=s["dates"], y=s["gmb_equity"], name="Buy & hold green-minus-brown",
        line=dict(color=GREY, width=1.6, dash="dash"),
    ))

    oos_start = report["performance"]["out_of_sample"].get("start")
    if oos_start:
        fig.add_vline(x=oos_start, line=dict(color=RED, width=1.4, dash="dot"))
        fig.add_annotation(x=oos_start, y=1, yref="paper", text="  out-of-sample →",
                           showarrow=False, xanchor="left", font=dict(color=RED, size=11))

    fig.update_layout(
        template=TEMPLATE, height=420, title="Growth of 1 unit (net of transaction costs)",
        yaxis_title="Cumulative growth", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18), margin=dict(t=60, b=40),
    )
    return fig


def drawdown_figure(report: dict) -> go.Figure:
    s = report["series"]
    fig = go.Figure(go.Scatter(
        x=s["dates"], y=s["drawdown"], fill="tozeroy",
        line=dict(color=RED, width=1), name="Drawdown",
    ))
    fig.update_layout(
        template=TEMPLATE, height=260, title="Strategy drawdown",
        yaxis_tickformat=".0%", margin=dict(t=50, b=30), showlegend=False,
    )
    return fig


def position_figure(report: dict) -> go.Figure:
    s = report["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["dates"], y=s["position"], name="Position (vol-scaled)",
        line=dict(color=BLUE, width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=s["dates"], y=s["carbon_momentum"], name="Carbon momentum (60d)",
        line=dict(color=GREEN, width=1.2), yaxis="y2",
    ))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.update_layout(
        template=TEMPLATE, height=320, title="Signal and resulting position",
        yaxis=dict(title="Position"),
        yaxis2=dict(title="Carbon momentum", overlaying="y", side="right", tickformat=".0%"),
        hovermode="x unified", legend=dict(orientation="h", y=-0.22),
        margin=dict(t=60, b=40),
    )
    return fig


def sensitivity_figure(report: dict) -> go.Figure:
    """The heart of the study: how much does the result depend on the knobs?"""
    df = pd.DataFrame(report["sensitivity"])
    pivot = df.pivot(index="lookback", columns="cost_bps", values="oos_sharpe")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{c:.0f} bps" for c in pivot.columns],
        y=[f"{i}d" for i in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=pivot.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="OOS<br>Sharpe"),
    ))
    fig.update_layout(
        template=TEMPLATE, height=360,
        title="Out-of-sample Sharpe across every parameter choice",
        xaxis_title="Transaction cost assumption", yaxis_title="Carbon momentum lookback",
        margin=dict(t=60, b=40),
    )
    return fig


def performance_table(report: dict) -> dash_table.DataTable:
    rows = []
    labels = {
        "in_sample": "In-sample", "out_of_sample": "Out-of-sample",
        "full_sample": "Full sample",
    }
    for key, label in labels.items():
        d = report["performance"].get(key, {})
        if not d:
            continue
        rows.append({
            "Period": label,
            "Window": f"{d.get('start')} → {d.get('end')}",
            "Obs": d.get("n_obs"),
            "Ann. return": _fmt_pct(d.get("ann_return")),
            "Ann. vol": _fmt_pct(d.get("ann_vol")),
            "Sharpe (net)": _fmt_num(d.get("sharpe"), 3),
            "Sharpe (gross)": _fmt_num(d.get("gross_sharpe"), 3),
            "Max DD": _fmt_pct(d.get("max_drawdown")),
            "Ann. turnover": _fmt_num(d.get("ann_turnover"), 1),
        })

    bh = report["performance"].get("buy_and_hold_gmb_oos", {})
    rows.append({
        "Period": "Buy & hold GMB (OOS)", "Window": "—", "Obs": bh.get("n_obs"),
        "Ann. return": _fmt_pct(bh.get("ann_return")), "Ann. vol": _fmt_pct(bh.get("ann_vol")),
        "Sharpe (net)": _fmt_num(bh.get("sharpe"), 3), "Sharpe (gross)": "—",
        "Max DD": _fmt_pct(bh.get("max_drawdown")), "Ann. turnover": "—",
    })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0]],
        style_cell={"fontSize": "0.85rem", "padding": "8px", "fontFamily": "system-ui"},
        style_header={"fontWeight": 600, "backgroundColor": "#f1f3f5"},
        style_data_conditional=[{
            "if": {"filter_query": "{Period} = 'Out-of-sample'"},
            "backgroundColor": "#eef4ff", "fontWeight": 600,
        }],
    )


def findings_panel(report: dict) -> dbc.Card:
    t = report["predictive_test"]
    sens = pd.DataFrame(report["sensitivity"])
    spread = f"{sens.oos_sharpe.min():.2f} to {sens.oos_sharpe.max():.2f}"

    return dbc.Card(dbc.CardBody([
        html.H5("What the data says", className="mb-3"),
        html.P([
            html.Strong("The hypothesis is not supported. "),
            f"Regressing next-day green-minus-brown returns on 60-day carbon momentum gives "
            f"a t-statistic of {t['t_stat']:.2f} (p = {t['p_value']:.2f}) and an R² of "
            f"{t['r_squared']:.4f}. That is indistinguishable from no relationship.",
        ]),
        html.P([
            "Out-of-sample Sharpe across the parameter grid ranges from ",
            html.Strong(spread),
            ". A result that flips sign depending on an arbitrary lookback choice is a "
            "property of the parameter, not of the market. Reporting only the best cell "
            "would have produced a far more flattering — and entirely unjustified — number.",
        ]),
        html.P([
            html.Strong("What did work: "),
            "volatility targeting. Realised strategy volatility came in at ",
            html.Strong(_fmt_pct(report["performance"]["full_sample"]["ann_vol"])),
            f" against a {_fmt_pct(report['parameters']['target_vol'], 0)} target, and the "
            "factor's correlation to the S&P 500 is just "
            f"{report['factor']['vs_market_full']['correlation']:.3f} — so green-minus-brown "
            "is a genuinely distinct exposure rather than repackaged market beta.",
        ], className="mb-0"),
    ]), className="shadow-sm border-start border-4", style={"borderColor": RED})


def create_layout(report: dict) -> html.Div:
    perf = report["performance"]
    oos, full = perf["out_of_sample"], perf["full_sample"]
    t = report["predictive_test"]
    sample = report["sample"]

    return dbc.Container([
        html.Div([
            html.H2("Does the carbon price predict green vs. brown equities?", className="mt-4 mb-1"),
            html.P([
                f"Daily data {sample['backtest_start']} → {sample['backtest_end']} "
                f"({sample['backtest_obs']:,} observations). Long/short "
                f"{'/'.join(report['universe']['green'])} versus "
                f"{'/'.join(report['universe']['brown'])}, signalled by "
                f"{report['universe']['carbon']}.",
            ], className="text-muted"),
        ]),

        dbc.Row([
            dbc.Col(kpi_card("Out-of-sample Sharpe", _fmt_num(oos["sharpe"], 2),
                             "net of costs", GREY), md=3),
            dbc.Col(kpi_card("Predictive t-stat", _fmt_num(t["t_stat"], 2),
                             f"p = {t['p_value']:.2f} — not significant", RED), md=3),
            dbc.Col(kpi_card("Realised vol", _fmt_pct(full["ann_vol"], 1),
                             f"vs {_fmt_pct(report['parameters']['target_vol'], 0)} target", GREEN), md=3),
            dbc.Col(kpi_card("Correlation to S&P", _fmt_num(report["factor"]["vs_market_full"]["correlation"], 3),
                             "a distinct factor", BLUE), md=3),
        ], className="g-3 my-2"),

        dbc.Row(dbc.Col(findings_panel(report)), className="my-3"),

        dbc.Row(dbc.Col(dcc.Graph(figure=sensitivity_figure(report))), className="my-2"),
        dbc.Row(dbc.Col(dcc.Graph(figure=equity_figure(report))), className="my-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=drawdown_figure(report)), md=6),
            dbc.Col(dcc.Graph(figure=position_figure(report)), md=6),
        ], className="my-2"),

        html.H5("Performance detail", className="mt-4 mb-2"),
        performance_table(report),

        html.Hr(className="mt-4"),
        html.P([
            f"Generated {report['generated']}. Positions are formed at the close of day t and "
            f"earn day t+1 returns; costs of {report['parameters']['cost_bps']:.0f} bps per leg "
            f"are charged on turnover. Carbon-allowance data begins at KRBN's 2020 inception, so "
            f"the sample is short — roughly {sample['backtest_obs']:,} trading days — and all "
            f"conclusions are stated with that limitation in mind.",
        ], className="text-muted", style={"fontSize": "0.8rem"}),
    ], fluid=True, style={"maxWidth": "1320px", "paddingBottom": "3rem"})


def create_app(report: dict) -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
               title="Carbon Signal Research")
    app.layout = create_layout(report)
    return app
