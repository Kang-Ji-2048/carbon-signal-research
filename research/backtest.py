"""Backtest engine: turn positions into P&L, honestly.

Two rules are enforced here and nowhere else, which is what makes them testable:

1. A position decided at the close of day ``t`` earns the return of day ``t+1``.
2. Changing the position costs money, charged on turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from . import stats
from .signal import build_signal


def run_backtest(
    panel: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = config.COST_BPS,
    legs: int = config.LEGS,
) -> pd.DataFrame:
    """Apply ``positions`` to the GMB factor and return a per-day P&L frame.

    The single ``shift(1)`` below is the entire lookahead defence: the position
    column is moved forward one day before it ever meets a return, so today's
    P&L can only ever be driven by yesterday's decision.
    """
    df = pd.DataFrame(index=panel.index)
    df["gmb"] = panel["gmb"]

    held = positions.reindex(panel.index).shift(1)
    df["position"] = held

    df["gross_return"] = held * df["gmb"]

    # Turnover is the change in position between consecutive days. The spread
    # holds two legs, so trading one unit of it transacts `legs` units of stock.
    turnover = held.diff().abs()
    df["turnover"] = turnover
    df["cost"] = turnover * (cost_bps / 1e4) * legs
    df["net_return"] = df["gross_return"] - df["cost"].fillna(0.0)

    return df.dropna(subset=["gross_return"])


def split_sample(df: pd.DataFrame, oos_split: float = config.OOS_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological in-sample / out-of-sample split."""
    cut = int(len(df) * oos_split)
    return df.iloc[:cut], df.iloc[cut:]


def evaluate(df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Summarise a backtest frame, in-sample and out-of-sample."""
    is_df, oos_df = split_sample(df)

    def block(d: pd.DataFrame) -> dict:
        if d.empty:
            return {}
        s = stats.summarise(d["net_return"], positions=d["position"])
        s["gross_sharpe"] = stats.sharpe(d["gross_return"])
        s["cost_drag_ann"] = float(d["cost"].mean() * config.TRADING_DAYS)
        s["start"] = str(d.index.min().date())
        s["end"] = str(d.index.max().date())
        return s

    bench = panel["gmb"].reindex(df.index)
    result = {
        "full_sample": block(df),
        "in_sample": block(is_df),
        "out_of_sample": block(oos_df),
        "buy_and_hold_gmb": stats.summarise(bench),
        "buy_and_hold_gmb_oos": stats.summarise(bench.reindex(oos_df.index)),
    }

    oos_bench = panel["benchmark"].reindex(oos_df.index)
    result["oos_vs_market"] = stats.beta_to(oos_df["net_return"], oos_bench)
    return result


def sensitivity_sweep(
    panel: pd.DataFrame,
    lookbacks: list[int] | None = None,
    costs: list[float] | None = None,
) -> pd.DataFrame:
    """Grid of out-of-sample Sharpe across lookbacks and cost assumptions.

    Reported in full rather than as a best case. A signal whose result survives
    only at one lookback and zero costs has not been demonstrated at all.
    """
    lookbacks = lookbacks or config.SWEEP_LOOKBACKS
    costs = costs or config.SWEEP_COSTS

    rows = []
    for lb in lookbacks:
        sig = build_signal(panel, lookback=lb)
        for c in costs:
            bt = run_backtest(panel, sig["position"], cost_bps=c)
            if bt.empty:
                continue
            _, oos = split_sample(bt)
            rows.append({
                "lookback": lb,
                "cost_bps": c,
                "oos_sharpe": stats.sharpe(oos["net_return"]),
                "oos_ann_return": stats.annualised_return(oos["net_return"]),
                "full_sharpe": stats.sharpe(bt["net_return"]),
            })
    return pd.DataFrame(rows)
