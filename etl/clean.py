"""Calendar alignment and return computation.

The missing-data policy matters more than it looks. Filling gaps in the wrong
direction is the classic way to leak future information into a backtest, so the
rules are explicit and narrow.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

MAX_FILL_DAYS = 3


def align_calendar(prices: pd.DataFrame, max_fill: int = MAX_FILL_DAYS) -> pd.DataFrame:
    """Align to a common trading calendar.

    Gaps are filled *forward* only (a stale price is the last thing we actually
    knew) and never backward, which would import tomorrow's price into today.
    Runs longer than ``max_fill`` days are left as NaN rather than invented, and
    leading NaNs before an asset's inception are preserved so that its history
    never appears to start early.
    """
    df = prices.sort_index().copy()
    df = df[~df.index.duplicated(keep="last")]
    filled = df.ffill(limit=max_fill)

    # Restore pre-inception NaNs: ffill cannot create them, but being explicit
    # documents that an asset contributes nothing before it existed.
    for col in filled.columns:
        first = df[col].first_valid_index()
        if first is not None:
            filled.loc[filled.index < first, col] = pd.NA

    return filled.astype("float64")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns.

    Simple (not log) returns are used because portfolios aggregate linearly
    across holdings: the return of an equal-weighted basket is the mean of its
    constituents' simple returns, which is not true of log returns.
    """
    returns = prices.pct_change()
    return returns.iloc[1:]


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(aligned_prices, returns)``."""
    aligned = align_calendar(prices)
    returns = compute_returns(aligned)
    log.info(
        "Cleaned %d rows spanning %s to %s",
        len(returns), returns.index.min().date(), returns.index.max().date(),
    )
    return aligned, returns
