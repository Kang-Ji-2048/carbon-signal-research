"""Build the research panel: baskets, the GMB factor, and carbon features."""

from __future__ import annotations

import logging

import pandas as pd

import config

log = logging.getLogger(__name__)


def basket_return(returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weighted, daily-rebalanced basket return.

    ``mean`` skips NaNs, so a basket stays well defined before every constituent
    has listed; it simply holds the members that exist on that date.
    """
    present = [t for t in tickers if t in returns.columns]
    if not present:
        raise ValueError(f"none of {tickers} present in returns")
    return returns[present].mean(axis=1, skipna=True)


def build_panel(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Assemble the per-date research panel.

    Columns:
      green, brown   equal-weighted basket returns
      gmb            green minus brown, the long/short factor
      benchmark      SPY return, for beta and alpha attribution
      carbon         KRBN return
      carbon_px      KRBN level, used to build momentum features
    """
    panel = pd.DataFrame(index=returns.index)
    panel["green"] = basket_return(returns, config.GREEN)
    panel["brown"] = basket_return(returns, config.BROWN)
    panel["gmb"] = panel["green"] - panel["brown"]
    panel["benchmark"] = returns[config.BENCHMARK]
    panel["carbon"] = returns[config.CARBON]
    panel["carbon_px"] = prices[config.CARBON].reindex(panel.index)

    panel = panel.dropna(subset=["gmb"])
    log.info(
        "Panel: %d rows, %s to %s (carbon available from %s)",
        len(panel), panel.index.min().date(), panel.index.max().date(),
        panel["carbon_px"].first_valid_index().date(),
    )
    return panel
