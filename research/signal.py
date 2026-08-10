"""Signal construction.

Every function here is *causal*: a value dated ``t`` uses only information
observable at the close of ``t``. Converting that into a tradeable position
(which requires a further one-day lag) is the backtester's job, not the
signal's. Keeping the two responsibilities apart means the lag exists in
exactly one place and can be tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def carbon_momentum(carbon_px: pd.Series, lookback: int = config.CARBON_LOOKBACK) -> pd.Series:
    """Trailing ``lookback``-day return of the carbon price.

    ``pct_change`` over a window ending at ``t`` uses prices at ``t`` and
    ``t - lookback``, both known at ``t``.
    """
    return carbon_px.pct_change(lookback)


def realised_vol(returns: pd.Series, lookback: int = config.VOL_LOOKBACK) -> pd.Series:
    """Annualised trailing realised volatility.

    The rolling window is right-aligned and closed on the right, so the value at
    ``t`` is computed from returns up to and including ``t``.
    """
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(
        config.TRADING_DAYS
    )


def build_signal(
    panel: pd.DataFrame,
    lookback: int = config.CARBON_LOOKBACK,
    vol_lookback: int = config.VOL_LOOKBACK,
    target_vol: float = config.TARGET_VOL,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.DataFrame:
    """Build the carbon-momentum position series.

    The rule: hold the green-minus-brown spread long when carbon momentum is
    positive and short when it is negative, sized so the position's *expected*
    volatility matches ``target_vol`` given recent realised volatility.

    Vol targeting is included because an unscaled long/short spread has wildly
    time-varying risk; without it, Sharpe comparisons mostly reflect changes in
    exposure rather than the quality of the signal.
    """
    out = pd.DataFrame(index=panel.index)
    out["carbon_momentum"] = carbon_momentum(panel["carbon_px"], lookback)
    out["gmb_vol"] = realised_vol(panel["gmb"], vol_lookback)

    direction = np.sign(out["carbon_momentum"])
    scale = (target_vol / out["gmb_vol"]).clip(upper=max_leverage)

    out["direction"] = direction
    out["scale"] = scale
    out["position"] = (direction * scale).where(
        out["carbon_momentum"].notna() & out["gmb_vol"].notna()
    )
    return out
