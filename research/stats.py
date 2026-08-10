"""Performance statistics and regression tests.

Kept free of any strategy logic so the same functions score the strategy, the
benchmark, and the raw factor identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth of 1 unit."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve, as a negative number."""
    curve = equity_curve(returns)
    return float((curve / curve.cummax() - 1.0).min())


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return (CAGR)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    total = float((1.0 + r).prod())
    if total <= 0:
        return float("nan")
    return total ** (config.TRADING_DAYS / len(r)) - 1.0


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio, excess of a zero risk-free rate.

    Uses the arithmetic mean rather than CAGR so that the ratio is the standard
    t-statistic-like quantity that scales with sqrt(time).
    """
    r = returns.dropna()
    sd = r.std(ddof=1)
    # A constant series has a standard deviation of ~1e-19 rather than exactly
    # zero, so an equality check here would divide by dust and report an
    # astronomically large Sharpe instead of an undefined one.
    if len(r) < 2 or np.isnan(sd) or sd < 1e-12:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(config.TRADING_DAYS))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero days that were positive."""
    r = returns.dropna()
    active = r[r != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def summarise(returns: pd.Series, positions: pd.Series | None = None) -> dict:
    """Standard performance block for a return stream."""
    out = {
        "n_obs": int(returns.dropna().shape[0]),
        "ann_return": annualised_return(returns),
        "ann_vol": annualised_vol(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "total_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
    }
    if positions is not None:
        turnover = positions.diff().abs().dropna()
        out["avg_daily_turnover"] = float(turnover.mean())
        out["ann_turnover"] = float(turnover.mean() * config.TRADING_DAYS)
        out["avg_abs_position"] = float(positions.abs().mean())
    return out


def newey_west_regression(y: pd.Series, x: pd.Series, maxlags: int | None = None) -> dict:
    """Regress ``y`` on ``x`` with HAC (Newey-West) standard errors.

    Overlapping-window predictors and daily returns are autocorrelated, which
    biases plain OLS standard errors downward and makes weak results look
    significant. HAC errors are the minimum defensible correction.
    """
    import statsmodels.api as sm

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {"error": "insufficient overlapping observations", "n_obs": len(df)}

    if maxlags is None:
        # Standard rule of thumb: 4 * (n/100)^(2/9).
        maxlags = int(np.ceil(4 * (len(df) / 100) ** (2 / 9)))

    model = sm.OLS(df["y"], sm.add_constant(df["x"]))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    return {
        "n_obs": int(len(df)),
        "beta": float(res.params["x"]),
        "t_stat": float(res.tvalues["x"]),
        "p_value": float(res.pvalues["x"]),
        "r_squared": float(res.rsquared),
        "alpha": float(res.params["const"]),
        "maxlags": int(maxlags),
    }


def beta_to(returns: pd.Series, benchmark: pd.Series) -> dict:
    """Market beta, annualised alpha and correlation versus a benchmark."""
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return {"error": "insufficient observations"}
    reg = newey_west_regression(df["r"], df["b"])
    return {
        "beta": reg.get("beta"),
        "t_stat": reg.get("t_stat"),
        "alpha_ann": reg.get("alpha", float("nan")) * config.TRADING_DAYS,
        "correlation": float(df["r"].corr(df["b"])),
        "r_squared": reg.get("r_squared"),
    }
