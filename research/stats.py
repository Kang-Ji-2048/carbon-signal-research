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


def _sharpe_moments(returns: pd.Series) -> tuple[float, float, float, int]:
    """Per-period Sharpe plus the skew and raw kurtosis the DSR formulas need.

    These are *per-observation*, not annualised. Bailey and Lopez de Prado's
    formulas are defined on the raw per-period ratio, and mixing in a sqrt(252)
    would silently invalidate every result below.
    """
    from scipy import stats as sps

    r = returns.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or np.isnan(sd) or sd < 1e-12:
        return float("nan"), float("nan"), float("nan"), n
    sr = float(r.mean() / sd)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))  # raw, not excess
    return sr, skew, kurt, n


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > ``benchmark_sr``), given the sample's length and shape.

    ``benchmark_sr`` is a *per-period* Sharpe. Unlike a plain t-test this
    penalises negative skew and fat tails, which is what makes it appropriate
    for return series.
    """
    from scipy import stats as sps

    sr, skew, kurt, n = _sharpe_moments(returns)
    if np.isnan(sr):
        return float("nan")
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var <= 0:
        return float("nan")
    return float(sps.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(var)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best per-period Sharpe from ``n_trials`` *skill-free* strategies.

    This is the benchmark that a winning backtest has to beat. Search hard
    enough over configurations and something will look good by chance; this
    quantifies how good, so the comparison can be made explicitly.
    """
    from scipy import stats as sps

    if n_trials < 2 or sr_variance <= 0 or np.isnan(sr_variance):
        return float("nan")
    euler = 0.5772156649015329
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - euler) * z1 + euler * z2))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: list[float]) -> dict:
    """Deflate the observed Sharpe by the number of configurations tried.

    ``trial_sharpes`` are the *annualised* Sharpes of every configuration in the
    search, converted here to per-period units. The returned ``dsr`` is the
    probability that the strategy's true Sharpe is positive *after* accounting
    for that search. Conventionally, below 0.95 means "not demonstrated".
    """
    trials = np.asarray([s for s in trial_sharpes if s is not None and not np.isnan(s)],
                        dtype=float)
    if len(trials) < 2:
        return {"error": "need at least two trials to deflate"}

    per_period = trials / np.sqrt(config.TRADING_DAYS)
    sr0 = expected_max_sharpe(len(per_period), float(np.var(per_period, ddof=1)))
    sr, _, _, n = _sharpe_moments(returns)
    if np.isnan(sr) or np.isnan(sr0):
        return {"error": "insufficient data to deflate"}

    return {
        "n_trials": int(len(per_period)),
        "observed_sharpe_ann": float(sr * np.sqrt(config.TRADING_DAYS)),
        "expected_max_sharpe_ann": float(sr0 * np.sqrt(config.TRADING_DAYS)),
        "psr_vs_zero": probabilistic_sharpe_ratio(returns, 0.0),
        "dsr": probabilistic_sharpe_ratio(returns, sr0),
        "n_obs": int(n),
        "significant": bool(probabilistic_sharpe_ratio(returns, sr0) > 0.95),
    }


def min_track_record_length(
    returns: pd.Series, target_sr_ann: float = 0.0, confidence: float = 0.95
) -> dict:
    """Observations needed to prove the Sharpe exceeds ``target_sr_ann``.

    Answers the question a short backtest always invites: is this sample even
    long enough to have detected the effect? If the required length exceeds the
    sample, a null result is inconclusive rather than negative.
    """
    from scipy import stats as sps

    sr, skew, kurt, n = _sharpe_moments(returns)
    if np.isnan(sr):
        return {"error": "insufficient data"}

    target = target_sr_ann / np.sqrt(config.TRADING_DAYS)
    edge = sr - target
    if edge <= 0:
        return {
            "n_obs": int(n),
            "required_obs": None,
            "required_years": None,
            "note": "observed Sharpe does not exceed the target, so no sample length suffices",
        }

    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    required = 1.0 + var * (sps.norm.ppf(confidence) / edge) ** 2
    return {
        "n_obs": int(n),
        "required_obs": int(np.ceil(required)),
        "required_years": float(required / config.TRADING_DAYS),
        "have_years": float(n / config.TRADING_DAYS),
        "sufficient": bool(n >= required),
        "confidence": confidence,
    }


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
