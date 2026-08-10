"""Run the full study and serialise results for the dashboard and README."""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

import config
from etl.pipeline import load_panel

from . import stats
from .backtest import evaluate, run_backtest, sensitivity_sweep, split_sample
from .signal import build_signal

log = logging.getLogger(__name__)


def _clean(obj):
    """Make numpy/pandas values JSON-serialisable, with NaN as null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def factor_characterisation(panel: pd.DataFrame) -> dict:
    """Describe the GMB factor itself, before any signal is applied."""
    gmb = panel["gmb"]
    carbon_start = panel["carbon_px"].first_valid_index()
    recent = panel.loc[carbon_start:]

    return {
        "full_history": {
            **stats.summarise(gmb),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
        },
        "carbon_era": {
            **stats.summarise(recent["gmb"]),
            "start": str(carbon_start.date()),
            "end": str(panel.index.max().date()),
        },
        "vs_market_full": stats.beta_to(gmb, panel["benchmark"]),
        "green_leg": stats.summarise(panel["green"]),
        "brown_leg": stats.summarise(panel["brown"]),
    }


def predictive_test(panel: pd.DataFrame, lookback: int = config.CARBON_LOOKBACK) -> dict:
    """Does carbon momentum predict *next day's* GMB return?

    The predictor is dated ``t`` and the target is the return of ``t+1``,
    obtained by shifting the target backwards so that each row pairs a decision
    with the outcome that followed it.
    """
    sig = build_signal(panel, lookback=lookback)
    forward_gmb = panel["gmb"].shift(-1)

    result = stats.newey_west_regression(forward_gmb, sig["carbon_momentum"])
    result["lookback"] = lookback
    result["interpretation"] = (
        "positive beta means rising carbon prices precede green outperformance"
    )

    # Same test on the contemporaneous relationship, which should be stronger:
    # it measures co-movement, not prediction, and is reported to make the
    # distinction explicit rather than to support a trading claim.
    result["contemporaneous"] = stats.newey_west_regression(
        panel["gmb"], panel["carbon"]
    )
    return result


def selection_bias_analysis(panel: pd.DataFrame, sweep: pd.DataFrame, bt: pd.DataFrame) -> dict:
    """Quantify how much of the best backtest is explained by having searched.

    Two questions a short, heavily-searched backtest always invites:
      1. The best cell looks good - but I tried N configurations. Is it real?
         -> Deflated Sharpe, which raises the bar by the expected best result
            from N skill-free trials.
      2. My headline result is weak. Was the sample even long enough to detect
         anything? -> Minimum track record length.
    """
    trial_sharpes = sweep["oos_sharpe"].tolist()
    best = sweep.loc[sweep["oos_sharpe"].idxmax()]

    best_sig = build_signal(panel, lookback=int(best["lookback"]))
    best_bt = run_backtest(panel, best_sig["position"], cost_bps=float(best["cost_bps"]))
    _, best_oos = split_sample(best_bt)

    _, headline_oos = split_sample(bt)

    return {
        "best_config": {
            "lookback": int(best["lookback"]),
            "cost_bps": float(best["cost_bps"]),
            "oos_sharpe": float(best["oos_sharpe"]),
        },
        "deflated_best": stats.deflated_sharpe_ratio(best_oos["net_return"], trial_sharpes),
        "deflated_headline": stats.deflated_sharpe_ratio(headline_oos["net_return"], trial_sharpes),
        "min_track_record_best": stats.min_track_record_length(best_oos["net_return"]),
        "min_track_record_to_prove_sharpe_0p5": stats.min_track_record_length(
            best_oos["net_return"], target_sr_ann=0.5
        ),
    }


def build_report(refresh: bool = False) -> dict:
    panel = load_panel(refresh=refresh)
    sig = build_signal(panel)
    bt = run_backtest(panel, sig["position"])
    sweep = sensitivity_sweep(panel)

    perf = evaluate(bt, panel)
    curve = stats.equity_curve(bt["net_return"])
    bench_curve = stats.equity_curve(panel["gmb"].reindex(bt.index))

    report = {
        "generated": str(date.today()),
        "universe": {
            "green": config.GREEN,
            "brown": config.BROWN,
            "carbon": config.CARBON,
            "benchmark": config.BENCHMARK,
        },
        "parameters": {
            "carbon_lookback": config.CARBON_LOOKBACK,
            "vol_lookback": config.VOL_LOOKBACK,
            "target_vol": config.TARGET_VOL,
            "max_leverage": config.MAX_LEVERAGE,
            "cost_bps": config.COST_BPS,
            "legs": config.LEGS,
            "oos_split": config.OOS_SPLIT,
        },
        "sample": {
            "panel_start": str(panel.index.min().date()),
            "panel_end": str(panel.index.max().date()),
            "panel_obs": int(len(panel)),
            "backtest_start": str(bt.index.min().date()),
            "backtest_end": str(bt.index.max().date()),
            "backtest_obs": int(len(bt)),
        },
        "factor": factor_characterisation(panel),
        "predictive_test": predictive_test(panel),
        "selection_bias": selection_bias_analysis(panel, sweep, bt),
        "performance": perf,
        "sensitivity": sweep.to_dict(orient="records"),
        "series": {
            "dates": [str(d.date()) for d in bt.index],
            "strategy_equity": curve.round(6).tolist(),
            "gmb_equity": bench_curve.round(6).tolist(),
            "position": bt["position"].round(4).tolist(),
            "carbon_momentum": sig["carbon_momentum"].reindex(bt.index).round(6).tolist(),
            "drawdown": (curve / curve.cummax() - 1).round(6).tolist(),
        },
    }
    return _clean(report)


def save_report(refresh: bool = False) -> dict:
    report = build_report(refresh=refresh)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Wrote %s", config.RESULTS_JSON)
    return report


def load_report() -> dict:
    if not config.RESULTS_JSON.exists():
        return save_report()
    with open(config.RESULTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)
