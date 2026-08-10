"""Tests for P&L accounting and, above all, the absence of lookahead bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from research.backtest import run_backtest, split_sample
from research.signal import build_signal


def _panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n, name="date")
    gmb = pd.Series(rng.normal(0, 0.012, n), index=idx)
    carbon = pd.Series(rng.normal(0.0004, 0.015, n), index=idx)
    return pd.DataFrame({
        "gmb": gmb,
        "carbon": carbon,
        "carbon_px": 30 * (1 + carbon).cumprod(),
        "benchmark": pd.Series(rng.normal(0.0003, 0.009, n), index=idx),
        "green": gmb / 2,
        "brown": -gmb / 2,
    }, index=idx)


class TestPnLAccounting:
    def test_position_earns_next_day_return(self):
        """P&L on day t must come from the position decided on day t-1."""
        panel = _panel(10)
        pos = pd.Series(1.0, index=panel.index)
        bt = run_backtest(panel, pos, cost_bps=0.0)

        # Day 0 has no prior position, so it drops out; every later day matches
        # the raw factor return exactly when fully invested at no cost.
        expected = panel["gmb"].iloc[1:]
        pd.testing.assert_series_equal(
            bt["gross_return"], expected, check_names=False,
        )

    def test_zero_position_earns_nothing(self):
        panel = _panel(20)
        bt = run_backtest(panel, pd.Series(0.0, index=panel.index), cost_bps=10.0)
        assert bt["net_return"].abs().sum() == pytest.approx(0.0)

    def test_short_position_flips_sign(self):
        panel = _panel(20)
        long = run_backtest(panel, pd.Series(1.0, index=panel.index), cost_bps=0.0)
        short = run_backtest(panel, pd.Series(-1.0, index=panel.index), cost_bps=0.0)
        assert long["gross_return"].sum() == pytest.approx(-short["gross_return"].sum())

    def test_costs_charged_on_turnover_only(self):
        """A constant position trades once, then never again."""
        panel = _panel(50)
        pos = pd.Series(1.0, index=panel.index)
        bt = run_backtest(panel, pos, cost_bps=10.0)
        # First held day has no prior position to difference against, so
        # turnover is NaN there and zero everywhere after.
        assert bt["cost"].fillna(0.0).sum() == pytest.approx(0.0)

    def test_cost_magnitude_matches_turnover(self):
        """Flipping from +1 to -1 is 2 units of turnover across `legs` legs."""
        panel = _panel(6)
        pos = pd.Series([1, 1, 1, -1, -1, -1], index=panel.index, dtype=float)
        bt = run_backtest(panel, pos, cost_bps=10.0, legs=2)
        expected = 2.0 * (10.0 / 1e4) * 2
        assert bt["cost"].max() == pytest.approx(expected)

    def test_higher_costs_reduce_net_return(self):
        panel = _panel(300)
        sig = build_signal(panel)
        cheap = run_backtest(panel, sig["position"], cost_bps=0.0)
        dear = run_backtest(panel, sig["position"], cost_bps=50.0)
        assert dear["net_return"].sum() < cheap["net_return"].sum()


class TestNoLookahead:
    def test_future_data_cannot_change_past_pnl(self):
        """The decisive test: corrupt the future, the past must not move.

        If any component peeked ahead - a centred rolling window, a backfill, a
        missing shift - rewriting data after the cut date would alter P&L before
        it. This asserts it does not.
        """
        panel = _panel(400, seed=7)
        cut = 250

        base_bt = run_backtest(panel, build_signal(panel)["position"])

        corrupted = panel.copy()
        rng = np.random.default_rng(99)
        corrupted.iloc[cut:, corrupted.columns.get_loc("gmb")] = rng.normal(5.0, 1.0, len(panel) - cut)
        corrupted.iloc[cut:, corrupted.columns.get_loc("carbon_px")] = 1e4
        corrupt_bt = run_backtest(corrupted, build_signal(corrupted)["position"])

        cut_date = panel.index[cut - 1]
        a = base_bt.loc[base_bt.index <= cut_date, "net_return"]
        b = corrupt_bt.loc[corrupt_bt.index <= cut_date, "net_return"]

        assert len(a) > 100, "test needs a meaningful pre-cut window"
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_signal_is_not_shifted_twice(self):
        """The lag lives in the backtester, so the signal itself is contemporaneous."""
        panel = _panel(200)
        sig = build_signal(panel, lookback=20, vol_lookback=20)
        mom = sig["carbon_momentum"].dropna()
        manual = panel["carbon_px"].pct_change(20).dropna()
        pd.testing.assert_series_equal(mom, manual, check_names=False)

    def test_realised_vol_uses_only_trailing_data(self):
        panel = _panel(100)
        sig = build_signal(panel, vol_lookback=30)
        t = panel.index[60]
        expected = panel["gmb"].loc[:t].iloc[-30:].std(ddof=1) * np.sqrt(config.TRADING_DAYS)
        assert sig["gmb_vol"].loc[t] == pytest.approx(expected)


class TestSplit:
    def test_split_is_chronological_and_disjoint(self):
        panel = _panel(200)
        bt = run_backtest(panel, pd.Series(1.0, index=panel.index))
        is_df, oos_df = split_sample(bt, oos_split=0.4)
        assert is_df.index.max() < oos_df.index.min()
        assert len(is_df) + len(oos_df) == len(bt)
