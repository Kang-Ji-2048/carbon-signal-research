"""Tests for the performance statistics and the ETL's missing-data policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from etl.aggregate import basket_return
from etl.clean import align_calendar, compute_returns
from research import stats


class TestStats:
    def test_max_drawdown_of_known_path(self):
        # 1.0 -> 1.10 -> 0.88: peak 1.10, trough 0.88 => -20%.
        r = pd.Series([0.10, -0.20])
        assert stats.max_drawdown(r) == pytest.approx(-0.20)

    def test_max_drawdown_never_positive(self):
        r = pd.Series([0.01] * 50)
        assert stats.max_drawdown(r) <= 0.0

    def test_annualised_return_compounds(self):
        """A constant daily return must annualise geometrically."""
        daily = 0.0004
        r = pd.Series([daily] * config.TRADING_DAYS)
        assert stats.annualised_return(r) == pytest.approx((1 + daily) ** 252 - 1, rel=1e-9)

    def test_annualised_vol_scales_with_sqrt_time(self):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0, 0.01, 5000))
        assert stats.annualised_vol(r) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)

    def test_sharpe_of_riskless_series_is_nan(self):
        assert np.isnan(stats.sharpe(pd.Series([0.001] * 20)))

    def test_sharpe_sign_follows_mean(self):
        rng = np.random.default_rng(5)
        good = pd.Series(rng.normal(0.001, 0.01, 2000))
        assert stats.sharpe(good) > 0
        assert stats.sharpe(-good) < 0

    def test_hit_rate_ignores_flat_days(self):
        r = pd.Series([0.01, -0.01, 0.0, 0.0, 0.01])
        assert stats.hit_rate(r) == pytest.approx(2 / 3)

    def test_summarise_reports_turnover_when_positions_given(self):
        idx = pd.bdate_range("2021-01-01", periods=10)
        r = pd.Series(0.001, index=idx)
        pos = pd.Series([1, 1, -1, -1, 1, 1, 1, -1, -1, -1], index=idx, dtype=float)
        out = stats.summarise(r, positions=pos)
        assert out["avg_daily_turnover"] > 0
        assert out["ann_turnover"] == pytest.approx(out["avg_daily_turnover"] * 252)


class TestRegression:
    def test_recovers_known_beta(self):
        rng = np.random.default_rng(11)
        x = pd.Series(rng.normal(0, 1, 3000))
        y = 0.35 * x + pd.Series(rng.normal(0, 0.4, 3000))
        res = stats.newey_west_regression(y, x)
        assert res["beta"] == pytest.approx(0.35, abs=0.05)
        assert abs(res["t_stat"]) > 5

    def test_pure_noise_is_insignificant(self):
        rng = np.random.default_rng(13)
        x = pd.Series(rng.normal(0, 1, 2000))
        y = pd.Series(rng.normal(0, 1, 2000))
        res = stats.newey_west_regression(y, x)
        assert abs(res["t_stat"]) < 3

    def test_short_sample_reports_error(self):
        s = pd.Series(np.arange(5, dtype=float))
        assert "error" in stats.newey_west_regression(s, s)


class TestCleaning:
    def test_forward_fill_only_never_backfills(self):
        """A leading gap must stay NaN; filling it backwards imports the future."""
        idx = pd.bdate_range("2021-01-01", periods=5)
        px = pd.DataFrame({"A": [np.nan, np.nan, 10.0, np.nan, 12.0]}, index=idx)
        out = align_calendar(px)
        assert out["A"].iloc[:2].isna().all()
        assert out["A"].iloc[3] == pytest.approx(10.0)

    def test_long_gaps_are_not_invented(self):
        idx = pd.bdate_range("2021-01-01", periods=8)
        px = pd.DataFrame({"A": [10.0] + [np.nan] * 6 + [20.0]}, index=idx)
        out = align_calendar(px, max_fill=2)
        assert out["A"].iloc[3:7].isna().all()

    def test_returns_drop_first_row(self):
        idx = pd.bdate_range("2021-01-01", periods=4)
        px = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx)
        r = compute_returns(px)
        assert len(r) == 3
        assert r["A"].iloc[0] == pytest.approx(0.1)

    def test_basket_is_equal_weighted_mean(self):
        idx = pd.bdate_range("2021-01-01", periods=3)
        r = pd.DataFrame({"A": [0.02, 0.04, 0.0], "B": [0.00, 0.00, 0.10]}, index=idx)
        b = basket_return(r, ["A", "B"])
        assert b.tolist() == pytest.approx([0.01, 0.02, 0.05])

    def test_basket_skips_missing_constituents(self):
        """Before a constituent lists, the basket holds only what exists."""
        idx = pd.bdate_range("2021-01-01", periods=2)
        r = pd.DataFrame({"A": [0.02, 0.04], "B": [np.nan, 0.06]}, index=idx)
        b = basket_return(r, ["A", "B"])
        assert b.iloc[0] == pytest.approx(0.02)
        assert b.iloc[1] == pytest.approx(0.05)

    def test_unknown_tickers_raise(self):
        r = pd.DataFrame({"A": [0.01]})
        with pytest.raises(ValueError):
            basket_return(r, ["ZZZ"])
