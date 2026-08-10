"""Tests for the selection-bias and sample-length statistics.

These formulas mix per-period and annualised Sharpe ratios, which is the easiest
place in the project to introduce a silent factor-of-sqrt(252) error. The tests
below pin the units down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from research import stats


def _returns(mean: float, sd: float, n: int = 2000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n))


class TestProbabilisticSharpe:
    def test_strong_track_record_is_near_certain(self):
        r = _returns(0.0008, 0.008, 3000)
        assert stats.probabilistic_sharpe_ratio(r) > 0.99

    def test_zero_mean_series_is_near_a_coin_flip(self):
        r = _returns(0.0, 0.01, 4000, seed=3)
        assert 0.2 < stats.probabilistic_sharpe_ratio(r) < 0.8

    def test_probability_falls_as_the_bar_rises(self):
        r = _returns(0.0005, 0.01, 2000, seed=5)
        low = stats.probabilistic_sharpe_ratio(r, 0.0)
        high = stats.probabilistic_sharpe_ratio(r, 0.05)
        assert high < low

    def test_shorter_samples_are_less_conclusive(self):
        """Same underlying process, fewer observations, weaker claim."""
        long = _returns(0.0005, 0.01, 4000, seed=7)
        short = long.iloc[:250]
        assert stats.probabilistic_sharpe_ratio(short) < stats.probabilistic_sharpe_ratio(long)

    def test_negative_skew_is_penalised(self):
        """Two series, same mean and vol, but one has a fat left tail."""
        rng = np.random.default_rng(11)
        base = rng.normal(0.0006, 0.01, 3000)
        skewed = base.copy()
        skewed[::200] -= 0.06  # occasional large losses
        sym = pd.Series(base)
        neg = pd.Series(skewed - skewed.mean() + base.mean())
        neg = neg / neg.std(ddof=1) * sym.std(ddof=1)
        neg = neg - neg.mean() + sym.mean()
        assert stats.probabilistic_sharpe_ratio(neg) < stats.probabilistic_sharpe_ratio(sym)


class TestExpectedMaxSharpe:
    def test_more_trials_raise_the_bar(self):
        v = 0.002
        assert stats.expected_max_sharpe(50, v) > stats.expected_max_sharpe(5, v)

    def test_more_dispersion_raises_the_bar(self):
        assert stats.expected_max_sharpe(20, 0.01) > stats.expected_max_sharpe(20, 0.001)

    def test_single_trial_is_undefined(self):
        assert np.isnan(stats.expected_max_sharpe(1, 0.01))

    def test_zero_dispersion_is_undefined(self):
        assert np.isnan(stats.expected_max_sharpe(10, 0.0))


class TestDeflatedSharpe:
    def test_deflation_never_flatters(self):
        """Accounting for search can only lower confidence, never raise it."""
        r = _returns(0.0006, 0.01, 2000, seed=13)
        trials = list(np.linspace(-0.4, 1.0, 20))
        out = stats.deflated_sharpe_ratio(r, trials)
        assert out["dsr"] <= out["psr_vs_zero"]

    def test_wider_search_deflates_harder(self):
        r = _returns(0.0006, 0.01, 2000, seed=17)
        narrow = stats.deflated_sharpe_ratio(r, [0.1, 0.15, 0.2])
        wide = stats.deflated_sharpe_ratio(r, list(np.linspace(-0.8, 1.2, 40)))
        assert wide["dsr"] < narrow["dsr"]

    def test_units_are_annualised_on_the_way_out(self):
        r = _returns(0.0006, 0.01, 2000, seed=19)
        out = stats.deflated_sharpe_ratio(r, list(np.linspace(-0.4, 1.0, 20)))
        assert out["observed_sharpe_ann"] == pytest.approx(stats.sharpe(r), rel=1e-9)

    def test_single_trial_reports_error(self):
        assert "error" in stats.deflated_sharpe_ratio(_returns(0.0005, 0.01), [0.3])


class TestMinTrackRecordLength:
    def test_weak_edge_needs_a_longer_record(self):
        strong = stats.min_track_record_length(_returns(0.0012, 0.01, 3000, seed=23))
        weak = stats.min_track_record_length(_returns(0.0002, 0.01, 3000, seed=23))
        assert weak["required_obs"] > strong["required_obs"]

    def test_years_and_observations_agree(self):
        out = stats.min_track_record_length(_returns(0.0008, 0.01, 2000, seed=29))
        assert out["required_years"] == pytest.approx(
            out["required_obs"] / config.TRADING_DAYS, rel=0.01
        )

    def test_unreachable_target_is_reported_not_faked(self):
        """A losing strategy cannot prove a positive Sharpe at any length."""
        out = stats.min_track_record_length(_returns(-0.0005, 0.01, 1000), target_sr_ann=1.0)
        assert out["required_obs"] is None
        assert "note" in out

    def test_higher_confidence_demands_more_data(self):
        r = _returns(0.0008, 0.01, 3000, seed=31)
        assert (stats.min_track_record_length(r, confidence=0.99)["required_obs"]
                > stats.min_track_record_length(r, confidence=0.90)["required_obs"])
