"""Orchestrate ingest -> clean -> aggregate into the research panel."""

from __future__ import annotations

import logging

import pandas as pd

import config
from .aggregate import build_panel
from .clean import clean_prices
from .ingest import load_prices

log = logging.getLogger(__name__)


def run_pipeline(refresh: bool = False) -> pd.DataFrame:
    """Run the full ETL and persist the panel. Returns the panel."""
    prices = load_prices(refresh=refresh)
    aligned, returns = clean_prices(prices)
    panel = build_panel(aligned, returns)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.PANEL_CSV)
    log.info("Wrote panel to %s", config.PANEL_CSV)
    return panel


def load_panel(refresh: bool = False) -> pd.DataFrame:
    """Return the cached panel, building it when missing."""
    if config.PANEL_CSV.exists() and not refresh:
        return pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
    return run_pipeline(refresh=refresh)
