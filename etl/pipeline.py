"""
Main ETL pipeline orchestrator.

Runs ingest -> clean -> aggregate and persists the processed output.
"""

import pandas as pd
from pathlib import Path

from .ingest import ingest_all_sources
from .clean import clean_dataset
from .aggregate import aggregate_dataset

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Execute the full ETL pipeline and save processed data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] INGESTING data sources...")
    raw = ingest_all_sources()

    print("\n[2/3] CLEANING dataset...")
    clean = clean_dataset(raw)

    # Save the unified clean dataset
    clean_path = PROCESSED_DIR / "climate_finance_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"  Saved clean dataset -> {clean_path}")

    print("\n[3/3] AGGREGATING views...")
    views = aggregate_dataset(clean)

    # Persist each aggregated view
    for name, view_df in views.items():
        if name == "detail":
            continue
        out_path = PROCESSED_DIR / f"{name}.csv"
        view_df.to_csv(out_path, index=False)
        print(f"  Saved {name} -> {out_path}")

    print("\nETL pipeline complete.\n")
    return views
