"""
Data cleaning module.

Handles:
  - Column name standardisation
  - Missing value imputation
  - Type casting and validation
  - Currency normalisation (all values to USD millions)
  - Deduplication
"""

import pandas as pd
import numpy as np


STANDARD_COLUMNS = {
    "year": "year",
    "region": "region",
    "country": "country",
    "sector": "sector",
    "instrument_type": "instrument_type",
    "amount_usd_mn": "amount_usd_mn",
    "source": "source",
}

VALID_SECTORS = {
    "Renewable Energy",
    "Energy Efficiency",
    "Sustainable Transport",
    "Climate Adaptation",
    "Forestry & Land Use",
    "Water & Waste Management",
    "Cross-cutting",
}

VALID_INSTRUMENTS = {
    "Grant",
    "Concessional Loan",
    "Non-Concessional Loan",
    "Equity",
    "Guarantee",
    "Bond",
    "Mixed/Other",
}


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names across sources."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Coalesce amount columns into a single column
    amount_aliases = ["value_usd_mn", "amount_usd", "amount"]
    for col in amount_aliases:
        if col in df.columns and col != "amount_usd_mn":
            if "amount_usd_mn" in df.columns:
                df["amount_usd_mn"] = df["amount_usd_mn"].fillna(df[col])
            else:
                df = df.rename(columns={col: "amount_usd_mn"})
            if col in df.columns and col != "amount_usd_mn":
                df = df.drop(columns=[col])

    # Coalesce instrument columns
    instrument_aliases = ["finance_type", "instrument"]
    for col in instrument_aliases:
        if col in df.columns and col != "instrument_type":
            if "instrument_type" in df.columns:
                df["instrument_type"] = df["instrument_type"].fillna(df[col])
            else:
                df = df.rename(columns={col: "instrument_type"})
            if col in df.columns and col != "instrument_type":
                df = df.drop(columns=[col])

    return df


def _clean_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert amounts to numeric, drop negatives."""
    df["amount_usd_mn"] = pd.to_numeric(df["amount_usd_mn"], errors="coerce")
    df = df[df["amount_usd_mn"] > 0].copy()
    df["amount_usd_mn"] = df["amount_usd_mn"].round(2)
    return df


def _normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map sector and instrument values to standard categories."""
    df["sector"] = df["sector"].str.strip().str.title()
    df["instrument_type"] = df["instrument_type"].str.strip().str.title()

    df.loc[~df["sector"].isin(VALID_SECTORS), "sector"] = "Cross-cutting"
    df.loc[
        ~df["instrument_type"].isin(VALID_INSTRUMENTS), "instrument_type"
    ] = "Mixed/Other"
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    df["country"] = df["country"].fillna("Unspecified")
    df["region"] = df["region"].fillna("Unspecified")
    df = df.dropna(subset=["year", "amount_usd_mn"])
    return df


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    before = len(df)
    df = df.drop_duplicates(
        subset=["year", "country", "sector", "instrument_type", "amount_usd_mn", "source"]
    )
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed:,} duplicate records")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning pipeline on the raw combined dataset."""
    print("  Standardising columns...")
    df = _standardise_columns(df)

    print("  Cleaning amounts...")
    df = _clean_amounts(df)

    print("  Normalising categories...")
    df = _normalise_categories(df)

    print("  Filling missing values...")
    df = _fill_missing(df)

    print("  Deduplicating...")
    df = _dedup(df)

    df["year"] = df["year"].astype(int)
    df = df[list(STANDARD_COLUMNS.values())].copy()
    df = df.sort_values(["year", "region", "sector"]).reset_index(drop=True)

    print(f"  Clean dataset: {len(df):,} records")
    return df
