"""
Aggregation module.

Produces pre-aggregated views used by the dashboard:
  - By year + sector
  - By year + region
  - By year + instrument type
  - By year + region + sector (full detail)
"""

import pandas as pd


def _agg_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate amount by given grouping columns."""
    return (
        df.groupby(group_cols, as_index=False)["amount_usd_mn"]
        .agg(total_usd_mn="sum", flow_count="count")
        .round(2)
    )


def aggregate_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce all aggregation views from the cleaned dataset."""
    views = {}

    views["by_year_sector"] = _agg_by(df, ["year", "sector"])
    print(f"  by_year_sector: {len(views['by_year_sector']):,} rows")

    views["by_year_region"] = _agg_by(df, ["year", "region"])
    print(f"  by_year_region: {len(views['by_year_region']):,} rows")

    views["by_year_instrument"] = _agg_by(df, ["year", "instrument_type"])
    print(f"  by_year_instrument: {len(views['by_year_instrument']):,} rows")

    views["by_year_region_sector"] = _agg_by(df, ["year", "region", "sector"])
    print(f"  by_year_region_sector: {len(views['by_year_region_sector']):,} rows")

    views["by_source"] = _agg_by(df, ["year", "source"])
    print(f"  by_source: {len(views['by_source']):,} rows")

    views["detail"] = df.copy()

    return views
