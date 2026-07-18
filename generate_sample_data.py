"""
Generate realistic sample climate finance data mimicking OECD, IRENA, and
World Bank public datasets. Run once to populate data/raw/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(2048)
RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2013, 2026))

REGIONS = {
    "Sub-Saharan Africa": ["Kenya", "Nigeria", "South Africa", "Ethiopia", "Ghana", "Tanzania"],
    "South Asia": ["India", "Bangladesh", "Pakistan", "Sri Lanka", "Nepal"],
    "East Asia & Pacific": ["China", "Indonesia", "Vietnam", "Philippines", "Thailand"],
    "Latin America & Caribbean": ["Brazil", "Mexico", "Colombia", "Chile", "Argentina"],
    "Europe & Central Asia": ["Turkey", "Ukraine", "Poland", "Romania", "Kazakhstan"],
    "Middle East & North Africa": ["Morocco", "Egypt", "Jordan", "Tunisia"],
}

SECTORS_OECD = [
    "Renewable Energy", "Energy Efficiency", "Sustainable Transport",
    "Climate Adaptation", "Forestry & Land Use", "Water & Waste Management",
]
SECTORS_IRENA = ["Renewable Energy"]
SECTORS_WB = [
    "Renewable Energy", "Energy Efficiency", "Sustainable Transport",
    "Climate Adaptation", "Forestry & Land Use", "Water & Waste Management", "Cross-cutting",
]

INSTRUMENTS_OECD = ["Grant", "Concessional Loan", "Non-Concessional Loan", "Equity"]
INSTRUMENTS_IRENA = ["Equity", "Non-Concessional Loan", "Bond", "Grant"]
INSTRUMENTS_WB = ["Concessional Loan", "Grant", "Guarantee", "Mixed/Other"]


def _growth_factor(year: int, base_year: int = 2013) -> float:
    """Simulate ~12% annual growth in climate finance with some noise."""
    years_elapsed = year - base_year
    trend = (1.12 ** years_elapsed)
    noise = np.random.uniform(0.85, 1.15)
    return trend * noise


def generate_oecd() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for region, countries in REGIONS.items():
            for country in countries:
                n_flows = np.random.randint(1, 4)
                for _ in range(n_flows):
                    sector = np.random.choice(SECTORS_OECD, p=[0.30, 0.15, 0.15, 0.20, 0.10, 0.10])
                    instrument = np.random.choice(INSTRUMENTS_OECD, p=[0.35, 0.30, 0.25, 0.10])
                    base = np.random.exponential(scale=80)
                    amount = round(base * _growth_factor(year), 2)
                    rows.append({
                        "year": year,
                        "region": region,
                        "country": country,
                        "sector": sector,
                        "instrument_type": instrument,
                        "amount_usd_mn": amount,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "oecd_climate_finance.csv", index=False)
    print(f"OECD: {len(df):,} records")
    return df


def generate_irena() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for region, countries in REGIONS.items():
            for country in countries:
                n_flows = np.random.randint(1, 3)
                for _ in range(n_flows):
                    instrument = np.random.choice(INSTRUMENTS_IRENA, p=[0.35, 0.25, 0.25, 0.15])
                    base = np.random.exponential(scale=150)
                    amount = round(base * _growth_factor(year) * 1.2, 2)
                    rows.append({
                        "year": year,
                        "region": region,
                        "country": country,
                        "sector": "Renewable Energy",
                        "finance_type": instrument,
                        "value_usd_mn": amount,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "irena_renewable_investment.csv", index=False)
    print(f"IRENA: {len(df):,} records")
    return df


def generate_world_bank() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        for region, countries in REGIONS.items():
            for country in countries:
                n_flows = np.random.randint(1, 3)
                for _ in range(n_flows):
                    sector = np.random.choice(
                        SECTORS_WB, p=[0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.10]
                    )
                    instrument = np.random.choice(INSTRUMENTS_WB, p=[0.40, 0.30, 0.15, 0.15])
                    base = np.random.exponential(scale=120)
                    amount = round(base * _growth_factor(year) * 0.9, 2)
                    rows.append({
                        "year": year,
                        "region": region,
                        "country": country,
                        "sector": sector,
                        "instrument": instrument,
                        "amount": amount,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "world_bank_climate_finance.csv", index=False)
    print(f"World Bank: {len(df):,} records")
    return df


if __name__ == "__main__":
    print("Generating sample climate finance data...\n")
    generate_oecd()
    generate_irena()
    generate_world_bank()
    print("\nDone. Files saved to data/raw/")
