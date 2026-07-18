"""
Data ingestion module for climate finance data.

Sources
-------
- **OECD**  – DAC2A (ODA disbursements by recipient) via the SDMX REST
  API.  Returns bilateral ODA flows for target countries in USD millions.
  Endpoint: ``sdmx.oecd.org``  — Dataflow ``DSD_DAC2@DF_DAC2A`` v1.0

- **World Bank** – Climate-tagged IBRD/IDA projects via the World Bank
  Projects Search API.  Returns project-level commitment amounts.
  Endpoint: ``search.worldbank.org``

- **IRENA** – Generated sample data (IRENA does not expose a structured
  public API).

Each source is normalised to a common schema::

    year | region | country | sector | instrument_type | amount_usd_mn

and cached as CSV in ``data/raw/``.  Caches older than
``CACHE_MAX_AGE_HOURS`` trigger a fresh API call; on API failure the
last cache is reused.
"""

import pandas as pd
import numpy as np
import requests
import io
import re
import time
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CACHE_MAX_AGE_HOURS = 24
REQUEST_TIMEOUT = 90

# ╔═══════════════════════════════════════════════════════════╗
# ║  Shared lookups                                          ║
# ╚═══════════════════════════════════════════════════════════╝

COUNTRY_META: dict[str, tuple[str, str]] = {
    # ISO-3166-1 alpha-3 -> (display name, dashboard region)
    # Sub-Saharan Africa
    "KEN": ("Kenya",        "Sub-Saharan Africa"),
    "NGA": ("Nigeria",      "Sub-Saharan Africa"),
    "ZAF": ("South Africa", "Sub-Saharan Africa"),
    "ETH": ("Ethiopia",     "Sub-Saharan Africa"),
    "GHA": ("Ghana",        "Sub-Saharan Africa"),
    "TZA": ("Tanzania",     "Sub-Saharan Africa"),
    # South Asia
    "IND": ("India",        "South Asia"),
    "BGD": ("Bangladesh",   "South Asia"),
    "PAK": ("Pakistan",     "South Asia"),
    "LKA": ("Sri Lanka",    "South Asia"),
    "NPL": ("Nepal",        "South Asia"),
    # East Asia & Pacific
    "CHN": ("China",        "East Asia & Pacific"),
    "IDN": ("Indonesia",    "East Asia & Pacific"),
    "VNM": ("Vietnam",      "East Asia & Pacific"),
    "PHL": ("Philippines",  "East Asia & Pacific"),
    "THA": ("Thailand",     "East Asia & Pacific"),
    # Latin America & Caribbean
    "BRA": ("Brazil",       "Latin America & Caribbean"),
    "MEX": ("Mexico",       "Latin America & Caribbean"),
    "COL": ("Colombia",     "Latin America & Caribbean"),
    "CHL": ("Chile",        "Latin America & Caribbean"),
    "ARG": ("Argentina",    "Latin America & Caribbean"),
    # Europe & Central Asia
    "TUR": ("Turkey",       "Europe & Central Asia"),
    "UKR": ("Ukraine",      "Europe & Central Asia"),
    "POL": ("Poland",       "Europe & Central Asia"),
    "ROU": ("Romania",      "Europe & Central Asia"),
    "KAZ": ("Kazakhstan",   "Europe & Central Asia"),
    # Middle East & North Africa
    "MAR": ("Morocco",      "Middle East & North Africa"),
    "EGY": ("Egypt",        "Middle East & North Africa"),
    "JOR": ("Jordan",       "Middle East & North Africa"),
    "TUN": ("Tunisia",      "Middle East & North Africa"),
}

# Reverse lookup: country display name -> region
_NAME_TO_REGION = {name: region for _, (name, region) in COUNTRY_META.items()}

# ╔═══════════════════════════════════════════════════════════╗
# ║  Cache helpers                                           ║
# ╚═══════════════════════════════════════════════════════════╝

def _is_cache_fresh(path: Path) -> bool:
    """Return True when *path* exists and is younger than the max age."""
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < CACHE_MAX_AGE_HOURS


def _load_with_api(
    fetch_fn,
    cache_path: Path,
    source_label: str,
) -> pd.DataFrame:
    """Try *fetch_fn*; on success write to *cache_path*.  On failure reuse cache."""
    if not _is_cache_fresh(cache_path):
        try:
            df = fetch_fn()
            df.to_csv(cache_path, index=False)
            print(f"  -> Fetched {len(df):,} records from {source_label} API (cached)")
            return df
        except Exception as exc:
            print(f"  -> {source_label} API unavailable: {exc}")
            if cache_path.exists():
                print(f"     Falling back to cached file ({cache_path.name})")
            else:
                raise RuntimeError(
                    f"No cached data and API unreachable for {source_label}"
                ) from exc
    else:
        print(f"  -> Using cached {source_label} data (<{CACHE_MAX_AGE_HOURS}h old)")

    return pd.read_csv(cache_path)


# ╔═══════════════════════════════════════════════════════════╗
# ║  OECD  --  SDMX REST API  (DAC2A dataflow)              ║
# ╚═══════════════════════════════════════════════════════════╝

# DAC2A MEASURE codes we request and how they map to sectors/instruments
# Columns returned: "DONOR: Donor", "RECIPIENT: Recipient",
#   "MEASURE: Measure", "UNIT_MEASURE: ...", "PRICE_BASE: ...",
#   "TIME_PERIOD: ...", "OBS_VALUE", "FLOW_TYPE: ...", etc.
_OECD_MEASURE_MAP: dict[str, tuple[str, str]] = {
    # code: (dashboard_sector, dashboard_instrument)
    "206": ("Cross-cutting",        "Grant"),          # ODA disbursements
    "201": ("Cross-cutting",        "Grant"),          # ODA grants, disbursements
    "204": ("Cross-cutting",        "Concessional Loan"),  # Gross ODA Loans
    "207": ("Energy Efficiency",    "Grant"),          # Technical cooperation
    "216": ("Climate Adaptation",   "Grant"),          # Humanitarian aid
    "217": ("Renewable Energy",     "Equity"),         # ODA equity investment
}

# ODA measures to request (climate-relevant flows)
_OECD_MEASURES = list(_OECD_MEASURE_MAP.keys())


def _extract_code(label: str) -> str:
    """Extract the code prefix from an SDMX 'CODE: Label' string."""
    return str(label).split(":")[0].strip()


def _fetch_oecd_api() -> pd.DataFrame:
    """
    Query the OECD SDMX REST API for DAC2A ODA disbursements.

    Dataflow : ``OECD.DCD.FSD,DSD_DAC2@DF_DAC2A,1.0``
    Dimensions: DONOR . RECIPIENT . MEASURE . UNIT_MEASURE . PRICE_BASE

    Reference
    ---------
    https://data-explorer.oecd.org/
    https://sdmx.oecd.org/public/rest/
    """
    base = "https://sdmx.oecd.org/public/rest/data"

    recipients = "+".join(sorted(COUNTRY_META.keys()))
    measures = "+".join(_OECD_MEASURES)

    # DAC2A v1.0 dimensions: DONOR.RECIPIENT.MEASURE.UNIT_MEASURE.PRICE_BASE
    url = (
        f"{base}/OECD.DCD.FSD,DSD_DAC2@DF_DAC2A,1.0/"
        f".{recipients}.{measures}.USD.V"
        f"?startPeriod=2013&endPeriod=2024"
    )

    headers = {
        "Accept": "application/vnd.sdmx.data+csv;file=true;labels=both",
    }

    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text))

    # Column names include labels: "RECIPIENT: Recipient", "MEASURE: Measure", etc.
    # Find the actual column names
    recip_col = [c for c in raw.columns if "RECIPIENT" in c.upper()][0]
    measure_col = [c for c in raw.columns if "MEASURE" in c.upper()][0]
    time_col = [c for c in raw.columns if "TIME_PERIOD" in c.upper()][0]

    rows: list[dict] = []
    for _, r in raw.iterrows():
        iso3 = _extract_code(str(r[recip_col]))
        if iso3 not in COUNTRY_META:
            continue

        obs = r.get("OBS_VALUE", None)
        if pd.isna(obs) or float(obs) <= 0:
            continue

        year = int(r[time_col])
        if year < 2013:
            continue

        country, region = COUNTRY_META[iso3]

        measure_code = _extract_code(str(r[measure_col]))
        sector, instrument = _OECD_MEASURE_MAP.get(
            measure_code, ("Cross-cutting", "Mixed/Other")
        )

        rows.append({
            "year": year,
            "region": region,
            "country": country,
            "sector": sector,
            "instrument_type": instrument,
            "amount_usd_mn": round(float(obs), 2),
        })

    if not rows:
        raise ValueError("OECD API returned no usable climate-finance records")

    return pd.DataFrame(rows)


# ╔═══════════════════════════════════════════════════════════╗
# ║  World Bank  --  Projects Search API                     ║
# ╚═══════════════════════════════════════════════════════════╝

_WB_SECTOR_MAP: dict[str, str] = {
    "energy":          "Renewable Energy",
    "renewable":       "Renewable Energy",
    "transport":       "Sustainable Transport",
    "water":           "Water & Waste Management",
    "agriculture":     "Forestry & Land Use",
    "forest":          "Forestry & Land Use",
    "environment":     "Climate Adaptation",
    "climate":         "Climate Adaptation",
    "health":          "Climate Adaptation",
    "industry":        "Energy Efficiency",
    "public admin":    "Cross-cutting",
    "education":       "Cross-cutting",
    "information":     "Cross-cutting",
    "finance":         "Cross-cutting",
    "social":          "Cross-cutting",
}

_WB_INSTRUMENT_MAP: dict[str, str] = {
    "investment project financing":     "Concessional Loan",
    "development policy financing":     "Grant",
    "program-for-results financing":    "Concessional Loan",
    "investment project":               "Concessional Loan",
    "specific investment loan":         "Concessional Loan",
    "technical assistance loan":        "Grant",
    "financial intermediary loan":      "Non-Concessional Loan",
    "adaptable program loan":          "Concessional Loan",
    "learning and innovation loan":     "Concessional Loan",
    "emergency recovery loan":          "Grant",
}


def _map_wb_sector(sector_str: str) -> str:
    """Best-effort mapping from WB sector name to dashboard sector."""
    if not sector_str:
        return "Cross-cutting"
    lower = sector_str.lower()
    for key, mapped in _WB_SECTOR_MAP.items():
        if key in lower:
            return mapped
    return "Cross-cutting"


def _map_wb_instrument(instr_str: str) -> str:
    """Map WB lending instrument to dashboard instrument type."""
    if not instr_str:
        return "Mixed/Other"
    lower = instr_str.lower()
    for key, mapped in _WB_INSTRUMENT_MAP.items():
        if key in lower:
            return mapped
    return "Mixed/Other"


def _parse_wb_amount(val) -> float:
    """Parse a World Bank amount string like '250,000,000' to a float."""
    if val is None:
        return 0.0
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _fetch_world_bank_api() -> pd.DataFrame:
    """
    Query the World Bank Projects Search API for climate-tagged projects.

    Reference
    ---------
    https://search.worldbank.org/api/v2/projects
    """
    all_rows: list[dict] = []
    page_size = 500
    offset = 0
    max_pages = 10

    target_countries = set(_NAME_TO_REGION.keys())

    for _ in range(max_pages):
        url = (
            "https://search.worldbank.org/api/v2/projects"
            f"?format=json"
            f"&flds=id,project_name,boardapprovaldate,totalamt,"
            f"countryshortname,sector1,sector,lendinginstr,theme1"
            f"&theme_exact=Climate+change"
            f"&rows={page_size}"
            f"&os={offset}"
        )

        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        total = int(data.get("total", 0))
        projects = data.get("projects", {})

        if not projects:
            break

        for proj_id, proj in projects.items():
            if isinstance(proj, str):
                continue

            country_name = (proj.get("countryshortname") or "").strip()
            country_list = [c.strip() for c in country_name.split(";") if c.strip()]

            for cname in country_list:
                if cname not in target_countries:
                    continue

                region = _NAME_TO_REGION.get(cname, "Unspecified")

                # Parse approval year
                approval_date = proj.get("boardapprovaldate", "")
                if not approval_date:
                    continue
                try:
                    year = int(str(approval_date)[:4])
                except (ValueError, IndexError):
                    continue
                if year < 2013 or year > 2025:
                    continue

                # Parse amount (may have commas: "250,000,000")
                amount_usd = _parse_wb_amount(proj.get("totalamt"))
                if amount_usd <= 0:
                    continue

                # Split multi-country projects evenly, convert to millions
                amount_mn = round(amount_usd / len(country_list) / 1e6, 2)
                if amount_mn <= 0:
                    continue

                # Sector mapping — try sector1 dict, then sector list
                sector_raw = ""
                sector_obj = proj.get("sector1")
                if isinstance(sector_obj, dict):
                    sector_raw = sector_obj.get("Name", "")
                elif isinstance(sector_obj, str):
                    sector_raw = sector_obj

                if not sector_raw:
                    sector_list = proj.get("sector")
                    if isinstance(sector_list, list) and sector_list:
                        first = sector_list[0]
                        sector_raw = first.get("Name", "") if isinstance(first, dict) else str(first)

                sector = _map_wb_sector(sector_raw)

                # Instrument mapping
                instr_raw = proj.get("lendinginstr") or ""
                if isinstance(instr_raw, dict):
                    instr_raw = instr_raw.get("Name", "")
                instrument = _map_wb_instrument(str(instr_raw))

                all_rows.append({
                    "year": year,
                    "region": region,
                    "country": cname,
                    "sector": sector,
                    "instrument": instrument,
                    "amount": amount_mn,
                })

        offset += page_size
        if offset >= total:
            break

        time.sleep(0.3)  # polite rate-limiting

    if not all_rows:
        raise ValueError("World Bank Projects API returned no usable records")

    return pd.DataFrame(all_rows)


# ╔═══════════════════════════════════════════════════════════╗
# ║  IRENA  --  synthetic data (no public API)               ║
# ╚═══════════════════════════════════════════════════════════╝

def _generate_irena_data() -> pd.DataFrame:
    """Generate realistic IRENA-style renewable energy investment data."""
    np.random.seed(42)

    years = list(range(2013, 2026))
    regions = {
        "Sub-Saharan Africa": ["Kenya", "Nigeria", "South Africa", "Ethiopia", "Ghana", "Tanzania"],
        "South Asia": ["India", "Bangladesh", "Pakistan", "Sri Lanka", "Nepal"],
        "East Asia & Pacific": ["China", "Indonesia", "Vietnam", "Philippines", "Thailand"],
        "Latin America & Caribbean": ["Brazil", "Mexico", "Colombia", "Chile", "Argentina"],
        "Europe & Central Asia": ["Turkey", "Ukraine", "Poland", "Romania", "Kazakhstan"],
        "Middle East & North Africa": ["Morocco", "Egypt", "Jordan", "Tunisia"],
    }
    instruments = ["Equity", "Non-Concessional Loan", "Bond", "Grant"]
    instrument_probs = [0.35, 0.25, 0.25, 0.15]

    rows: list[dict] = []
    for year in years:
        growth = (1.12 ** (year - 2013)) * np.random.uniform(0.85, 1.15)
        for region, countries in regions.items():
            for country in countries:
                for _ in range(np.random.randint(1, 3)):
                    base = np.random.exponential(scale=150)
                    rows.append({
                        "year": year,
                        "region": region,
                        "country": country,
                        "sector": "Renewable Energy",
                        "finance_type": np.random.choice(instruments, p=instrument_probs),
                        "value_usd_mn": round(base * growth * 1.2, 2),
                    })

    return pd.DataFrame(rows)


# ╔═══════════════════════════════════════════════════════════╗
# ║  Public loaders                                          ║
# ╚═══════════════════════════════════════════════════════════╝

def load_oecd(path: Path | None = None) -> pd.DataFrame:
    """Load OECD climate finance data (API with cache fallback)."""
    cache_path = path or RAW_DIR / "oecd_climate_finance.csv"
    df = _load_with_api(_fetch_oecd_api, cache_path, "OECD")
    df["source"] = "OECD"
    return df


def load_world_bank(path: Path | None = None) -> pd.DataFrame:
    """Load World Bank climate project data (API with cache fallback)."""
    cache_path = path or RAW_DIR / "world_bank_climate_finance.csv"
    df = _load_with_api(_fetch_world_bank_api, cache_path, "World Bank")
    df["source"] = "World Bank"
    return df


def load_irena(path: Path | None = None) -> pd.DataFrame:
    """Load IRENA data (generated -- no public API available)."""
    cache_path = path or RAW_DIR / "irena_renewable_investment.csv"

    if not cache_path.exists():
        df = _generate_irena_data()
        df.to_csv(cache_path, index=False)
        print(f"  -> Generated {len(df):,} synthetic IRENA records (cached)")
    else:
        print(f"  -> Using cached IRENA data ({cache_path.name})")
        df = pd.read_csv(cache_path)

    df["source"] = "IRENA"
    return df


# ╔═══════════════════════════════════════════════════════════╗
# ║  Orchestration                                           ║
# ╚═══════════════════════════════════════════════════════════╝

def ingest_all_sources() -> pd.DataFrame:
    """Ingest and concatenate all climate finance data sources."""
    frames: list[pd.DataFrame] = []
    loaders = [load_oecd, load_irena, load_world_bank]

    for loader in loaders:
        try:
            df = loader()
            print(f"  Loaded {len(df):,} records from {df['source'].iloc[0]}")
            frames.append(df)
        except Exception as e:
            print(f"  Warning: {e}")

    if not frames:
        raise RuntimeError("No data sources could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total ingested: {len(combined):,} records")
    return combined
