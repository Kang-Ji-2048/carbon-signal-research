# Climate Finance Dashboard

An ETL pipeline and interactive dashboard that unifies global climate-finance data from **three sources — OECD, World Bank, and IRENA** — into a single clean dataset, then visualises investment flows by sector, region, and instrument type. Ships both as a live Dash app and as a static, deployable site.

## Features

Six interactive visualisations plus four KPI cards and cross-filtering (year-range slider, region dropdown, sector dropdown):

1. **Stacked area chart** — climate-finance trends by sector over time
2. **Donut chart** — sector distribution
3. **Grouped bar chart** — investment by region over time
4. **Treemap** — finance by instrument type & sector
5. **Heatmap** — region × sector investment intensity
6. **Line chart** — data-source comparison (OECD vs IRENA vs World Bank)

## Architecture

```
OECD / World Bank / IRENA  ->  ingest  ->  clean  ->  aggregate  ->  processed CSV
                                                                        |
                                          +-----------------------------+-----------------------------+
                                          |                                                           |
                                   Dash app (live)                                     build_static.py -> JSON + static site
                                   python app.py                                       (deploy dist/ to Netlify)
```

The ETL stages live in `etl/` (`ingest.py`, `clean.py`, `aggregate.py`, orchestrated by `pipeline.py`). The unified dataset is keyed on `year`, `region`, `country`, `sector`, `instrument_type`, `amount_usd_mn`, and `source`.

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3 |
| Data | pandas, numpy |
| Visualisation | Plotly, Dash, dash-bootstrap-components |
| Ingestion | requests, openpyxl |
| Static build / deploy | custom `build_static.py` → JSON, Netlify |

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the sample dataset (see note below), then run the ETL + dashboard
python generate_sample_data.py
python app.py --etl

# Subsequent runs — just launch the dashboard
python app.py

# Run the ETL only, no dashboard
python app.py --etl-only
```

The dashboard serves at `http://localhost:8050` by default (`--port` to change).

## Static build (Netlify)

```bash
python build_static.py --etl   # re-run ETL, then export JSON + assets into dist/
cd dist && python -m http.server 8080   # preview locally
```

Deploy the generated `dist/` directory to Netlify (`netlify.toml` included).

## Data note

IRENA has no publicly available API, so the dataset used here is **synthetic**, produced by `generate_sample_data.py` with realistic sectors, regions, instrument types, and value distributions. The ETL, aggregation, and visualisation logic are designed to run identically against real OECD/World Bank/IRENA extracts.

## Configuration

| Flag | Description |
|------|-------------|
| `--etl` | Run the ETL pipeline before launching the dashboard |
| `--etl-only` | Run the ETL pipeline only (no dashboard) |
| `--port` | Dashboard port (default: 8050) |
| `--debug` | Run the Dash server in debug mode |
