"""
Build the static Netlify site from the ETL pipeline output.

Runs the ETL pipeline (if needed), then exports processed data as JSON
into dist/data/ for the client-side dashboard.

Usage
-----
    python build_static.py          # build from existing processed data
    python build_static.py --etl    # re-run ETL first, then build
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"
DIST_DIR = PROJ_ROOT / "dist"


def run_etl():
    """Run the ETL pipeline to refresh processed data."""
    from etl.pipeline import run_pipeline
    run_pipeline()


def export_data():
    """Export processed CSV data to JSON files in dist/data/."""
    data_dir = DIST_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load the detail dataset
    detail_path = PROCESSED_DIR / "climate_finance_clean.csv"
    if not detail_path.exists():
        raise FileNotFoundError(
            f"{detail_path} not found. Run 'python build_static.py --etl' first."
        )
    detail = pd.read_csv(detail_path)

    # Extract metadata BEFORE renaming columns
    meta = {
        "years": sorted(int(y) for y in detail["year"].unique()),
        "regions": sorted(detail["region"].unique().tolist()),
        "sectors": sorted(detail["sector"].unique().tolist()),
        "instruments": sorted(detail["instrument_type"].unique().tolist()),
        "sources": sorted(detail["source"].unique().tolist()),
        "sectorColors": {
            "Renewable Energy": "#2ecc71",
            "Energy Efficiency": "#3498db",
            "Sustainable Transport": "#9b59b6",
            "Climate Adaptation": "#e67e22",
            "Forestry & Land Use": "#27ae60",
            "Water & Waste Management": "#1abc9c",
            "Cross-cutting": "#95a5a6",
        },
        "regionColors": {
            "Sub-Saharan Africa": "#e74c3c",
            "South Asia": "#f39c12",
            "East Asia & Pacific": "#2ecc71",
            "Latin America & Caribbean": "#3498db",
            "Europe & Central Asia": "#9b59b6",
            "Middle East & North Africa": "#1abc9c",
        },
    }
    meta_out = data_dir / "meta.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json: {len(meta['years'])} years, {len(meta['regions'])} regions")

    # Round amounts and use short keys to reduce JSON payload
    detail["amount_usd_mn"] = detail["amount_usd_mn"].round(1)
    detail = detail.rename(columns={
        "year": "y",
        "region": "r",
        "country": "c",
        "sector": "s",
        "instrument_type": "i",
        "amount_usd_mn": "a",
        "source": "src",
    })

    records = detail.to_dict(orient="records")
    detail_out = data_dir / "detail.json"
    with open(detail_out, "w") as f:
        json.dump(records, f, separators=(",", ":"))
    size_mb = detail_out.stat().st_size / (1024 * 1024)
    print(f"  detail.json: {len(records):,} records ({size_mb:.1f} MB)")

    return len(records)


def copy_static_assets():
    """Copy static HTML/JS/CSS into dist/."""
    static_src = PROJ_ROOT / "static"
    if not static_src.exists():
        print("  No static/ directory found, skipping asset copy")
        return

    for sub in ["js", "css"]:
        src = static_src / sub
        dst = DIST_DIR / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {sub}/ ({len(list(dst.rglob('*')))} files)")

    # Copy index.html
    index_src = static_src / "index.html"
    if index_src.exists():
        shutil.copy2(index_src, DIST_DIR / "index.html")
        print("  Copied index.html")


def main():
    parser = argparse.ArgumentParser(description="Build static Netlify site")
    parser.add_argument("--etl", action="store_true", help="Re-run ETL pipeline first")
    args = parser.parse_args()

    print("=" * 60)
    print("  BUILDING STATIC SITE")
    print("=" * 60)

    if args.etl:
        print("\n[1/3] Running ETL pipeline...")
        run_etl()
    else:
        print("\n[1/3] Skipping ETL (use --etl to refresh data)")

    print("\n[2/3] Exporting data to JSON...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    n_records = export_data()

    print("\n[3/3] Copying static assets...")
    copy_static_assets()

    print(f"\nBuild complete. {n_records:,} records exported to dist/")
    print(f"Deploy dist/ to Netlify or test with:")
    print(f"  cd dist && python -m http.server 8080")


if __name__ == "__main__":
    main()
