"""
Climate Finance Dashboard — Main Entry Point

Usage:
    # First run: generate sample data and run ETL
    python generate_sample_data.py
    python app.py --etl

    # Subsequent runs: just launch the dashboard
    python app.py

    # Run ETL only (no dashboard)
    python app.py --etl-only
"""

import argparse
from pathlib import Path

from etl.pipeline import run_pipeline
from dashboard.app import load_data, create_app

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Climate Finance ETL & Dashboard")
    parser.add_argument("--etl", action="store_true", help="Run ETL pipeline before launching dashboard")
    parser.add_argument("--etl-only", action="store_true", help="Run ETL pipeline only (no dashboard)")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    # Check if processed data exists
    clean_file = PROCESSED_DIR / "climate_finance_clean.csv"
    need_etl = args.etl or args.etl_only or not clean_file.exists()

    if need_etl:
        print("=" * 60)
        print("  CLIMATE FINANCE ETL PIPELINE")
        print("=" * 60)
        run_pipeline()

    if args.etl_only:
        return

    # Launch dashboard
    print("=" * 60)
    print("  LAUNCHING DASHBOARD")
    print(f"  http://localhost:{args.port}")
    print("=" * 60)

    views = load_data()
    app = create_app(views)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
