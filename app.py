"""
Carbon Signal Research — Main Entry Point

Usage:
    # First run: fetch market data, run ETL + research, launch dashboard
    python app.py --refresh

    # Subsequent runs: use cached data
    python app.py

    # Rebuild the research results only (no dashboard)
    python app.py --research-only
"""

import argparse
import logging

from research.report import load_report, save_report


def main():
    parser = argparse.ArgumentParser(description="Carbon Signal Research & Dashboard")
    parser.add_argument("--refresh", action="store_true", help="Re-download market data before running")
    parser.add_argument("--research-only", action="store_true", help="Rebuild results, do not launch the dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.refresh or args.research_only:
        print("=" * 64)
        print("  RUNNING ETL + RESEARCH PIPELINE")
        print("=" * 64)
        report = save_report(refresh=args.refresh)
    else:
        report = load_report()

    if args.research_only:
        oos = report["performance"]["out_of_sample"]
        test = report["predictive_test"]
        print(f"\nOut-of-sample Sharpe : {oos['sharpe']:.3f}")
        print(f"Predictive t-stat    : {test['t_stat']:.3f}  (p = {test['p_value']:.3f})")
        return

    from dashboard.app import create_app

    print("=" * 64)
    print("  LAUNCHING DASHBOARD")
    print(f"  http://localhost:{args.port}")
    print("=" * 64)

    app = create_app(report)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
