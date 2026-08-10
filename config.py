"""Central configuration for the carbon-signal research project.

Every tunable lives here so that a reader can see the entire specification of the
study in one screen, and so no magic numbers hide inside the research code.
"""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CERTS_DIR = ROOT / ".certs"

PRICES_CSV = DATA_RAW / "prices.csv"
PANEL_CSV = DATA_PROCESSED / "panel.csv"
RESULTS_JSON = DATA_PROCESSED / "results.json"

# --- Universe -------------------------------------------------------------
# Green: clean-energy equity ETFs. Brown: fossil-fuel producers/services.
# Carbon: KRBN tracks a basket of carbon-allowance futures (EU ETS, CCA, RGGI),
# which makes it the tradeable proxy for "the price of carbon".
GREEN = ["ICLN", "TAN", "PBW"]
BROWN = ["XLE", "XOP"]
CARBON = "KRBN"
BENCHMARK = "SPY"

ALL_TICKERS = GREEN + BROWN + [CARBON, BENCHMARK]

START_DATE = "2010-01-01"

# --- Signal ---------------------------------------------------------------
CARBON_LOOKBACK = 60        # trading days of carbon momentum
VOL_LOOKBACK = 60           # trading days for realised-vol estimate
TARGET_VOL = 0.10           # annualised volatility target for the strategy
MAX_LEVERAGE = 2.0          # cap so vol-targeting cannot explode position size

# --- Backtest -------------------------------------------------------------
COST_BPS = 5.0              # one-way cost per leg, in basis points
LEGS = 2                    # long/short spread trades two legs per unit of turnover
OOS_SPLIT = 0.40            # first 40% of the sample is in-sample; rest is OOS

TRADING_DAYS = 252

# --- Robustness sweep -----------------------------------------------------
SWEEP_LOOKBACKS = [20, 40, 60, 90, 120]
SWEEP_COSTS = [0.0, 5.0, 10.0, 20.0]
