"""Fetch and cache daily adjusted closes.

Network access is the only non-deterministic part of this project, so it is
isolated here: everything downstream reads the cached CSV. Re-running the
research offline is therefore bit-for-bit reproducible.
"""

from __future__ import annotations

import logging
import os
import ssl
import subprocess
import time
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)


def _build_ca_bundle() -> Path | None:
    """Return a CA bundle that trusts this machine's TLS-inspection root.

    Corporate AV/proxies re-sign HTTPS with a private root. ``truststore`` teaches
    Python's ``ssl`` module about the OS trust store, but ``yfinance`` fetches via
    ``curl_cffi`` (libcurl), which consults its own bundle and ignores that patch.
    So we export the Windows cert store, concatenate it with certifi, and point
    libcurl at the result via ``CURL_CA_BUNDLE``.

    Returns None on non-Windows platforms or if export fails, in which case the
    default bundle is used unchanged.
    """
    combined = config.CERTS_DIR / "combined-ca-bundle.pem"
    if combined.exists():
        return combined
    if os.name != "nt":
        return None

    config.CERTS_DIR.mkdir(parents=True, exist_ok=True)
    win_pem = config.CERTS_DIR / "windows-ca-bundle.pem"

    ps = (
        "$sb = New-Object System.Text.StringBuilder; "
        "foreach ($s in @('Root','CA')) { "
        "Get-ChildItem \"Cert:\\LocalMachine\\$s\", \"Cert:\\CurrentUser\\$s\" "
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "$b = [System.Convert]::ToBase64String($_.RawData, 'InsertLineBreaks'); "
        "[void]$sb.AppendLine('-----BEGIN CERTIFICATE-----'); "
        "[void]$sb.AppendLine($b); "
        "[void]$sb.AppendLine('-----END CERTIFICATE-----') } }; "
        f"Set-Content -Path '{win_pem}' -Value $sb.ToString() -Encoding ascii"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("Could not export Windows CA store (%s); using default bundle", exc)
        return None

    try:
        import certifi
        base = Path(certifi.where()).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        base = ""
    combined.write_text(
        base + "\n" + win_pem.read_text(encoding="ascii", errors="ignore"),
        encoding="ascii", errors="ignore",
    )
    return combined


def _prepare_tls() -> None:
    """Make HTTPS work for both the ``ssl`` module and libcurl."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        log.debug("truststore not installed; relying on default SSL context")

    bundle = _build_ca_bundle()
    if bundle is not None:
        for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(var, str(bundle))


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    retries: int = 3,
) -> pd.DataFrame:
    """Download adjusted closes for ``tickers`` as a wide date-indexed frame."""
    tickers = tickers or config.ALL_TICKERS
    _prepare_tls()

    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers, start=start, auto_adjust=True, progress=False,
                group_by="column",
            )
            if raw is None or raw.empty:
                raise RuntimeError("provider returned an empty frame")

            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            close = close.copy()
            missing = [t for t in tickers if t not in close.columns]
            if missing:
                raise RuntimeError(f"no data returned for {missing}")

            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.index.name = "date"
            return close[tickers].sort_index()
        except Exception as exc:  # noqa: BLE001 - retry any provider failure
            last_error = exc
            log.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Price download failed after {retries} attempts") from last_error


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Return cached prices, downloading them if absent or if ``refresh``."""
    if config.PRICES_CSV.exists() and not refresh:
        df = pd.read_csv(config.PRICES_CSV, index_col="date", parse_dates=["date"])
        log.info("Loaded %d cached rows from %s", len(df), config.PRICES_CSV)
        return df

    df = fetch_prices()
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PRICES_CSV)
    log.info("Cached %d rows to %s", len(df), config.PRICES_CSV)
    return df
