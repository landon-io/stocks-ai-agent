"""Download historical ETF data and save to /data as CSV."""

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from analyzer import find_missing_tickers
from ticker_config import TICKER_CONFIG, all_tickers, theme_for_ticker

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARK_TICKER = "QQQ"

COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
INTERVAL = "1d"
YEARS_OF_HISTORY = 3
REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def date_range() -> tuple[date, date]:
    """Return (start, end_exclusive) for the last 3 years through today."""
    # yfinance treats `end` as exclusive — add 1 day to include today's bar.
    end_exclusive = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=YEARS_OF_HISTORY * 365)
    return start, end_exclusive


def download_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker with retries."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", ticker, attempt, MAX_RETRIES)
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            missing = [col for col in COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for {ticker}: {missing}")

            return df[COLUMNS].copy()

        except Exception as exc:
            last_error = exc
            logger.warning("Failed to download %s: %s", ticker, exc)
            if attempt < MAX_RETRIES:
                logger.info("Retrying %s in %d seconds...", ticker, REQUEST_DELAY_SECONDS)
                time.sleep(REQUEST_DELAY_SECONDS)

    raise RuntimeError(f"Unable to download {ticker} after {MAX_RETRIES} attempts") from last_error


def save_ticker_csv(ticker: str, df: pd.DataFrame) -> Path:
    """Save a ticker's DataFrame to /data/{ticker}.csv."""
    output_path = DATA_DIR / f"{ticker}.csv"
    df.to_csv(output_path, index=True)
    return output_path


def csv_last_date(ticker: str = BENCHMARK_TICKER) -> date | None:
    """Return the latest Date in a local CSV, or None if missing/empty."""
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    dates = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])["Date"]
    if dates.empty:
        return None
    return dates.iloc[-1].date()


def expected_completed_session(as_of: pd.Timestamp | None = None) -> date:
    """
    Latest NYSE session whose daily bar should be available locally.

    - Weekday after 4 PM ET → today's close
    - Otherwise → previous business day
    """
    now = as_of or pd.Timestamp.now(tz="America/New_York")
    session = now.normalize()
    if session.dayofweek >= 5:
        return (session - BDay(1)).date()
    if now.hour < 16:
        return (session - BDay(1)).date()
    return session.date()


def is_market_data_stale(ticker: str = BENCHMARK_TICKER) -> tuple[bool, date | None, date]:
    """True when local CSVs are missing or older than the expected completed session."""
    latest = csv_last_date(ticker)
    expected = expected_completed_session()
    if latest is None:
        return True, None, expected
    return latest < expected, latest, expected


def run_download() -> tuple[list[str], list[str]]:
    """Bulk-download all configured tickers. Returns (succeeded, failed)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start, end = date_range()
    tickers_to_download = all_tickers()

    logger.info(
        "Starting bulk download for %d tickers across %d themes (%s to %s)",
        len(tickers_to_download),
        len(TICKER_CONFIG),
        start.isoformat(),
        end.isoformat(),
    )
    for theme, theme_tickers in TICKER_CONFIG.items():
        logger.info("  %s: %s", theme, ", ".join(theme_tickers))

    succeeded: list[str] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers_to_download):
        theme = theme_for_ticker(ticker) or "LEVERAGED_EXEC"
        try:
            df = download_ticker(ticker, start, end)
            output_path = save_ticker_csv(ticker, df)
            logger.info(
                "Saved %s [%s] | %d rows | %s",
                ticker,
                theme,
                len(df),
                output_path,
            )
            succeeded.append(ticker)
        except Exception as exc:
            logger.error("Skipping %s [%s] due to error: %s", ticker, theme, exc)
            failed.append(ticker)

        if i < len(tickers_to_download) - 1:
            logger.info("Waiting %d seconds before next request...", REQUEST_DELAY_SECONDS)
            time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        "Download complete. Succeeded: %d | Failed: %d",
        len(succeeded),
        len(failed),
    )
    if succeeded:
        logger.info("  OK: %s", ", ".join(succeeded))
    if failed:
        logger.error("  FAILED: %s", ", ".join(failed))
    return succeeded, failed


def ensure_fresh_data(force: bool = False) -> bool:
    """
    Refresh CSVs when files are missing or the benchmark bar is stale.

    Returns True if a download was run.
    """
    missing = find_missing_tickers(all_tickers())
    stale, latest, expected = is_market_data_stale()

    if not force and not missing and not stale:
        logger.info(
            "Market data is current through %s (expected %s).",
            latest,
            expected,
        )
        return False

    if missing:
        logger.info("Missing %d CSV file(s) — downloading universe...", len(missing))
    elif stale:
        logger.info(
            "Local data ends %s; refreshing through %s...",
            latest,
            expected,
        )

    _, failed = run_download()
    if failed:
        raise RuntimeError(
            f"Download failed for {len(failed)} ticker(s): {', '.join(failed)}"
        )
    return True


def main() -> None:
    _, failed = run_download()
    if failed:
        logger.error("Re-run download_data.py or fix invalid tickers before scanning.")
        sys.exit(1)


if __name__ == "__main__":
    main()
