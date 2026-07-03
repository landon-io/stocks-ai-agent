"""Sector theme definitions and ticker utilities."""

# =============================================================================
# Production strategy: QQQ park + swing sector rotation (sandbox-aligned)
# =============================================================================
PARK_TICKER = "QQQ"
PARK_SYMBOLS = frozenset({PARK_TICKER, "QQQM"})

SWING_SECTORS: dict[str, str] = {
    "AI_INFRA": "SMH",
    "ENERGY": "XLE",
    "UTILITIES": "XLU",
    "MACRO_ROTATION": "XLF",
}

RSI_PERIOD = 7
RSI_OVERSOLD_THRESHOLD = 30.0
TAKE_PROFIT_RSI = 70.0
SMA_TREND = 200
STOP_LOSS_PCT = 0.025
SWING_ALLOCATION = 0.85
MAXIMUM_SWING_POSITIONS = 4
SLIPPAGE_PCT = 0.0005
STARTING_CAPITAL = 10_000.0

# Candidate universe aligned with watchlist sectors (ETF-AI Infra / ETF-Tech / ETF-Energy / ETF-Others).
# Each ticker appears in exactly one theme. Run download_data.py after changes.
TICKER_CONFIG: dict[str, list[str]] = {
    # ETF-AI Infra (+ legacy semis / infrastructure)
    # RACK excluded — yfinance history too short (~1mo); breaks 12mo backtest alignment.
    "AI_INFRA": ["PAVE", "AIPO", "SMH", "SOXX", "SOXQ", "BOTZ"],
    # ETF-Tech
    "TECH": ["IYW", "XLK", "QQQM", "XLC"],
    # ETF-Others / broad market
    "BROAD_MARKET": ["ITDE", "ITDG", "ITDF", "QQQ", "VOO", "SPY", "VT"],
    # ETF-Energy (+ utilities)
    "ENERGY": ["IXC", "VDE", "XOP", "XLE", "FENY", "IEO", "XLU"],
    # Legacy themes (still scanned if data exists)
    "PURE_INTERNET": ["FDN", "PNQI"],  # KWEB excluded — negative alpha in backtest
    "MACRO_ROTATION": ["XLF", "KRE", "XLRE", "VNQ"],
}

THEME_LEADERS: dict[str, str] = {
    theme: tickers[0] for theme, tickers in TICKER_CONFIG.items()
}

# Leveraged execution mapping for live/paper trading (signal on 1x, fill on 3x).
LEVERAGED_EXEC: dict[str, str] = {
    "SMH": "SOXL",
    "SOXX": "SOXL",
    "QQQ": "TQQQ",
    "QQQM": "TQQQ",
}

# 50/50 buy-and-hold benchmark for analyzer backtests (less volatile than semis-heavy SMH).
BENCHMARK_SPLIT = ("QQQ", "SPY")

# Backtest / analysis window (months). Indicators still warm up on full CSV history.
ANALYSIS_LOOKBACK_MONTHS = 12


def execution_ticker(signal_ticker: str) -> str:
    """Return the ETF to trade for a given signal ticker."""
    return LEVERAGED_EXEC.get(signal_ticker, signal_ticker)


def swing_tickers() -> list[str]:
    """Park + swing signal tickers for production backtest/scanner."""
    return list(dict.fromkeys([PARK_TICKER, *SWING_SECTORS.values()]))


def scanner_tickers() -> list[str]:
    """Tickers required for daily scanner (swing universe + leveraged fills)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in swing_tickers():
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    for exec_ticker in LEVERAGED_EXEC.values():
        if exec_ticker not in seen:
            seen.add(exec_ticker)
            ordered.append(exec_ticker)
    return ordered


def swing_sector_for_ticker(ticker: str) -> str | None:
    """Map a signal or execution ticker to its swing sector name."""
    for sector, signal_ticker in SWING_SECTORS.items():
        if ticker == signal_ticker or ticker == execution_ticker(signal_ticker):
            return sector
    return None


def is_park_symbol(symbol: str) -> bool:
    return symbol in PARK_SYMBOLS


def iter_config_tickers() -> list[tuple[str, str]]:
    """Yield (theme, ticker) pairs preserving TICKER_CONFIG order."""
    pairs: list[tuple[str, str]] = []
    for theme, tickers in TICKER_CONFIG.items():
        for ticker in tickers:
            pairs.append((theme, ticker))
    return pairs


def all_tickers() -> list[str]:
    """Return every signal ticker plus any leveraged execution tickers."""
    seen: set[str] = set()
    ordered: list[str] = []
    for tickers in TICKER_CONFIG.values():
        for ticker in tickers:
            if ticker not in seen:
                seen.add(ticker)
                ordered.append(ticker)
    for exec_ticker in LEVERAGED_EXEC.values():
        if exec_ticker not in seen:
            seen.add(exec_ticker)
            ordered.append(exec_ticker)
    return ordered


def theme_for_ticker(ticker: str) -> str | None:
    """Return the theme name that contains this ticker, if any."""
    for theme, tickers in TICKER_CONFIG.items():
        if ticker in tickers:
            return theme
    return None


def rotation_tickers() -> list[str]:
    """Theme leader tickers that drive the rotation backtest."""
    return list(THEME_LEADERS.values())
