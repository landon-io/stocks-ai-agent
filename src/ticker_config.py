"""Sector theme definitions and ticker utilities."""

# Candidate universe aligned with watchlist sectors (ETF-AI Infra / ETF-Tech / ETF-Energy / ETF-Others).
# Each ticker appears in exactly one theme. Run download_data.py after changes.
TICKER_CONFIG: dict[str, list[str]] = {
    # ETF-AI Infra (+ legacy semis / infrastructure)
    "AI_INFRA": ["RACK", "PAVE", "AIPO", "SMH", "SOXX", "SOXQ", "BOTZ"],
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

BENCHMARK_SPLIT = ("QQQ", "SMH")

# Backtest / analysis window (months). Indicators still warm up on full CSV history.
ANALYSIS_LOOKBACK_MONTHS = 12


def execution_ticker(signal_ticker: str) -> str:
    """Return the ETF to trade for a given signal ticker."""
    return LEVERAGED_EXEC.get(signal_ticker, signal_ticker)


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
