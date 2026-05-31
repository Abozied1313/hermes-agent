"""Production-oriented EGX quote and Telegram portfolio helpers.

Quotes come from Yahoo Finance through yfinance. Portfolios are deliberately
keyed by Telegram ``user_id`` rather than a chat ID so users in the same group
cannot see or mutate one another's holdings.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from utils import atomic_json_write

logger = logging.getLogger(__name__)

_EGX_SUFFIX = ".CA"
_SYMBOL_RE = re.compile(r"^[A-Z0-9-]{1,20}$")


class QuoteUnavailableError(RuntimeError):
    """Raised when neither a live quote nor a cached quote is available."""


class PortfolioError(ValueError):
    """Raised for invalid portfolio input or unreadable portfolio state."""


def normalize_egx_symbol(symbol: str) -> str:
    """Return a validated Yahoo Finance symbol for an EGX-listed security."""
    normalized = symbol.strip().upper()
    if normalized.endswith(_EGX_SUFFIX):
        normalized = normalized[: -len(_EGX_SUFFIX)]
    if not _SYMBOL_RE.fullmatch(normalized):
        raise ValueError("EGX symbol must contain 1-20 letters, numbers, or hyphens")
    return f"{normalized}{_EGX_SUFFIX}"


def default_financial_home() -> Path:
    """Return the financial agent's state directory without writing to it."""
    if configured := os.getenv("FINANCIAL_AGENT_HOME"):
        return Path(configured).expanduser()
    hermes_home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
    return hermes_home / "financial_agent"


@dataclass(frozen=True)
class Quote:
    """An EGX quote, including whether it came from the stale-cache fallback."""

    symbol: str
    price: float
    currency: str
    observed_at: str
    source: str
    stale: bool = False


class EGXQuoteService:
    """Fetch EGX prices from yfinance and fall back to the last cached quote."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        fetcher: Optional[Callable[[str], Quote]] = None,
    ) -> None:
        self.cache_path = cache_path or default_financial_home() / "quote_cache.json"
        self._fetcher = fetcher or self._fetch_yfinance_quote

    def get_quote(self, symbol: str) -> Quote:
        """Return a live quote, or a clearly marked stale quote after failures."""
        egx_symbol = normalize_egx_symbol(symbol)
        try:
            quote = self._fetcher(egx_symbol)
            if quote.symbol != egx_symbol:
                raise QuoteUnavailableError("quote provider returned an unexpected symbol")
            self._cache_quote(quote)
            return quote
        except Exception as exc:
            logger.warning("Live EGX quote unavailable for %s: %s", egx_symbol, exc)
            cached = self._load_cache().get(egx_symbol)
            if cached:
                return Quote(**{**cached, "stale": True, "source": "cache"})
            raise QuoteUnavailableError(
                f"Live quote unavailable for {egx_symbol}; no cached quote exists"
            ) from exc

    @staticmethod
    def _fetch_yfinance_quote(symbol: str) -> Quote:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if history.empty or "Close" not in history:
            raise QuoteUnavailableError(f"Yahoo Finance returned no EGX data for {symbol}")
        closes = history["Close"].dropna()
        if closes.empty:
            raise QuoteUnavailableError(f"Yahoo Finance returned no close price for {symbol}")
        observed = closes.index[-1]
        observed_at = observed.isoformat() if hasattr(observed, "isoformat") else str(observed)
        return Quote(
            symbol=symbol,
            price=float(closes.iloc[-1]),
            currency="EGP",
            observed_at=observed_at,
            source="yfinance",
        )

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable EGX quote cache at %s", self.cache_path)
            return {}

    def _cache_quote(self, quote: Quote) -> None:
        data = self._load_cache()
        data[quote.symbol] = asdict(quote)
        atomic_json_write(self.cache_path, data)


class TelegramPortfolioStore:
    """Persist holdings in separate namespaces for each Telegram user ID."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or default_financial_home() / "portfolios.json"

    def get_portfolio(self, user_id: str) -> dict[str, float]:
        """Return a copy of one Telegram user's holdings."""
        owner = self._normalize_user_id(user_id)
        return dict(self._load().get(owner, {}))

    def add_holding(self, user_id: str, symbol: str, shares: float) -> dict[str, float]:
        """Add shares to one Telegram user's isolated portfolio."""
        owner = self._normalize_user_id(user_id)
        quantity = self._normalize_shares(shares)
        egx_symbol = normalize_egx_symbol(symbol)
        portfolios = self._load()
        holdings = portfolios.setdefault(owner, {})
        holdings[egx_symbol] = holdings.get(egx_symbol, 0.0) + quantity
        atomic_json_write(self.path, portfolios)
        return dict(holdings)

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        owner = str(user_id).strip()
        if not owner:
            raise PortfolioError("Telegram user_id is required")
        return owner

    @staticmethod
    def _normalize_shares(shares: float) -> float:
        try:
            quantity = float(shares)
        except (TypeError, ValueError) as exc:
            raise PortfolioError("shares must be a positive number") from exc
        if not math.isfinite(quantity) or quantity <= 0:
            raise PortfolioError("shares must be a positive finite number")
        return quantity

    def _load(self) -> dict[str, dict[str, float]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortfolioError(f"Portfolio store is unreadable: {self.path}") from exc
        if not isinstance(data, dict):
            raise PortfolioError(f"Portfolio store has invalid contents: {self.path}")
        return data


class HermesFinancialAgent:
    """Small facade used by demos and Telegram command handlers."""

    def __init__(
        self,
        quotes: Optional[EGXQuoteService] = None,
        portfolios: Optional[TelegramPortfolioStore] = None,
    ) -> None:
        self.quotes = quotes or EGXQuoteService()
        self.portfolios = portfolios or TelegramPortfolioStore()

    def portfolio_snapshot(self, user_id: str) -> dict:
        """Value one user's portfolio without exposing any other user's state."""
        positions = []
        total_egp = 0.0
        for symbol, shares in sorted(self.portfolios.get_portfolio(user_id).items()):
            try:
                quote = self.quotes.get_quote(symbol)
            except QuoteUnavailableError as exc:
                positions.append({"symbol": symbol, "shares": shares, "error": str(exc)})
                continue
            value = shares * quote.price
            total_egp += value
            positions.append({**asdict(quote), "shares": shares, "value_egp": value})
        return {"user_id": str(user_id), "positions": positions, "total_egp": total_egp}

    def generate_daily_report(self, user_id: str, report_date: Optional[str] = None) -> dict:
        """Generate a dated report for one Telegram user's isolated portfolio."""
        snapshot = self.portfolio_snapshot(user_id)
        return {
            "report_date": report_date or datetime.now(timezone.utc).date().isoformat(),
            **snapshot,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes EGX financial-agent demo")
    commands = parser.add_subparsers(dest="command", required=True)
    quote = commands.add_parser("quote", help="Fetch a real EGX quote")
    quote.add_argument("symbol", help="EGX ticker, with or without the .CA suffix")
    add = commands.add_parser("add", help="Add shares to one Telegram user's portfolio")
    add.add_argument("--user-id", required=True, help="Telegram user ID")
    add.add_argument("symbol", help="EGX ticker, with or without the .CA suffix")
    add.add_argument("shares", type=float, help="Positive number of shares")
    portfolio = commands.add_parser("portfolio", help="Value one Telegram user's portfolio")
    portfolio.add_argument("--user-id", required=True, help="Telegram user ID")
    report = commands.add_parser("daily-report", help="Generate one Telegram user's daily report")
    report.add_argument("--user-id", required=True, help="Telegram user ID")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    agent = HermesFinancialAgent()
    try:
        if args.command == "quote":
            result = asdict(agent.quotes.get_quote(args.symbol))
        elif args.command == "add":
            result = {"user_id": args.user_id, "holdings": agent.portfolios.add_holding(args.user_id, args.symbol, args.shares)}
        elif args.command == "portfolio":
            result = agent.portfolio_snapshot(args.user_id)
        else:
            result = agent.generate_daily_report(args.user_id)
    except (PortfolioError, QuoteUnavailableError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
