"""Smoke tests for the hackathon EGX financial agent."""

from financial_agent import (
    EGXQuoteService,
    HermesFinancialAgent,
    Quote,
    QuoteUnavailableError,
    TelegramPortfolioStore,
    normalize_egx_symbol,
)


def _live_quote(symbol: str) -> Quote:
    return Quote(
        symbol=symbol,
        price=42.5,
        currency="EGP",
        observed_at="2026-05-30T00:00:00+00:00",
        source="yfinance",
    )


def test_normalizes_egx_symbols():
    assert normalize_egx_symbol("comi") == "COMI.CA"
    assert normalize_egx_symbol("COMI.CA") == "COMI.CA"


def test_quote_service_uses_live_data_then_safe_stale_cache(tmp_path):
    cache_path = tmp_path / "quotes.json"
    live = EGXQuoteService(cache_path=cache_path, fetcher=_live_quote)
    assert live.get_quote("comi").source == "yfinance"

    def fail_fetch(_symbol: str) -> Quote:
        raise RuntimeError("network unavailable")

    fallback = EGXQuoteService(cache_path=cache_path, fetcher=fail_fetch).get_quote("COMI")
    assert fallback.symbol == "COMI.CA"
    assert fallback.price == 42.5
    assert fallback.source == "cache"
    assert fallback.stale is True


def test_quote_service_fails_closed_without_cache(tmp_path):
    service = EGXQuoteService(
        cache_path=tmp_path / "quotes.json",
        fetcher=lambda _symbol: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    try:
        service.get_quote("SWDY")
    except QuoteUnavailableError as exc:
        assert "no cached quote exists" in str(exc)
    else:
        raise AssertionError("Expected an explicit unavailable result")


def test_telegram_portfolios_are_isolated_by_user_id(tmp_path):
    store = TelegramPortfolioStore(tmp_path / "portfolios.json")
    store.add_holding(user_id="telegram-user-1", symbol="COMI", shares=2)
    store.add_holding(user_id="telegram-user-2", symbol="SWDY", shares=3)

    assert store.get_portfolio("telegram-user-1") == {"COMI.CA": 2.0}
    assert store.get_portfolio("telegram-user-2") == {"SWDY.CA": 3.0}


def test_portfolio_snapshot_only_values_requested_user(tmp_path):
    store = TelegramPortfolioStore(tmp_path / "portfolios.json")
    store.add_holding("100", "COMI", 2)
    store.add_holding("200", "SWDY", 100)
    agent = HermesFinancialAgent(
        quotes=EGXQuoteService(cache_path=tmp_path / "quotes.json", fetcher=_live_quote),
        portfolios=store,
    )

    snapshot = agent.portfolio_snapshot("100")
    assert snapshot["user_id"] == "100"
    assert snapshot["total_egp"] == 85.0
    assert [position["symbol"] for position in snapshot["positions"]] == ["COMI.CA"]
    assert snapshot["positions"][0]["allocation_pct"] == 100.0
    assert snapshot["insights"] == {
        "position_count": 1,
        "priced_position_count": 1,
        "unavailable_position_count": 0,
        "stale_position_count": 0,
        "quote_coverage_pct": 100.0,
        "largest_position": {"symbol": "COMI.CA", "allocation_pct": 100.0},
        "risk_flags": ["Concentration risk: COMI.CA is 100.00% of priced holdings"],
    }


def test_daily_report_generation_uses_offline_quote_fetcher(tmp_path):
    store = TelegramPortfolioStore(tmp_path / "portfolios.json")
    store.add_holding("telegram-user-1", "COMI", 2)
    agent = HermesFinancialAgent(
        quotes=EGXQuoteService(cache_path=tmp_path / "quotes.json", fetcher=_live_quote),
        portfolios=store,
    )

    report = agent.generate_daily_report("telegram-user-1", report_date="2026-05-31")
    assert report == {
        "report_date": "2026-05-31",
        "user_id": "telegram-user-1",
        "positions": [
            {
                "symbol": "COMI.CA",
                "price": 42.5,
                "currency": "EGP",
                "observed_at": "2026-05-30T00:00:00+00:00",
                "source": "yfinance",
                "stale": False,
                "shares": 2.0,
                "value_egp": 85.0,
                "allocation_pct": 100.0,
            }
        ],
        "total_egp": 85.0,
        "insights": {
            "position_count": 1,
            "priced_position_count": 1,
            "unavailable_position_count": 0,
            "stale_position_count": 0,
            "quote_coverage_pct": 100.0,
            "largest_position": {"symbol": "COMI.CA", "allocation_pct": 100.0},
            "risk_flags": ["Concentration risk: COMI.CA is 100.00% of priced holdings"],
        },
    }


def test_portfolio_insights_surface_stale_and_unavailable_quotes(tmp_path):
    store = TelegramPortfolioStore(tmp_path / "portfolios.json")
    store.add_holding("100", "COMI", 2)
    store.add_holding("100", "SWDY", 3)
    store.add_holding("100", "ORAS", 4)
    cache_path = tmp_path / "quotes.json"
    EGXQuoteService(cache_path=cache_path, fetcher=_live_quote).get_quote("SWDY")

    def fetch_quote(symbol: str) -> Quote:
        if symbol == "COMI.CA":
            return _live_quote(symbol)
        raise RuntimeError("offline")

    agent = HermesFinancialAgent(
        quotes=EGXQuoteService(cache_path=cache_path, fetcher=fetch_quote),
        portfolios=store,
    )

    snapshot = agent.portfolio_snapshot("100")
    assert snapshot["total_egp"] == 212.5
    assert snapshot["positions"][0]["allocation_pct"] == 40.0
    assert snapshot["positions"][2]["allocation_pct"] == 60.0
    assert "allocation_pct" not in snapshot["positions"][1]
    assert snapshot["insights"] == {
        "position_count": 3,
        "priced_position_count": 2,
        "unavailable_position_count": 1,
        "stale_position_count": 1,
        "quote_coverage_pct": 66.67,
        "largest_position": {"symbol": "SWDY.CA", "allocation_pct": 60.0},
        "risk_flags": [
            "Concentration risk: SWDY.CA is 60.00% of priced holdings",
            "Market data risk: 1 position(s) use stale cached quotes",
            "Valuation gap: 1 position(s) have no available quote",
        ],
    }


def test_empty_portfolio_insights_report_full_quote_coverage(tmp_path):
    agent = HermesFinancialAgent(
        quotes=EGXQuoteService(cache_path=tmp_path / "quotes.json", fetcher=_live_quote),
        portfolios=TelegramPortfolioStore(tmp_path / "portfolios.json"),
    )

    snapshot = agent.portfolio_snapshot("100")
    assert snapshot["insights"] == {
        "position_count": 0,
        "priced_position_count": 0,
        "unavailable_position_count": 0,
        "stale_position_count": 0,
        "quote_coverage_pct": 100.0,
        "largest_position": None,
        "risk_flags": [],
    }
