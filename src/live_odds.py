"""Market-data adapters used by the dashboard.

The Odds API is the preferred source because it supplies sportsbook-level prices.
An API key is optional: without one the app displays the ESPN game board, which is
useful for schedule/status but is not a substitute for a licensed odds feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import requests
from src.feed import get_json


@dataclass(frozen=True)
class MarketQuote:
    game: str
    commence_time: str
    bookmaker: str
    market: str
    selection: str
    point: float | None
    american_odds: int
    updated_at: str | None = None


class OddsFeedError(RuntimeError):
    pass


def fetch_the_odds_api(api_key: str, regions: str = "us") -> list[MarketQuote]:
    """Fetch MLB h2h, spread, and totals quotes from The Odds API."""
    data = get_json(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        OddsFeedError,
        params={"apiKey": api_key, "regions": regions, "markets": "h2h,spreads,totals", "oddsFormat": "american"},
    )

    quotes: list[MarketQuote] = []
    for event in data:
        game = f"{event['away_team']} @ {event['home_team']}"
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    quotes.append(MarketQuote(
                        game=game, commence_time=event.get("commence_time", ""),
                        bookmaker=book.get("title", book.get("key", "Unknown")),
                        market=market.get("key", ""), selection=outcome.get("name", ""),
                        point=outcome.get("point"), american_odds=int(price),
                        updated_at=book.get("last_update"),
                    ))
    return quotes


def fetch_espn_scoreboard(target_date: date | None = None) -> list[dict[str, str]]:
    """Fetch MLB schedule/status; ESPN can expose a consensus line on some games."""
    target = target_date or date.today()
    response = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
        params={"dates": target.strftime("%Y%m%d"), "limit": 100}, timeout=12,
    )
    if response.status_code != 200:
        raise OddsFeedError(f"ESPN scoreboard returned {response.status_code}")
    games: list[dict[str, str]] = []
    for event in response.json().get("events", []):
        competition = event.get("competitions", [{}])[0]
        teams = competition.get("competitors", [])
        away = next((x.get("team", {}).get("displayName", "Away") for x in teams if x.get("homeAway") == "away"), "Away")
        home = next((x.get("team", {}).get("displayName", "Home") for x in teams if x.get("homeAway") == "home"), "Home")
        odds = competition.get("odds", [{}])[0]
        games.append({
            "game": f"{away} @ {home}", "time": event.get("date", ""),
            "status": event.get("status", {}).get("type", {}).get("shortDetail", "Scheduled"),
            "details": odds.get("details", "No consensus line available"),
            "over_under": str(odds.get("overUnder", "—")),
            "provider": odds.get("provider", {}).get("name", "ESPN schedule"),
        })
    return games
