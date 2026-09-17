"""DraftKings MLB moneylines from ESPN's public scoreboard: no key, no monthly quota.

One request returns every game on a date with the DraftKings line ESPN displays. ESPN only
carries the line until first pitch, which is all the board compares against anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.feed import get_json

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
# ESPN display name -> MLB Stats API name, where they differ.
TEAM_NAMES = {"Athletics Athletics": "Athletics", "Oakland Athletics": "Athletics"}


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


def _american(value) -> int | None:
    """ESPN writes American odds as '+135', '-163', or 'EVEN'."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ("EVEN", "EV"):
        return 100
    try:
        return int(float(text.replace("+", "")))
    except ValueError:
        return None


def fetch_espn_moneylines(game_date: date) -> list[MarketQuote]:
    """Moneyline quotes for every game ESPN files under game_date (US calendar date)."""
    data = get_json(SCOREBOARD, OddsFeedError, params={"dates": game_date.strftime("%Y%m%d"), "limit": 100})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    quotes: list[MarketQuote] = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c.get("team", {}).get("displayName", "") for c in comp.get("competitors", [])}
        teams = {k: TEAM_NAMES.get(v, v) for k, v in sides.items()}
        if not teams.get("home") or not teams.get("away"):
            continue
        game = f"{teams['away']} @ {teams['home']}"
        for item in comp.get("odds") or []:
            book = (item.get("provider") or {}).get("name", "Unknown")
            moneyline = item.get("moneyline") or {}
            for side in ("home", "away"):
                line = moneyline.get(side) or {}
                price = _american((line.get("close") or line.get("current") or line.get("open") or {}).get("odds"))
                if price is None:
                    continue
                quotes.append(MarketQuote(
                    game=game, commence_time=event.get("date", ""), bookmaker=book, market="h2h",
                    selection=teams[side], point=None, american_odds=price, updated_at=now,
                ))
    return quotes
