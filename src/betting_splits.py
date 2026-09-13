"""Public betting-splits scraper for DraftKings Network's MLB splits page.

DraftKings does not expose bet%/handle% through a public API, so this parses
the server-rendered HTML table DraftKings Network publishes for MLB moneyline
splits. If the page markup changes, parsing fails soft (returns {}) rather
than raising, since this is a "nice to have" overlay, not core model data.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

SPLITS_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class SplitsFeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeamSplit:
    team: str
    odds: int | None
    pct_handle: float | None
    pct_bets: float | None


@dataclass(frozen=True)
class GameSplits:
    matchup: str
    teams: dict[str, TeamSplit]


def _to_int_odds(text: str) -> int | None:
    cleaned = text.strip().replace("−", "-")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _to_pct(text: str) -> float | None:
    cleaned = text.strip().rstrip("%")
    try:
        return float(cleaned) / 100
    except ValueError:
        return None


def fetch_mlb_betting_splits() -> dict[str, GameSplits]:
    """Returns moneyline splits keyed by team name (e.g. 'NY Yankees')."""
    try:
        response = requests.get(
            SPLITS_URL, params={"tb_eg": "MLB", "tb_edate": "n7days", "tb_emt": 0},
            headers=HEADERS, timeout=12,
        )
    except requests.RequestException as error:
        raise SplitsFeedError(str(error)) from error
    if response.status_code != 200:
        raise SplitsFeedError(f"DraftKings Network returned {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    by_team: dict[str, GameSplits] = {}
    for event in soup.select(".tb-se"):
        title = event.select_one("h5")
        if not title:
            continue
        matchup = title.get_text(strip=True)

        moneyline_wrap = None
        for wrap in event.select(".tb-market-wrap > div"):
            head = wrap.select_one(".tb-se-head")
            if head and "Moneyline" in head.get_text():
                moneyline_wrap = wrap
                break
        if moneyline_wrap is None:
            continue

        teams: dict[str, TeamSplit] = {}
        for row in moneyline_wrap.select(".tb-sodd"):
            name_el = row.select_one(".tb-slipline")
            odds_el = row.select_one(".tb-odd-s")
            cells = row.select("div.flex-1")
            if not name_el or len(cells) < 4:
                continue
            team = name_el.get_text(strip=True)
            odds = _to_int_odds(odds_el.get_text()) if odds_el else None
            pct_handle = _to_pct(cells[2].contents[0]) if cells[2].contents else None
            pct_bets = _to_pct(cells[3].contents[0]) if cells[3].contents else None
            split = TeamSplit(team=team, odds=odds, pct_handle=pct_handle, pct_bets=pct_bets)
            teams[team] = split
            by_team[team] = GameSplits(matchup=matchup, teams=teams)

        for team in teams:
            by_team[team] = GameSplits(matchup=matchup, teams=teams)

    return by_team
