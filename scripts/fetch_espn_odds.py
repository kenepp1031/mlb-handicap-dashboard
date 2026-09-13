"""Pull historical MLB moneylines from ESPN's public API for the market benchmark.

ESPN is already one of the project's sources (config/sources.json). Its core API keeps
the odds for past games: several books for 2021-2024 (DraftKings among them), ESPN BET for
2025, and DraftKings with opening and closing lines for 2026. One scoreboard call per date
lists the games, then one odds call per game returns every book's line.

Cached one file per date under data/raw/espn_odds_<season>/, so an interrupted run picks up
where it stopped. Dates are the US calendar dates ESPN files games under.

Usage:
    python scripts/fetch_espn_odds.py 2021 2022 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ODDS = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{id}/competitions/{id}/odds"


def get_json(session: requests.Session, url: str, params: dict | None = None, tries: int = 5) -> dict | None:
    for attempt in range(tries):
        try:
            response = session.get(url, params=params, timeout=20)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ESPN request kept failing: {url}")


def american(value) -> int | None:
    """ESPN writes American odds as an int, '+135', '-163', or 'EVEN'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper()
    if text in ("EVEN", "EV"):
        return 100
    try:
        return int(float(text.replace("+", "")))
    except ValueError:
        return None


def side_lines(team_odds: dict) -> dict:
    """One side's moneyline: ESPN's headline number plus open/close/current where it has them."""
    def block(name):
        return american(((team_odds.get(name) or {}).get("moneyLine") or {}).get("american"))
    return {
        "ml": american(team_odds.get("moneyLine")),
        "open": block("open"), "close": block("close"), "current": block("current"),
    }


def event_odds(session: requests.Session, event_id: str) -> list[dict]:
    data = get_json(session, ODDS.format(id=event_id), {"limit": 100}) or {}
    return [
        {
            "provider": item.get("provider", {}).get("name"),
            "priority": item.get("provider", {}).get("priority"),
            "home": side_lines(item.get("homeTeamOdds") or {}),
            "away": side_lines(item.get("awayTeamOdds") or {}),
            "total": item.get("overUnder"),
        }
        for item in data.get("items", [])
    ]


def fetch_day(day: date) -> list[dict]:
    session = requests.Session()
    board = get_json(session, SCOREBOARD, {"dates": day.strftime("%Y%m%d"), "limit": 100}) or {}
    games = []
    for event in board.get("events", []):
        comp = event["competitions"][0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        games.append({
            "event_id": event["id"], "date": event.get("date"),
            "season_type": event.get("season", {}).get("type"),
            "completed": comp.get("status", {}).get("type", {}).get("completed", False),
            "home": sides["home"]["team"].get("displayName"), "away": sides["away"]["team"].get("displayName"),
            "home_score": int(sides["home"].get("score") or 0), "away_score": int(sides["away"].get("score") or 0),
            "odds": event_odds(session, event["id"]),
        })
    return games


def season_dates(season: int) -> list[date]:
    start, end = date(season, 3, 15), min(date(season, 10, 5), date.today() - timedelta(days=1))
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def main(seasons: list[int]) -> None:
    for season in seasons:
        out_dir = RAW_DIR / f"espn_odds_{season}"
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [d for d in season_dates(season) if not (out_dir / f"{d.isoformat()}.json").exists()]
        print(f"{season}: {len(todo)} dates to fetch", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_day, d): d for d in todo}
            for i, future in enumerate(as_completed(futures), 1):
                d = futures[future]
                try:
                    games = future.result()
                except RuntimeError as error:
                    print(f"  {d}: {error}", file=sys.stderr, flush=True)
                    continue
                (out_dir / f"{d.isoformat()}.json").write_text(json.dumps(games))
                if i % 25 == 0:
                    print(f"  {season}: {i}/{len(todo)} dates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seasons", type=int, nargs="+")
    main(parser.parse_args().seasons)
