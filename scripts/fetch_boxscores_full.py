"""Re-pull full boxscores for the regular seasons already in the data/raw cache.

The original cache (data/raw/boxscores_<season>/) keeps only strikeouts, runs, and outs.
Component pitcher and bullpen ratings need walks, HBP, home runs, and hits too, so this
keeps every pitcher's and batter's full line, the starting lineups, and the game-time
weather note, one file per game under data/raw/box_full_<season>/.

Usage:
    python scripts/fetch_boxscores_full.py 2020 2021 2022 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest import RAW_DIR, regular_season_games

BOX = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
PITCHING_KEYS = (
    "gamesStarted", "battersFaced", "outs", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
    "intentionalWalks", "hitByPitch", "strikeOuts", "runs", "earnedRuns", "numberOfPitches",
    "groundOuts", "airOuts", "inheritedRunners", "inheritedRunnersScored",
)
BATTING_KEYS = (
    "plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
    "intentionalWalks", "hitByPitch", "strikeOuts", "sacFlies", "sacBunts", "runs",
    "stolenBases", "caughtStealing",
)


def trim_team(team: dict) -> dict:
    players = team.get("players", {})

    def line(pid: int, group: str, keys: tuple[str, ...]) -> dict:
        player = players.get(f"ID{pid}", {})
        stats = player.get("stats", {}).get(group, {})
        return {
            "id": pid, "name": player.get("person", {}).get("fullName"),
            "pos": player.get("position", {}).get("abbreviation"),
            **{k: stats.get(k, 0) for k in keys},
        }

    team_stats = team.get("teamStats", {})
    return {
        "team_id": team.get("team", {}).get("id"), "team_name": team.get("team", {}).get("name"),
        "batting": {k: team_stats.get("batting", {}).get(k, 0) for k in BATTING_KEYS},
        "pitching": {k: team_stats.get("pitching", {}).get(k, 0) for k in PITCHING_KEYS},
        "lineup": team.get("battingOrder", []),
        "pitchers": [line(pid, "pitching", PITCHING_KEYS) for pid in team.get("pitchers", [])],
        "batters": [line(pid, "batting", BATTING_KEYS) for pid in team.get("batters", [])],
    }


def fetch(session: requests.Session, pk: int, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            response = session.get(BOX.format(pk=pk), timeout=20)
            if response.status_code == 200:
                data = response.json()
                return {
                    "game_pk": pk,
                    "home": trim_team(data["teams"]["home"]), "away": trim_team(data["teams"]["away"]),
                    "info": [i for i in data.get("info", []) if i.get("label") in ("Weather", "Wind")],
                }
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"boxscore {pk} kept failing")


def main(seasons: list[int]) -> None:
    session = requests.Session()
    for season in seasons:
        schedule = json.loads((RAW_DIR / f"schedule_{season}.json").read_text())["games"]
        out_dir = RAW_DIR / f"box_full_{season}"
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [g["game_pk"] for g in regular_season_games(schedule, season)
                if not (out_dir / f"{g['game_pk']}.json").exists()]
        print(f"{season}: {len(todo)} boxscores to fetch", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch, session, pk): pk for pk in todo}
            for i, future in enumerate(as_completed(futures), 1):
                pk = futures[future]
                try:
                    box = future.result()
                except RuntimeError as error:
                    print(f"  {error}", file=sys.stderr, flush=True)
                    continue
                (out_dir / f"{pk}.json").write_text(json.dumps(box))
                if i % 500 == 0:
                    print(f"  {season}: {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seasons", type=int, nargs="+")
    main(parser.parse_args().seasons)
