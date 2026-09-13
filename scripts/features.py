"""Pre-game component totals for every cached regular-season game, from the full boxscores.

scripts/backtest.py's pregame_inputs() only tracks runs and outs. This walks the same games
in date order (no lookahead) and snapshots, before each game, the season-to-date totals that
component ratings need: each team's batting line (for BaseRuns), what its staff and its
relievers allowed, and the starter's line this season and the two before, including how deep
his starts go. Each season's final totals seed the next season's priors.

Rates and regression happen later, vectorized, in scripts/tune_win_model.py, so prior sizes
can be tuned without re-walking the games.

Needs data/raw/box_full_<season>/ from scripts/fetch_boxscores_full.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from backtest import RAW_DIR, TEAM_ALIASES, regular_season_games

BAT_KEYS = {
    "PA": "plateAppearances", "AB": "atBats", "H": "hits", "2B": "doubles", "3B": "triples",
    "HR": "homeRuns", "BB": "baseOnBalls", "IBB": "intentionalWalks", "HBP": "hitByPitch",
    "K": "strikeOuts", "SF": "sacFlies", "SB": "stolenBases", "CS": "caughtStealing", "R": "runs",
}
PIT_KEYS = {
    "BF": "battersFaced", "outs": "outs", "H": "hits", "HR": "homeRuns", "BB": "baseOnBalls",
    "HBP": "hitByPitch", "K": "strikeOuts", "R": "runs",
}
STARTER_EXTRA = ("GS", "GSouts")


def _tally() -> defaultdict:
    return defaultdict(float)


def _add(tally: defaultdict, line: dict, keys: dict[str, str]) -> None:
    for short, long in keys.items():
        tally[short] += float(line.get(long, 0) or 0)


@dataclass
class SeasonTotals:
    """A finished season's totals, used as the next season's priors."""
    team_games: dict[str, int] = field(default_factory=dict)
    team_bat: dict[str, dict] = field(default_factory=dict)
    team_pit: dict[str, dict] = field(default_factory=dict)
    team_rp: dict[str, dict] = field(default_factory=dict)
    pitchers: dict[int, dict] = field(default_factory=dict)
    league_bat: dict = field(default_factory=dict)
    league_pit: dict = field(default_factory=dict)
    league_games: int = 0


def load_full_season(season: int) -> tuple[list[dict], dict[int, dict]] | None:
    schedule_file = RAW_DIR / f"schedule_{season}.json"
    box_dir = RAW_DIR / f"box_full_{season}"
    if not schedule_file.exists() or not box_dir.exists():
        return None
    games = regular_season_games(json.loads(schedule_file.read_text())["games"], season)
    boxes = {}
    for g in games:
        f = box_dir / f"{g['game_pk']}.json"
        if f.exists():
            boxes[g["game_pk"]] = json.loads(f.read_text())
    return games, boxes


def _starter(side: dict) -> dict | None:
    pitchers = side.get("pitchers", [])
    return next((p for p in pitchers if p.get("gamesStarted")), pitchers[0] if pitchers else None)


def season_rows(
    season: int, games: list[dict], boxes: dict[int, dict],
    prev: SeasonTotals | None, prev2: SeasonTotals | None,
) -> tuple[list[dict], SeasonTotals]:
    team_games: dict[str, int] = defaultdict(int)
    team_bat, team_pit, team_rp = defaultdict(_tally), defaultdict(_tally), defaultdict(_tally)
    pitchers = defaultdict(_tally)
    league_bat, league_pit = _tally(), _tally()
    league_games = 0
    prev = prev or SeasonTotals()
    prev2 = prev2 or SeasonTotals()

    rows = []
    for g in games:
        box = boxes.get(g["game_pk"])
        if box is None:
            continue
        row = {
            "season": season, "game_pk": g["game_pk"], "date": g["game_date"],
            "home_team": g["home_team"], "away_team": g["away_team"],
            "home_score": g["home_score"], "away_score": g["away_score"],
            "home_won": int(g["home_score"] > g["away_score"]),
            "lg_g": league_games, "pv_lg_g": prev.league_games,
        }
        row.update({f"lg_b_{k}": league_bat[k] for k in BAT_KEYS})
        row.update({f"lg_p_{k}": league_pit[k] for k in (*PIT_KEYS, *STARTER_EXTRA)})
        row.update({f"pv_lg_b_{k}": prev.league_bat.get(k, 0.0) for k in BAT_KEYS})
        row.update({f"pv_lg_p_{k}": prev.league_pit.get(k, 0.0) for k in (*PIT_KEYS, *STARTER_EXTRA)})

        sides = (("h", "home", TEAM_ALIASES.get(g["home_team"], g["home_team"])),
                 ("a", "away", TEAM_ALIASES.get(g["away_team"], g["away_team"])))
        for s, side_name, team in sides:
            starter = _starter(box[side_name])
            pid = starter["id"] if starter else None
            row[f"{s}_g"] = team_games[team]
            row[f"{s}_pv_g"] = prev.team_games.get(team, 0)
            row[f"{s}_sp_id"] = pid
            row.update({f"{s}_b_{k}": team_bat[team][k] for k in BAT_KEYS})
            row.update({f"{s}_p_{k}": team_pit[team][k] for k in PIT_KEYS})
            row.update({f"{s}_rp_{k}": team_rp[team][k] for k in PIT_KEYS})
            row.update({f"{s}_pv_b_{k}": prev.team_bat.get(team, {}).get(k, 0.0) for k in BAT_KEYS})
            row.update({f"{s}_pv_p_{k}": prev.team_pit.get(team, {}).get(k, 0.0) for k in PIT_KEYS})
            row.update({f"{s}_pv_rp_{k}": prev.team_rp.get(team, {}).get(k, 0.0) for k in PIT_KEYS})
            for tag, source in (("sp", pitchers), ("sp1", prev.pitchers), ("sp2", prev2.pitchers)):
                line = source.get(pid, {}) if pid is not None else {}
                row.update({f"{s}_{tag}_{k}": line.get(k, 0.0) for k in (*PIT_KEYS, *STARTER_EXTRA)})
        rows.append(row)

        for side_name, team in ((sides[0][1], sides[0][2]), (sides[1][1], sides[1][2])):
            side = box[side_name]
            starter = _starter(side)
            team_games[team] += 1
            _add(team_bat[team], side["batting"], BAT_KEYS)
            _add(team_pit[team], side["pitching"], PIT_KEYS)
            _add(league_bat, side["batting"], BAT_KEYS)
            _add(league_pit, side["pitching"], PIT_KEYS)
            league_games += 1
            for p in side.get("pitchers", []):
                _add(pitchers[p["id"]], p, PIT_KEYS)
                if starter is not None and p["id"] == starter["id"]:
                    pitchers[p["id"]]["GS"] += 1
                    pitchers[p["id"]]["GSouts"] += float(p.get("outs", 0) or 0)
                    league_pit["GS"] += 1
                    league_pit["GSouts"] += float(p.get("outs", 0) or 0)
                else:
                    _add(team_rp[team], p, PIT_KEYS)

    totals = SeasonTotals(
        team_games=dict(team_games), team_bat={t: dict(v) for t, v in team_bat.items()},
        team_pit={t: dict(v) for t, v in team_pit.items()}, team_rp={t: dict(v) for t, v in team_rp.items()},
        pitchers={p: dict(v) for p, v in pitchers.items()},
        league_bat=dict(league_bat), league_pit=dict(league_pit), league_games=league_games,
    )
    return rows, totals


def feature_frame(seasons=range(2020, 2027)) -> pd.DataFrame:
    """Every cached season's pre-game component totals, one row per game, priors chained."""
    rows, prev, prev2 = [], None, None
    for season in seasons:
        loaded = load_full_season(season)
        if loaded is None:
            print(f"  {season}: no full boxscores cached -- run scripts/fetch_boxscores_full.py {season}")
            prev, prev2 = None, prev
            continue
        season_data, totals = season_rows(season, *loaded, prev, prev2)
        rows.extend(season_data)
        prev, prev2 = totals, prev
    return pd.DataFrame(rows)
