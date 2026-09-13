"""Backtest the win%/total-runs and strikeout models against a season's completed games.

Walks completed regular-season games chronologically, using only stats accumulated
*before* each game (no lookahead), runs them through src.team_model.game_projection --
the exact function the app calls -- and compares to actual results. When the prior
season is cached too, its final team and pitcher totals seed the priors, the same way
the app uses last season's stats.

Only regular-season games count (gameType=R). The date-range schedule pull also returns
spring training, exhibition, and All-Star games (130-260 a season), which used to leak
into both the running team stats and the scored sample.

Caches raw API pulls to data/raw/ so repeated runs while tuning constants don't re-hit
the MLB Stats API.

Usage:
    python scripts/backtest.py [--season YYYY] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.mlb_stats import (
    BoxscorePitcherLine, BoxscoreResult, BoxscoreTeamLine,
    StatsFeedError, fetch_boxscore, fetch_regular_season_game_pks, fetch_schedule,
)
from src.models import project_strikeouts
from src.team_model import StarterLine, TeamLine, game_projection, league_runs_per_game, park_hr_factor

RAW_DIR = ROOT / "data" / "raw"
MIN_TEAM_GAMES = 3
# Franchise renames, so last season's totals still seed this season's prior.
TEAM_ALIASES = {"Cleveland Indians": "Cleveland Guardians", "Oakland Athletics": "Athletics"}


def _cache_path(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / name


def load_schedule(start: date, end: date, season: int) -> list[dict]:
    cache = _cache_path(f"schedule_{season}.json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("start") == start.isoformat() and cached.get("end") == end.isoformat():
            return cached["games"]

    games: list[dict] = []
    day = start
    while day <= end:
        try:
            for g in fetch_schedule(day):
                if g.detailed_state == "Final":
                    games.append(asdict(g))
        except StatsFeedError as error:
            print(f"  schedule fetch failed for {day}: {error}", file=sys.stderr)
        day += timedelta(days=1)

    cache.write_text(json.dumps({"start": start.isoformat(), "end": end.isoformat(), "games": games}))
    return games


def load_regular_season_pks(season: int) -> set[int]:
    """Cached for finished seasons; re-pulled for the current one (a single request)."""
    cache = _cache_path(f"regular_pks_{season}.json")
    if cache.exists() and season < date.today().year:
        return set(json.loads(cache.read_text()))
    pks = fetch_regular_season_game_pks(season)
    cache.write_text(json.dumps(sorted(pks)))
    return pks


def regular_season_games(games: list[dict], season: int) -> list[dict]:
    """Drop spring training/exhibition/All-Star games and duplicate rows; sort by date."""
    regular = load_regular_season_pks(season)
    seen: set[int] = set()
    kept = []
    for g in sorted(games, key=lambda g: g["game_date"]):
        if g["game_pk"] in regular and g["game_pk"] not in seen:
            seen.add(g["game_pk"])
            kept.append(g)
    return kept


def _boxscore_to_dict(box: BoxscoreResult) -> dict:
    def team(t):
        return {
            "team_name": t.team_name, "plate_appearances": t.plate_appearances,
            "strikeouts": t.strikeouts,
            "pitchers": [asdict(p) for p in t.pitchers],
        }
    return {"game_pk": box.game_pk, "home": team(box.home), "away": team(box.away)}


def _boxscore_from_dict(data: dict) -> BoxscoreResult:
    def team(t):
        return BoxscoreTeamLine(
            team_name=t["team_name"], plate_appearances=t["plate_appearances"],
            strikeouts=t["strikeouts"],
            pitchers=[BoxscorePitcherLine(**p) for p in t["pitchers"]],
        )
    return BoxscoreResult(game_pk=data["game_pk"], home=team(data["home"]), away=team(data["away"]))


def load_boxscores(game_pks: list[int], season: int) -> dict[int, BoxscoreResult]:
    cache_dir = _cache_path(f"boxscores_{season}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, BoxscoreResult] = {}
    to_fetch = []
    for pk in game_pks:
        f = cache_dir / f"{pk}.json"
        if f.exists():
            results[pk] = _boxscore_from_dict(json.loads(f.read_text()))
        else:
            to_fetch.append(pk)

    if to_fetch:
        print(f"  fetching {len(to_fetch)} boxscores (uncached)...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_boxscore, pk): pk for pk in to_fetch}
            done = 0
            for future in as_completed(futures):
                pk = futures[future]
                try:
                    box = future.result()
                except StatsFeedError:
                    continue
                results[pk] = box
                (cache_dir / f"{pk}.json").write_text(json.dumps(_boxscore_to_dict(box)))
                done += 1
                if done % 200 == 0:
                    print(f"    {done}/{len(to_fetch)}")
    return results


def cached_season(season: int) -> tuple[list[dict], dict[int, BoxscoreResult]] | None:
    """A season's regular-season games + boxscores from the data/raw cache, or None if
    that season's schedule was never pulled."""
    cache = RAW_DIR / f"schedule_{season}.json"
    if not cache.exists():
        return None
    games = regular_season_games(json.loads(cache.read_text())["games"], season)
    return games, load_boxscores([g["game_pk"] for g in games], season)


@dataclass
class SeasonTotals:
    """A finished season's per-game team rates and pitcher totals -- next season's priors."""
    team_rates: dict[str, tuple[float, float]]
    pitcher_totals: dict[int, tuple[float, float]]
    league_rpg: float


@dataclass
class GameInputs:
    game: dict
    home: TeamLine
    away: TeamLine
    league_rpg: float
    prev_league_rpg: float | None


def pregame_inputs(
    games: list[dict], boxscores: dict[int, BoxscoreResult], prior: SeasonTotals | None,
) -> tuple[list[GameInputs], SeasonTotals]:
    """Model inputs for every game built only from the games before it, plus the season's
    final totals (to seed the next season)."""
    team_rs: dict[str, float] = {}
    team_ra: dict[str, float] = {}
    team_gp: dict[str, int] = {}
    pitcher_runs: dict[int, float] = {}
    pitcher_outs: dict[int, float] = {}
    league_runs, league_team_games = 0.0, 0
    prev_league_rpg = prior.league_rpg if prior else None

    def team_line(team: str, team_box: BoxscoreTeamLine | None) -> TeamLine:
        starter = next((p for p in team_box.pitchers if p.is_starter), None) if team_box else None
        prev_rs, prev_ra = prior.team_rates.get(team, (None, None)) if prior else (None, None)
        starter_line = None
        if starter is not None:
            prev_runs, prev_outs = prior.pitcher_totals.get(starter.player_id, (0.0, 0.0)) if prior else (0.0, 0.0)
            starter_line = StarterLine(
                runs=pitcher_runs.get(starter.player_id, 0.0), outs=pitcher_outs.get(starter.player_id, 0.0),
                prev_runs=prev_runs, prev_outs=prev_outs,
            )
        return TeamLine(
            runs_scored=team_rs.get(team, 0.0), runs_allowed=team_ra.get(team, 0.0), games=team_gp.get(team, 0),
            prev_rs_pg=prev_rs, prev_ra_pg=prev_ra, starter=starter_line,
        )

    inputs = []
    for g in games:
        box = boxscores.get(g["game_pk"])
        home = TEAM_ALIASES.get(g["home_team"], g["home_team"])
        away = TEAM_ALIASES.get(g["away_team"], g["away_team"])
        inputs.append(GameInputs(
            game=g,
            home=team_line(home, box.home if box else None),
            away=team_line(away, box.away if box else None),
            league_rpg=league_runs_per_game(league_runs, league_team_games, prev_league_rpg),
            prev_league_rpg=prev_league_rpg,
        ))

        for team, scored, allowed in ((home, g["home_score"], g["away_score"]), (away, g["away_score"], g["home_score"])):
            team_rs[team] = team_rs.get(team, 0) + scored
            team_ra[team] = team_ra.get(team, 0) + allowed
            team_gp[team] = team_gp.get(team, 0) + 1
        league_runs += g["home_score"] + g["away_score"]
        league_team_games += 2
        if box is not None:
            for p in box.home.pitchers + box.away.pitchers:
                pitcher_runs[p.player_id] = pitcher_runs.get(p.player_id, 0) + p.runs_allowed
                pitcher_outs[p.player_id] = pitcher_outs.get(p.player_id, 0) + p.outs

    totals = SeasonTotals(
        team_rates={t: (team_rs[t] / team_gp[t], team_ra[t] / team_gp[t]) for t in team_gp},
        pitcher_totals={pid: (pitcher_runs[pid], pitcher_outs[pid]) for pid in pitcher_outs},
        league_rpg=league_runs / league_team_games if league_team_games else 0.0,
    )
    return inputs, totals


def strikeout_rows(games: list[dict], boxscores: dict[int, BoxscoreResult]) -> list[dict]:
    """Starter strikeout projections from pre-game K rates only, vs. actual Ks."""
    team_bat_pa: dict[str, float] = {}
    team_bat_so: dict[str, float] = {}
    pitcher_bf: dict[int, float] = {}
    pitcher_so: dict[int, float] = {}
    rows = []
    for g in games:
        box = boxscores.get(g["game_pk"])
        if box is None:
            continue
        home, away = g["home_team"], g["away_team"]
        for team_name, opp_name, team_line in ((home, away, box.home), (away, home, box.away)):
            starter = next((p for p in team_line.pitchers if p.is_starter), None)
            if starter is None:
                continue
            pre_bf, pre_so = pitcher_bf.get(starter.player_id, 0), pitcher_so.get(starter.player_id, 0)
            opp_pa, opp_so = team_bat_pa.get(opp_name, 0), team_bat_so.get(opp_name, 0)
            if pre_bf >= 40 and opp_pa >= 100:
                proj = project_strikeouts(starter.batters_faced, pre_so / pre_bf, opp_so / opp_pa)
                rows.append({
                    "game_pk": g["game_pk"], "date": g["game_date"], "pitcher": starter.full_name,
                    "team": team_name, "opponent": opp_name,
                    "expected_ks": proj.expected_ks, "actual_ks": starter.strikeouts,
                    "k_error": proj.expected_ks - starter.strikeouts,
                })

        for team_name, team_line in ((home, box.home), (away, box.away)):
            for p in team_line.pitchers:
                pitcher_bf[p.player_id] = pitcher_bf.get(p.player_id, 0) + p.batters_faced
                pitcher_so[p.player_id] = pitcher_so.get(p.player_id, 0) + p.strikeouts
            team_bat_pa[team_name] = team_bat_pa.get(team_name, 0) + team_line.plate_appearances
            team_bat_so[team_name] = team_bat_so.get(team_name, 0) + team_line.strikeouts
    return rows


def run(start: date, end: date, season: int) -> None:
    print(f"Loading schedule {start} .. {end}...")
    games = regular_season_games(load_schedule(start, end, season), season)
    print(f"  {len(games)} completed regular-season games")

    print("Loading boxscores...")
    boxscores = load_boxscores([g["game_pk"] for g in games], season)

    prior_season = cached_season(season - 1)
    prior = pregame_inputs(*prior_season, None)[1] if prior_season else None
    if prior is None:
        print(f"  no cached {season - 1} season -- priors fall back to league average "
              f"(run with --season {season - 1} first to use them)")

    park_factors = pd.read_csv(ROOT / "data" / "ballparks.csv")
    inputs, _ = pregame_inputs(games, boxscores, prior)

    win_rows = []
    for inp in inputs:
        if inp.home.games < MIN_TEAM_GAMES or inp.away.games < MIN_TEAM_GAMES:
            continue
        g = inp.game
        hr_factor = (park_hr_factor(park_factors, g["home_team"]) + 1.0) / 2
        proj = game_projection(inp.home, inp.away, inp.league_rpg, inp.prev_league_rpg, hr_factor)
        actual_total = g["home_score"] + g["away_score"]
        win_rows.append({
            "game_pk": g["game_pk"], "date": g["game_date"], "home_team": g["home_team"], "away_team": g["away_team"],
            "home_wp": proj.home_win_probability, "home_won": 1 if g["home_score"] > g["away_score"] else 0,
            "projected_total": proj.projected_total_runs, "actual_total": actual_total,
            "total_error": proj.projected_total_runs - actual_total,
            "starters_known": inp.home.starter is not None and inp.away.starter is not None,
        })

    df = pd.DataFrame(win_rows)
    print(f"\n  {len(df)} games with sufficient pre-game history "
          f"({df['starters_known'].sum()} with both starters known)")

    p = df["home_wp"].clip(1e-6, 1 - 1e-6)
    brier = ((df["home_wp"] - df["home_won"]) ** 2).mean()
    log_loss = -(df["home_won"] * np.log(p) + (1 - df["home_won"]) * np.log(1 - p)).mean()
    print("\n=== WIN PROBABILITY (src.team_model.game_projection) ===")
    print(f"Brier: {brier:.4f}  |  log loss: {log_loss:.4f}  |  "
          f"mean predicted {df['home_wp'].mean():.1%} vs actual {df['home_won'].mean():.1%}")
    flat_p = df["home_won"].mean()
    flat_brier = ((flat_p - df["home_won"]) ** 2).mean()
    print(f"(flat baseline: always predict {flat_p:.1%} home win -> Brier {flat_brier:.4f})")

    print("\n=== TOTAL RUNS ===")
    print(f"MAE: {df['total_error'].abs().mean():.2f}  |  bias: {df['total_error'].mean():+.2f}")

    # Calibration by confidence: does the model's pick hit at (roughly) the rate it
    # claims, and does that hold up as claimed edge grows?
    print("\n=== CALIBRATION BY CONFIDENCE ===")
    favorite_is_home = df["home_wp"] >= 0.5
    pick_prob = df["home_wp"].where(favorite_is_home, 1 - df["home_wp"])
    pick_correct = (favorite_is_home == (df["home_won"] == 1)).astype(int)
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
    bucket = pd.cut(pick_prob, bins=bins, labels=labels, right=False)
    for b in labels:
        sub_correct, sub_prob = pick_correct[bucket == b], pick_prob[bucket == b]
        if len(sub_correct) == 0:
            continue
        print(f"  {b:>8}: n={len(sub_correct):4d}  predicted {sub_prob.mean():.1%}  "
              f"actual hit rate {sub_correct.mean():.1%}")

    df.to_csv(RAW_DIR / f"backtest_results_{season}.csv", index=False)
    print(f"\nWrote {RAW_DIR / f'backtest_results_{season}.csv'}")

    kdf = pd.DataFrame(strikeout_rows(games, boxscores))
    print(f"\n=== STARTER STRIKEOUTS ===")
    print(f"{len(kdf)} starts with sufficient pre-game history")
    if len(kdf):
        print(f"MAE: {kdf['k_error'].abs().mean():.2f} Ks  |  bias: {kdf['k_error'].mean():+.2f} Ks")
        kdf.to_csv(RAW_DIR / f"backtest_strikeouts_{season}.csv", index=False)
        print(f"Wrote {RAW_DIR / f'backtest_strikeouts_{season}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    default_start = date(args.season, 3, 15)
    default_end = (
        date.today() - timedelta(days=1)
        if args.season == date.today().year
        else date(args.season, 10, 1)
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else default_start
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else default_end

    run(start, end, args.season)
