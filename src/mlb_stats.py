"""Client for the free, keyless MLB Stats API (statsapi.mlb.com).

Used to auto-fill model inputs (team runs scored/allowed, probable pitchers,
player K/hit/HR rates) instead of requiring the user to guess slider values.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import requests
from src.feed import get_json

BASE = "https://statsapi.mlb.com/api/v1"
CURRENT_SEASON = date.today().year


class StatsFeedError(RuntimeError):
    pass


def _get(path: str, **params: Any) -> dict:
    return get_json(f"{BASE}{path}", StatsFeedError, params=params)


@dataclass(frozen=True)
class TeamInfo:
    team_id: int
    name: str
    abbreviation: str


@dataclass(frozen=True)
class TeamStats:
    runs_scored: float
    runs_allowed: float
    games_played: int
    batting_k_rate: float | None = None
    pitching_hr_rate_allowed: float | None = None
    batting_avg: float | None = None
    ops: float | None = None
    team_era: float | None = None

    @property
    def runs_scored_per_game(self) -> float:
        return self.runs_scored / self.games_played if self.games_played else 0.0

    @property
    def runs_allowed_per_game(self) -> float:
        return self.runs_allowed / self.games_played if self.games_played else 0.0


@dataclass(frozen=True)
class ScheduledGame:
    game_pk: int
    game: str
    game_date: str
    detailed_state: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int


@dataclass(frozen=True)
class LiveGame:
    game_pk: int
    detailed_state: str
    inning: int | None
    inning_half: str | None
    outs: int | None
    balls: int | None
    strikes: int | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    runner_on_first: bool
    runner_on_second: bool
    runner_on_third: bool
    current_batter: str | None
    current_pitcher: str | None


@dataclass(frozen=True)
class ProbablePitcher:
    game: str
    game_date: str
    home_team: str
    away_team: str
    home_pitcher: str | None
    home_pitcher_id: int | None
    away_pitcher: str | None
    away_pitcher_id: int | None
    game_pk: int | None = None


@dataclass(frozen=True)
class PlayerInfo:
    player_id: int
    full_name: str
    position: str


@dataclass(frozen=True)
class PlayerStats:
    player_id: int
    full_name: str
    plate_appearances_or_batters: float
    k_rate: float | None = None
    hit_rate: float | None = None
    hr_rate: float | None = None
    era: float | None = None
    whip: float | None = None
    runs_allowed: float | None = None  # pitching season stats only: all runs, earned or not
    outs: float | None = None  # pitching season stats only


@dataclass(frozen=True)
class TeamStanding:
    team_id: int
    wins: int
    losses: int
    division_rank: int
    games_back: float
    wildcard_games_back: float | None
    streak_type: str | None
    streak_number: int
    last10_wins: int
    last10_losses: int

    @property
    def is_division_leader(self) -> bool:
        return self.division_rank == 1

    @property
    def is_hot(self) -> bool:
        """Flame-worthy: winning at least 4 straight or 7+ of the last 10."""
        hot_streak = self.streak_type == "wins" and self.streak_number >= 4
        hot_last10 = self.last10_wins >= 7
        return hot_streak or hot_last10

    @property
    def playoff_status(self) -> str | None:
        """A short playoff-race read, or None if the team is out of it."""
        if self.is_division_leader:
            return "Leading division"
        if self.wildcard_games_back is not None and self.wildcard_games_back <= 0:
            return "Holding wild card"
        if self.wildcard_games_back is not None and self.wildcard_games_back <= 3:
            return "In the hunt"
        if self.games_back <= 3:
            return "In the hunt"
        return None


def fetch_teams() -> list[TeamInfo]:
    data = _get("/teams", sportId=1, activeStatus="Yes")
    return [
        TeamInfo(team_id=t["id"], name=t["name"], abbreviation=t.get("abbreviation", ""))
        for t in data.get("teams", [])
    ]


def fetch_team_season_stats(team_id: int, season: int = CURRENT_SEASON) -> TeamStats:
    data = _get(f"/teams/{team_id}/stats", stats="season", group="hitting", season=season)
    hitting = _first_split(data)
    runs_scored = float(hitting.get("runs", 0)) if hitting else 0.0
    games_played = int(hitting.get("gamesPlayed", 0)) if hitting else 0
    plate_appearances = float(hitting.get("plateAppearances", 0)) if hitting else 0.0
    strikeouts_batting = float(hitting.get("strikeOuts", 0)) if hitting else 0.0
    batting_k_rate = strikeouts_batting / plate_appearances if plate_appearances else None
    batting_avg = float(hitting.get("avg", 0)) if hitting and hitting.get("avg") else None
    ops = float(hitting.get("ops", 0)) if hitting and hitting.get("ops") else None

    pitching_data = _get(f"/teams/{team_id}/stats", stats="season", group="pitching", season=season)
    pitching = _first_split(pitching_data)
    runs_allowed = float(pitching.get("runs", 0)) if pitching else 0.0
    batters_faced = float(pitching.get("battersFaced", 0)) if pitching else 0.0
    home_runs_allowed = float(pitching.get("homeRuns", 0)) if pitching else 0.0
    hr_rate_allowed = home_runs_allowed / batters_faced if batters_faced else None
    team_era = float(pitching.get("era", 0)) if pitching and pitching.get("era") else None

    return TeamStats(
        runs_scored=runs_scored, runs_allowed=runs_allowed, games_played=games_played,
        batting_k_rate=batting_k_rate, pitching_hr_rate_allowed=hr_rate_allowed,
        batting_avg=batting_avg, ops=ops, team_era=team_era,
    )


def fetch_league_runs(season: int = CURRENT_SEASON) -> tuple[float, int]:
    """Total runs and team-games across all MLB teams for a season -- the league run environment."""
    data = _get("/teams/stats", stats="season", group="hitting", sportIds=1, season=season)
    splits = (data.get("stats") or [{}])[0].get("splits", [])
    runs = sum(float(s.get("stat", {}).get("runs", 0)) for s in splits)
    team_games = sum(int(s.get("stat", {}).get("gamesPlayed", 0)) for s in splits)
    return runs, team_games


def fetch_regular_season_game_pks(season: int) -> set[int]:
    """gamePks of every regular-season game (gameType=R). A plain date-range schedule pull
    also returns spring training, exhibition, and All-Star games."""
    data = _get("/schedule", sportId=1, gameType="R", startDate=f"{season}-03-01", endDate=f"{season}-11-30")
    return {g["gamePk"] for day in data.get("dates", []) for g in day.get("games", [])}


def _first_split(data: dict) -> dict | None:
    stats = data.get("stats", [])
    if not stats:
        return None
    splits = stats[0].get("splits", [])
    if not splits:
        return None
    return splits[0].get("stat", {})


def fetch_schedule(game_date: date) -> list[ScheduledGame]:
    data = _get("/schedule", sportId=1, date=game_date.strftime("%m/%d/%Y"))
    games: list[ScheduledGame] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_name = home.get("team", {}).get("name", "Home")
            away_name = away.get("team", {}).get("name", "Away")
            games.append(ScheduledGame(
                game_pk=g["gamePk"],
                game=f"{away_name} @ {home_name}",
                game_date=g.get("gameDate", ""),
                detailed_state=g.get("status", {}).get("detailedState", "Scheduled"),
                home_team=home_name,
                away_team=away_name,
                home_score=home.get("score", 0),
                away_score=away.get("score", 0),
            ))
    return games


def fetch_live_game(game_pk: int) -> LiveGame:
    data = _get_live(game_pk)
    live = data.get("liveData", {})
    linescore = live.get("linescore", {})
    game_data = data.get("gameData", {})
    teams = game_data.get("teams", {})
    offense = linescore.get("offense", {})
    defense = linescore.get("defense", {})
    line_teams = linescore.get("teams", {})

    return LiveGame(
        game_pk=game_pk,
        detailed_state=game_data.get("status", {}).get("detailedState", "Unknown"),
        inning=linescore.get("currentInning"),
        inning_half=linescore.get("inningHalf"),
        outs=linescore.get("outs"),
        balls=linescore.get("balls"),
        strikes=linescore.get("strikes"),
        home_team=teams.get("home", {}).get("name", "Home"),
        away_team=teams.get("away", {}).get("name", "Away"),
        home_score=line_teams.get("home", {}).get("runs", 0),
        away_score=line_teams.get("away", {}).get("runs", 0),
        runner_on_first=bool(offense.get("first")),
        runner_on_second=bool(offense.get("second")),
        runner_on_third=bool(offense.get("third")),
        current_batter=offense.get("batter", {}).get("fullName"),
        current_pitcher=defense.get("pitcher", {}).get("fullName"),
    )


def _get_live(game_pk: int) -> dict:
    return get_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", StatsFeedError)


def fetch_probable_pitchers(game_date: date) -> list[ProbablePitcher]:
    data = _get(
        "/schedule", sportId=1, date=game_date.strftime("%m/%d/%Y"),
        hydrate="probablePitcher",
    )
    games: list[ProbablePitcher] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_pitcher = home.get("probablePitcher") or {}
            away_pitcher = away.get("probablePitcher") or {}
            home_name = home.get("team", {}).get("name", "Home")
            away_name = away.get("team", {}).get("name", "Away")
            games.append(ProbablePitcher(
                game_pk=g["gamePk"],
                game=f"{away_name} @ {home_name}",
                game_date=g.get("gameDate", ""),
                home_team=home_name,
                away_team=away_name,
                home_pitcher=home_pitcher.get("fullName"),
                home_pitcher_id=home_pitcher.get("id"),
                away_pitcher=away_pitcher.get("fullName"),
                away_pitcher_id=away_pitcher.get("id"),
            ))
    return games


def search_players(name: str, active_only: bool = True) -> list[PlayerInfo]:
    if not name or len(name.strip()) < 2:
        return []
    data = _get("/people/search", names=name)
    players = []
    for p in data.get("people", []):
        if active_only and not p.get("active", True):
            continue
        players.append(PlayerInfo(
            player_id=p["id"],
            full_name=p.get("fullName", "Unknown"),
            position=p.get("primaryPosition", {}).get("abbreviation", ""),
        ))
    return players


@dataclass(frozen=True)
class BoxscorePitcherLine:
    player_id: int
    full_name: str
    batters_faced: float
    strikeouts: float
    runs_allowed: float
    outs: float
    is_starter: bool


@dataclass(frozen=True)
class BoxscoreTeamLine:
    team_name: str
    plate_appearances: float
    strikeouts: float
    pitchers: list[BoxscorePitcherLine]


@dataclass(frozen=True)
class BoxscoreResult:
    game_pk: int
    home: BoxscoreTeamLine
    away: BoxscoreTeamLine


def _team_boxscore_line(team_data: dict) -> BoxscoreTeamLine:
    team_name = team_data.get("team", {}).get("name", "")
    team_batting = team_data.get("teamStats", {}).get("batting", {})
    plate_appearances = float(team_batting.get("plateAppearances", 0))
    strikeouts = float(team_batting.get("strikeOuts", 0))

    pitcher_ids = team_data.get("pitchers", [])
    players = team_data.get("players", {})
    pitchers: list[BoxscorePitcherLine] = []
    for i, pid in enumerate(pitcher_ids):
        player = players.get(f"ID{pid}", {})
        stats = player.get("stats", {}).get("pitching", {})
        batters_faced = float(stats.get("battersFaced", 0))
        if batters_faced <= 0:
            continue
        pitchers.append(BoxscorePitcherLine(
            player_id=pid,
            full_name=player.get("person", {}).get("fullName", "Unknown"),
            batters_faced=batters_faced,
            strikeouts=float(stats.get("strikeOuts", 0)),
            runs_allowed=float(stats.get("runs", 0)),
            outs=float(stats.get("outs", 0)),
            is_starter=(i == 0),
        ))
    return BoxscoreTeamLine(
        team_name=team_name, plate_appearances=plate_appearances,
        strikeouts=strikeouts, pitchers=pitchers,
    )


def fetch_boxscore(game_pk: int) -> BoxscoreResult:
    data = _get(f"/game/{game_pk}/boxscore")
    teams = data.get("teams", {})
    return BoxscoreResult(
        game_pk=game_pk,
        home=_team_boxscore_line(teams.get("home", {})),
        away=_team_boxscore_line(teams.get("away", {})),
    )


def _parse_games_back(value: str | None) -> float:
    if not value or value == "-":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def fetch_standings(season: int = CURRENT_SEASON) -> dict[int, TeamStanding]:
    """Division rank, games back, streak, and last-10 record for every team."""
    data = _get("/standings", leagueId="103,104", season=season, standingsTypes="regularSeason")
    standings: dict[int, TeamStanding] = {}
    for record in data.get("records", []):
        for team_record in record.get("teamRecords", []):
            team_id = team_record.get("team", {}).get("id")
            if team_id is None:
                continue
            streak = team_record.get("streak", {})
            last10 = next(
                (s for s in team_record.get("records", {}).get("splitRecords", []) if s.get("type") == "lastTen"),
                {},
            )
            league_record = team_record.get("leagueRecord", {})
            wc_raw = team_record.get("wildCardGamesBack")
            standings[team_id] = TeamStanding(
                team_id=team_id,
                wins=int(league_record.get("wins", 0)),
                losses=int(league_record.get("losses", 0)),
                division_rank=int(team_record.get("divisionRank", 0) or 0),
                games_back=_parse_games_back(team_record.get("divisionGamesBack")),
                wildcard_games_back=_parse_games_back(wc_raw) if wc_raw is not None else None,
                streak_type=streak.get("streakType"),
                streak_number=int(streak.get("streakNumber", 0) or 0),
                last10_wins=int(last10.get("wins", 0)),
                last10_losses=int(last10.get("losses", 0)),
            )
    return standings


def fetch_player_recent_stats(player_id: int, full_name: str, group: str, days: int = 14) -> PlayerStats:
    """Same shape as fetch_player_season_stats, but over the trailing `days` window
    only — used to spot a player heating up relative to their season line."""
    from datetime import date as _date, timedelta as _timedelta

    start = (_date.today() - _timedelta(days=days)).isoformat()
    end = _date.today().isoformat()
    data = _get(f"/people/{player_id}/stats", stats="byDateRange", group=group, startDate=start, endDate=end, season=CURRENT_SEASON)
    stat = _first_split(data)
    if not stat:
        return PlayerStats(player_id=player_id, full_name=full_name, plate_appearances_or_batters=0.0)

    if group == "pitching":
        batters_faced = float(stat.get("battersFaced", 0))
        strikeouts = float(stat.get("strikeOuts", 0))
        k_rate = strikeouts / batters_faced if batters_faced else None
        era = float(stat.get("era", 0)) if stat.get("era") else None
        whip = float(stat.get("whip", 0)) if stat.get("whip") else None
        return PlayerStats(
            player_id=player_id, full_name=full_name,
            plate_appearances_or_batters=batters_faced, k_rate=k_rate, era=era, whip=whip,
        )

    plate_appearances = float(stat.get("plateAppearances", 0))
    hits = float(stat.get("hits", 0))
    home_runs = float(stat.get("homeRuns", 0))
    hit_rate = hits / plate_appearances if plate_appearances else None
    hr_rate = home_runs / plate_appearances if plate_appearances else None
    return PlayerStats(
        player_id=player_id, full_name=full_name,
        plate_appearances_or_batters=plate_appearances, hit_rate=hit_rate, hr_rate=hr_rate,
    )


def fetch_player_season_stats(player_id: int, full_name: str, group: str, season: int = CURRENT_SEASON) -> PlayerStats:
    """group is 'pitching' or 'hitting'."""
    data = _get(f"/people/{player_id}/stats", stats="season", group=group, season=season)
    stat = _first_split(data)
    if not stat:
        return PlayerStats(player_id=player_id, full_name=full_name, plate_appearances_or_batters=0.0)

    if group == "pitching":
        batters_faced = float(stat.get("battersFaced", 0))
        strikeouts = float(stat.get("strikeOuts", 0))
        k_rate = strikeouts / batters_faced if batters_faced else None
        era = float(stat.get("era", 0)) if stat.get("era") else None
        whip = float(stat.get("whip", 0)) if stat.get("whip") else None
        return PlayerStats(
            player_id=player_id, full_name=full_name,
            plate_appearances_or_batters=batters_faced, k_rate=k_rate, era=era, whip=whip,
            runs_allowed=float(stat.get("runs", 0)), outs=float(stat.get("outs", 0)),
        )

    plate_appearances = float(stat.get("plateAppearances", 0))
    hits = float(stat.get("hits", 0))
    home_runs = float(stat.get("homeRuns", 0))
    hit_rate = hits / plate_appearances if plate_appearances else None
    hr_rate = home_runs / plate_appearances if plate_appearances else None
    return PlayerStats(
        player_id=player_id, full_name=full_name,
        plate_appearances_or_batters=plate_appearances, hit_rate=hit_rate, hr_rate=hr_rate,
    )
