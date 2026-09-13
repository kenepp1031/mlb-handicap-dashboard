from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Model constants. Grid-searched by scripts/tune_win_model.py on 2021-2024 regular-season
# games, then scored on 2025-2026 games it never saw -- results in
# calibrate_win_probability(). Re-run that script as seasons complete.
PYTHAG_EXPONENT = 1.83
TEAM_PRIOR_GAMES = 150            # team R/G is regressed as if the prior were this many extra games
PRIOR_SEASON_WEIGHT = 0.25        # share of last season's above/below-average level kept in the prior
LEAGUE_PRIOR_TEAM_GAMES = 200     # early-season league run environment leans on last season's
DEFAULT_LEAGUE_RPG = 4.5          # only used when last season's league rate is unknown
STARTER_SHARE = 0.60              # weight of the starter's own (regressed) RA9 in his team's run prevention
STARTER_PRIOR_OUTS = 1000         # starter RA9 is regressed as if he'd also thrown this many team-average outs
STARTER_PRIOR_SEASON_WEIGHT = 0.5 # last season's pitching line counts half as much as this season's
PLATT_A, PLATT_B = 0.118, 1.349   # calibration: intercept = home field, slope on the neutral log5 logit


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pythagorean_win_pct(runs_scored: float, runs_allowed: float, exponent: float = PYTHAG_EXPONENT) -> float:
    """Bill James Pythagorean expectation; transparent baseline for team strength.
    The exponent barely matters once calibrate_win_probability() rescales the result
    (the tuning grid showed identical fits for 1.6-2.0), so this keeps the textbook 1.83."""
    if runs_scored <= 0 and runs_allowed <= 0:
        return 0.5
    rs, ra = max(runs_scored, 0.1), max(runs_allowed, 0.1)
    return clamp(rs**exponent / (rs**exponent + ra**exponent), 0.02, 0.98)


def log5_win_probability(team_pyth: float, opponent_pyth: float, home_field_edge: float = 0.0) -> float:
    """Bill James log5 method for head-to-head probability, plus an optional home-field bump.
    game_projection() leaves the bump at 0: home field is the fitted intercept in
    calibrate_win_probability() instead of a guessed additive constant."""
    team_pyth = clamp(team_pyth, 0.02, 0.98)
    opponent_pyth = clamp(opponent_pyth, 0.02, 0.98)
    numerator = team_pyth * (1 - opponent_pyth)
    denominator = numerator + opponent_pyth * (1 - team_pyth)
    base = numerator / denominator if denominator else 0.5
    return clamp(base + home_field_edge, 0.01, 0.99)


def league_runs_per_game(season_runs: float, season_team_games: int, prev_league_rpg: float | None = None) -> float:
    """This season's run environment (runs per team-game), leaning on last season's until
    enough games are in that one hot April week can't swing it."""
    prev = prev_league_rpg if prev_league_rpg else DEFAULT_LEAGUE_RPG
    return (season_runs + LEAGUE_PRIOR_TEAM_GAMES * prev) / (season_team_games + LEAGUE_PRIOR_TEAM_GAMES)


def regressed_rate(
    season_total: float,
    games: int,
    league_rpg: float,
    prev_rate: float | None = None,
    prev_league_rpg: float | None = None,
) -> float:
    """A team's runs scored (or allowed) per game, regressed toward a prior as if the prior
    were TEAM_PRIOR_GAMES extra games. The prior is league average, nudged by how far above
    or below average the team was last season.

    Raw season-to-date R/G was the old model's biggest problem: a few weeks in it's mostly
    noise, and even a full season carries a lot of luck (the fitted 150-game prior means 162
    games of R/G only gets about half the say). Taken at face value it made the model so
    overconfident that the old calibration had to halve every prediction's logit, which
    flattened the real gaps between teams along with the noise.
    """
    prior = league_rpg
    if prev_rate is not None and prev_league_rpg:
        prior += PRIOR_SEASON_WEIGHT * (prev_rate - prev_league_rpg)
    return (season_total + TEAM_PRIOR_GAMES * prior) / (games + TEAM_PRIOR_GAMES)


@dataclass(frozen=True)
class StarterLine:
    """A starting pitcher's runs allowed and outs recorded (all appearances), this season and last."""
    runs: float = 0.0
    outs: float = 0.0
    prev_runs: float = 0.0
    prev_outs: float = 0.0


@dataclass(frozen=True)
class TeamLine:
    """Everything game_projection() needs about one side: season-to-date totals, last
    season's per-game rates (None if unknown), and the starter (None if not announced)."""
    runs_scored: float
    runs_allowed: float
    games: int
    prev_rs_pg: float | None = None
    prev_ra_pg: float | None = None
    starter: StarterLine | None = None


def starter_adjusted_runs_allowed(team_ra_per_game: float, starter: StarterLine | None) -> float:
    """Blend the starter's own RA9 into his team's run prevention.

    His RA9 is regressed toward the team's rate as if he'd also thrown STARTER_PRIOR_OUTS
    outs (~333 IP) of team-average ball, with last season's line at half weight. The heavy
    regression is what makes the starter usable: the old version trusted a starter's raw RA9
    once he passed 30 IP, which was noisy enough that it had to be cut to a 25% share to stop
    hurting the backtest. Regressed properly, the starter earns a 60% share -- roughly the
    part of the game he actually pitches.
    """
    if starter is None:
        return team_ra_per_game
    runs = starter.runs + STARTER_PRIOR_SEASON_WEIGHT * starter.prev_runs
    outs = starter.outs + STARTER_PRIOR_SEASON_WEIGHT * starter.prev_outs
    ra9 = (runs + STARTER_PRIOR_OUTS / 27 * team_ra_per_game) / ((outs + STARTER_PRIOR_OUTS) / 27)
    return STARTER_SHARE * ra9 + (1 - STARTER_SHARE) * team_ra_per_game


def project_team_runs(
    team_a_rs_pg: float,
    team_a_ra_pg: float,
    team_b_rs_pg: float,
    team_b_ra_pg: float,
    park_factor: float = 1.0,
) -> tuple[float, float]:
    """Average each side's offense against the other's run prevention, scale by park.
    Returns (team_a_runs, team_b_runs)."""
    team_a_runs = (team_a_rs_pg + team_b_ra_pg) / 2 * park_factor
    team_b_runs = (team_b_rs_pg + team_a_ra_pg) / 2 * park_factor
    return team_a_runs, team_b_runs


def calibrate_win_probability(neutral_probability: float) -> float:
    """Turn the neutral-site log5 probability into a home-win probability.

    Platt scaling (logistic regression of the actual home result on the log5 logit), fitted
    on 2021-2024 regular-season games along with the constants at the top of this file. The
    intercept is home-field advantage (~53% between equal teams); the slope is above 1
    because regressing team ratings toward average is deliberately conservative, and this
    stretches them back to honest confidence.

    Held-out 2025-2026 regular seasons (4,531 games, never used in fitting):
      Brier 0.2442 / log loss 0.6812, vs 0.2476 / 0.6887 for the previous model and
      0.2488 / 0.6908 for always predicting the home-win rate. In the first 30 games of a
      season, 0.2424 vs the previous model's 0.2539, which was worse than that flat guess.
      By pick confidence: 60-65% picks hit 61.9%, 65-70% hit 66.4%, 70%+ hit 78.1% (n=73).
    """
    p = clamp(neutral_probability, 0.02, 0.98)
    z = PLATT_A + PLATT_B * math.log(p / (1 - p))
    return clamp(1 / (1 + math.exp(-z)), 0.01, 0.99)


def park_hr_factor(park_factors: pd.DataFrame, team_name: str) -> float:
    """Average lhb/rhb HR park factor for a team's home park; 1.0 if unknown."""
    row = park_factors[park_factors["team"].str.contains(team_name.split()[-1], case=False, na=False)]
    if row.empty:
        return 1.0
    return float((row.iloc[0]["hr_factor_lhb"] + row.iloc[0]["hr_factor_rhb"]) / 2)


@dataclass
class GameProjection:
    home_win_probability: float
    away_win_probability: float
    projected_total_runs: float
    home_runs: float
    away_runs: float


def game_projection(
    home: TeamLine,
    away: TeamLine,
    league_rpg: float,
    prev_league_rpg: float | None = None,
    park_factor: float = 1.0,
) -> GameProjection:
    """The win-probability and run model. app.py and scripts/backtest.py both call this one
    function, so the backtested model is always exactly the one on screen. They used to
    drift: the app skipped the starter adjustment and used a different home edge than the
    backtest that fit its calibration."""
    def rates(side: TeamLine) -> tuple[float, float]:
        rs = regressed_rate(side.runs_scored, side.games, league_rpg, side.prev_rs_pg, prev_league_rpg)
        ra = regressed_rate(side.runs_allowed, side.games, league_rpg, side.prev_ra_pg, prev_league_rpg)
        return rs, starter_adjusted_runs_allowed(ra, side.starter)

    home_rs, home_ra = rates(home)
    away_rs, away_ra = rates(away)
    neutral = log5_win_probability(pythagorean_win_pct(home_rs, home_ra), pythagorean_win_pct(away_rs, away_ra))
    home_wp = calibrate_win_probability(neutral)
    home_runs, away_runs = project_team_runs(home_rs, home_ra, away_rs, away_ra, park_factor)
    return GameProjection(home_wp, 1 - home_wp, home_runs + away_runs, home_runs, away_runs)
