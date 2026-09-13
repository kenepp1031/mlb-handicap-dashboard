"""Fit the win-probability constants in src/team_model.py against the cached seasons.

Walks every cached regular season 2020-2026 with scripts/backtest.py's no-lookahead
pre-game inputs (each season's final totals seed the next season's priors), grid-searches
the regression/starter constants on 2021-2024, fits the Platt calibration on those same
seasons, and scores the result on 2025-2026 -- seasons it never saw. Then it re-scores the
constants actually in src/team_model.py through game_projection() itself, so a typo there
or drift between this grid and the real function shows up here.

Uses the data/raw cache; run scripts/backtest.py --season <year> for any season not yet
cached.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest import MIN_TEAM_GAMES, cached_season, pregame_inputs
from src import team_model
from src.team_model import game_projection

SEASONS = range(2020, 2027)
TRAIN, TEST = [2021, 2022, 2023, 2024], [2025, 2026]
GRID = {
    "k": [70, 100, 150, 200, 300],            # TEAM_PRIOR_GAMES
    "w": [0, 0.25, 0.5, 0.75, 1.0],           # PRIOR_SEASON_WEIGHT
    "share": [0.3, 0.45, 0.6, 0.75],          # STARTER_SHARE
    "m": [300, 450, 700, 1000, 1500],         # STARTER_PRIOR_OUTS
    "dec": [0.25, 0.5, 0.75, 1.0],            # STARTER_PRIOR_SEASON_WEIGHT
}


def feature_table():
    """One row of raw pre-game inputs per game (both teams past MIN_TEAM_GAMES), plus the
    matching GameInputs so the shipped model can be scored on exactly the same games."""
    rows, kept, prior = [], [], None
    for season in SEASONS:
        cached = cached_season(season)
        if cached is None:
            print(f"  {season} not cached -- skipping (run scripts/backtest.py --season {season})")
            prior = None
            continue
        inputs, prior_next = pregame_inputs(*cached, prior)
        for inp in inputs:
            if inp.home.games < MIN_TEAM_GAMES or inp.away.games < MIN_TEAM_GAMES:
                continue
            row = {
                "season": season, "home_won": int(inp.game["home_score"] > inp.game["away_score"]),
                "lg": inp.league_rpg, "prev_lg": inp.prev_league_rpg or team_model.DEFAULT_LEAGUE_RPG,
            }
            for s, t in (("h", inp.home), ("a", inp.away)):
                starter = t.starter or team_model.StarterLine()
                row.update({
                    f"{s}_rs": t.runs_scored, f"{s}_ra": t.runs_allowed, f"{s}_g": t.games,
                    f"{s}_prs": np.nan if t.prev_rs_pg is None else t.prev_rs_pg,
                    f"{s}_pra": np.nan if t.prev_ra_pg is None else t.prev_ra_pg,
                    f"{s}p_has": t.starter is not None,
                    f"{s}p_r": starter.runs, f"{s}p_o": starter.outs,
                    f"{s}p_pr": starter.prev_runs, f"{s}p_po": starter.prev_outs,
                })
            rows.append(row)
            kept.append(inp)
        prior = prior_next
    return pd.DataFrame(rows), kept


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def model_logit(F: pd.DataFrame, k, w, share, m, dec) -> np.ndarray:
    """Vectorized game_projection() up to (not including) calibration, for one set of constants."""
    lg, prev_lg, e = F["lg"].values, F["prev_lg"].values, team_model.PYTHAG_EXPONENT

    def regressed(total, games, prev):
        prior = np.where(np.isnan(prev), lg, lg + w * (prev - prev_lg))
        return (total + k * prior) / (games + k)

    pyth = {}
    for s in ("h", "a"):
        rs = regressed(F[f"{s}_rs"].values, F[f"{s}_g"].values, F[f"{s}_prs"].values)
        ra = regressed(F[f"{s}_ra"].values, F[f"{s}_g"].values, F[f"{s}_pra"].values)
        runs = F[f"{s}p_r"].values + dec * F[f"{s}p_pr"].values
        outs = F[f"{s}p_o"].values + dec * F[f"{s}p_po"].values
        ra9 = (runs + m / 27 * ra) / ((outs + m) / 27)
        ra = np.where(F[f"{s}p_has"].values, share * ra9 + (1 - share) * ra, ra)
        pyth[s] = rs**e / (rs**e + ra**e)
    h, a = pyth["h"], pyth["a"]
    return logit(h * (1 - a) / (h * (1 - a) + a * (1 - h)))


def fit_platt(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic regression of y on [1, x] by Newton's method."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        beta += np.linalg.solve(hessian, X.T @ (y - p))
    return float(beta[0]), float(beta[1])


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": np.mean((p - y) ** 2),
        "logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "sd": p.std(),
    }


def main():
    F, inputs = feature_table()
    train, test = F["season"].isin(TRAIN).values, F["season"].isin(TEST).values
    y = F["home_won"].values.astype(float)
    print(f"{train.sum()} train games ({TRAIN[0]}-{TRAIN[-1]}), {test.sum()} test games "
          f"({TEST[0]}-{TEST[-1]}), regular season only\n")

    results = []
    for k, w, share, m, dec in itertools.product(*GRID.values()):
        x = model_logit(F, k, w, share, m, dec)
        a, b = fit_platt(x[train], y[train])
        p = 1 / (1 + np.exp(-(a + b * x)))
        results.append({"k": k, "w": w, "share": share, "m": m, "dec": dec, "a": a, "b": b,
                        "train": scores(p[train], y[train]), "test": scores(p[test], y[test])})
    results.sort(key=lambda r: r["train"]["logloss"])

    print("Top 8 by train log loss (test = held-out seasons):")
    for r in results[:8]:
        print(f"  k={r['k']} w={r['w']} share={r['share']} m={r['m']} dec={r['dec']} "
              f"a={r['a']:.3f} b={r['b']:.3f} | train LL {r['train']['logloss']:.4f} | "
              f"test Brier {r['test']['brier']:.4f} LL {r['test']['logloss']:.4f}")

    shipped = np.array([
        game_projection(i.home, i.away, i.league_rpg, i.prev_league_rpg).home_win_probability for i in inputs
    ])
    flat = np.full_like(y, y[train].mean())
    early = (np.minimum(F["h_g"], F["a_g"]) < 30).values
    print("\nShipped constants (src/team_model.py) vs. always predicting the train home-win rate:")
    segments = [(f"{s} {'test' if s in TEST else 'train'}", (F["season"] == s).values) for s in TRAIN + TEST]
    segments += [("ALL TEST", test), ("TEST first 30 G", test & early)]
    for label, mask in segments:
        sm, fm = scores(shipped[mask], y[mask]), scores(flat[mask], y[mask])
        print(f"  {label:<16} n={mask.sum():5d} | model Brier {sm['brier']:.4f} LL {sm['logloss']:.4f} "
              f"sd {sm['sd']:.3f} | flat Brier {fm['brier']:.4f} LL {fm['logloss']:.4f}")

    print("\nHeld-out calibration by pick confidence (shipped constants):")
    p, yy = shipped[test], y[test]
    pick = np.where(p >= 0.5, p, 1 - p)
    hit = np.where(p >= 0.5, yy, 1 - yy)
    for lo, hi, label in ((0.5, 0.55, "50-55%"), (0.55, 0.6, "55-60%"), (0.6, 0.65, "60-65%"),
                          (0.65, 0.7, "65-70%"), (0.7, 1.01, "70%+")):
        mask = (pick >= lo) & (pick < hi)
        if mask.any():
            print(f"  {label:>7}: n={mask.sum():4d}  predicted {pick[mask].mean():.1%}  actual {hit[mask].mean():.1%}")


if __name__ == "__main__":
    main()
