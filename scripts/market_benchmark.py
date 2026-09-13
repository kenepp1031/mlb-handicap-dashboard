"""Benchmark the win model against the betting market's closing (and opening) moneylines.

A model can beat "always pick the home team" and still be well behind the book. This lines
up every regular-season game 2021-2026 that has both a model prediction (game_projection(),
no lookahead, built exactly as scripts/tune_win_model.py builds them) and an ESPN-archived
moneyline (scripts/fetch_espn_odds.py), then asks four questions:

1. Accuracy: model vs no-vig market log loss and Brier on the same games.
2. Information: fitted on 2021-2024, does a blend of market + model beat the market alone
   on 2025-2026? The model's weight in that blend is what it knows that the market doesn't.
3. Money: flat $1 bets wherever the model disagrees with the no-vig price by a threshold,
   paid at the real (vigged) price.
4. Line movement (2025-2026, where ESPN keeps openers): when the model disagrees with the
   opening line, does the line move its way by the close? Positive closing-line value is
   the most reliable sign of a real edge -- far less noisy than win/loss results.

Usage:
    python scripts/market_benchmark.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest import RAW_DIR, TEAM_ALIASES
from tune_win_model import feature_table
from src.odds import american_to_implied
from src.team_model import game_projection

SEASONS = range(2021, 2027)
TRAIN, TEST = [2021, 2022, 2023, 2024], [2025, 2026]
# The book that stands in for "the market", best first. DraftKings is the book the app
# prices against; ESPN only archived ESPN BET for 2025.
PREFERRED_BOOKS = ("DraftKings", "DraftKings (old)", "ESPN BET", "consensus", "Caesars Sportsbook", "Bet365", "MGM")
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


def team_key(name: str) -> str:
    name = TEAM_ALIASES.get(name, name)
    words = name.split()
    return " ".join(words[-2:]) if words[-1] == "Sox" else words[-1]


def no_vig_home(home_ml: int | None, away_ml: int | None) -> float | None:
    """No-vig home probability, or None for a missing or implausible pair (zeros, live prices)."""
    if not home_ml or not away_ml or abs(home_ml) < 100 or abs(away_ml) < 100:
        return None
    home, away = american_to_implied(home_ml), american_to_implied(away_ml)
    if not 0.99 <= home + away <= 1.12:
        return None
    return home / (home + away)


def book_lines(odds: list[dict]) -> dict[str, dict]:
    """Every pregame book's closing (and, where ESPN has it, opening) moneyline for one game."""
    lines = {}
    for o in odds:
        name = o.get("provider") or ""
        if "Live" in name:
            continue
        home_close = o["home"]["close"] or o["home"]["ml"]
        away_close = o["away"]["close"] or o["away"]["ml"]
        p_close = no_vig_home(home_close, away_close)
        if p_close is None:
            continue
        lines[name] = {
            "home_close": home_close, "away_close": away_close, "p_close": p_close,
            "home_open": o["home"]["open"], "away_open": o["away"]["open"],
            "p_open": no_vig_home(o["home"]["open"], o["away"]["open"]),
        }
    return lines


def load_market(season: int) -> list[dict]:
    games = []
    for f in sorted((RAW_DIR / f"espn_odds_{season}").glob("*.json")):
        for g in json.loads(f.read_text()):
            if not g["completed"]:
                continue
            lines = book_lines(g["odds"])
            book = next((b for b in PREFERRED_BOOKS if b in lines), next(iter(lines), None))
            if book is None:
                continue
            games.append({
                **{k: g[k] for k in ("event_id", "home", "away", "home_score", "away_score")},
                "start": datetime.fromisoformat(g["date"].replace("Z", "+00:00")),
                "book": book, **lines[book],
                "p_consensus": float(np.mean([line["p_close"] for line in lines.values()])),
                "n_books": len(lines),
            })
    return games


def match(games: list[dict], market: list[dict]) -> dict[int, dict]:
    """game_pk -> market row: same home/away teams, start within two days, same final score
    (which also separates doubleheaders), closest start time first."""
    pool = defaultdict(list)
    for m in market:
        pool[(team_key(m["home"]), team_key(m["away"]))].append(m)
    matched, used = {}, set()
    for g in games:
        start = datetime.fromisoformat(g["game_date"].replace("Z", "+00:00"))
        candidates = [
            (abs((m["start"] - start).total_seconds()), m)
            for m in pool[(team_key(g["home_team"]), team_key(g["away_team"]))]
            if m["event_id"] not in used
            and (m["home_score"], m["away_score"]) == (g["home_score"], g["away_score"])
        ]
        candidates = [c for c in candidates if c[0] <= 2 * 86400]
        if candidates:
            m = min(candidates, key=lambda c: c[0])[1]
            used.add(m["event_id"])
            matched[g["game_pk"]] = m
    return matched


def joined_table() -> pd.DataFrame:
    F, inputs = feature_table()
    rows = []
    for season in SEASONS:
        season_inputs = [inp for s, inp in zip(F["season"], inputs) if s == season]
        market = load_market(season)
        matched = match([inp.game for inp in season_inputs], market)
        print(f"  {season}: {len(season_inputs)} model games, {len(market)} priced ESPN games, "
              f"{len(matched)} matched")
        for inp in season_inputs:
            m = matched.get(inp.game["game_pk"])
            if m is None:
                continue
            g = inp.game
            rows.append({
                "season": season, "game_pk": g["game_pk"], "date": g["game_date"][:10],
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_won": int(g["home_score"] > g["away_score"]),
                "p_model": game_projection(inp.home, inp.away, inp.league_rpg, inp.prev_league_rpg).home_win_probability,
                "both_starters": inp.home.starter is not None and inp.away.starter is not None,
                **{k: m[k] for k in ("book", "n_books", "p_close", "p_consensus", "p_open",
                                     "home_close", "away_close", "home_open", "away_open")},
            })
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def per_game_ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton's method; returns coefficients and standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        beta += np.linalg.solve(hessian, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ beta))
    hessian = X.T @ (X * (p * (1 - p))[:, None])
    return beta, np.sqrt(np.diag(np.linalg.inv(hessian)))


def payout(american: float) -> float:
    return american / 100 if american > 0 else 100 / -american


def bets(df: pd.DataFrame, p_col: str, market_col: str, home_price: str, away_price: str, threshold: float) -> pd.DataFrame:
    """One $1 bet per game on whichever side the model rates at least `threshold` above the
    no-vig price, settled at the actual vigged price. Adds each bet's closing-line value:
    how far the no-vig price of the side bet moved toward it by the close."""
    edge = df[p_col] - df[market_col]
    home_bet, away_bet = edge >= threshold, -edge >= threshold
    picked = df[home_bet | away_bet].copy()
    on_home = home_bet[home_bet | away_bet]
    price = np.where(on_home, picked[home_price], picked[away_price]).astype(float)
    won = np.where(on_home, picked["home_won"] == 1, picked["home_won"] == 0)
    picked["profit"] = np.where(won, [payout(x) for x in price], -1.0)
    picked["clv"] = np.where(on_home, picked["p_close"] - picked[market_col], picked[market_col] - picked["p_close"])
    return picked


def describe_bets(label: str, b: pd.DataFrame, with_clv: bool = False) -> None:
    if len(b) < 20:
        print(f"    {label:<14} n={len(b):5d}  (too few bets)")
        return
    roi, se = b["profit"].mean(), b["profit"].std(ddof=1) / np.sqrt(len(b))
    line = (f"    {label:<14} n={len(b):5d}  ROI {roi:+6.1%}  (95% CI {roi - 1.96 * se:+6.1%} to "
            f"{roi + 1.96 * se:+6.1%})")
    if with_clv:
        clv, clv_se = b["clv"].mean(), b["clv"].std(ddof=1) / np.sqrt(len(b))
        line += f"  |  CLV {clv * 100:+.2f} pts (CI {(clv - 1.96 * clv_se) * 100:+.2f} to {(clv + 1.96 * clv_se) * 100:+.2f})"
    print(line)


def main() -> None:
    print("Joining model predictions to ESPN moneylines...")
    df = joined_table()
    df.to_csv(RAW_DIR / "market_joined.csv", index=False)
    y = df["home_won"].values.astype(float)
    train, test = df["season"].isin(TRAIN).values, df["season"].isin(TEST).values
    print("\nBooks used as 'the market':")
    print(df.groupby("season")["book"].value_counts().to_string())

    print("\n=== 1. ACCURACY ON THE SAME GAMES (lower is better) ===")
    print("  model-minus-market: negative would mean the model beat the closing line")
    for label, mask in [(str(s), (df["season"] == s).values) for s in SEASONS] + [("TRAIN 21-24", train), ("TEST 25-26", test)]:
        if not mask.any():
            continue
        d = per_game_ll(df["p_model"][mask], y[mask]) - per_game_ll(df["p_close"][mask], y[mask])
        print(f"  {label:<12} n={mask.sum():5d} | log loss: model {log_loss(df['p_model'][mask], y[mask]):.4f}  "
              f"market {log_loss(df['p_close'][mask], y[mask]):.4f}  consensus {log_loss(df['p_consensus'][mask], y[mask]):.4f} | "
              f"model-market {d.mean():+.4f} +/- {1.96 * d.std(ddof=1) / np.sqrt(mask.sum()):.4f} | "
              f"Brier model {np.mean((df['p_model'][mask] - y[mask]) ** 2):.4f} market {np.mean((df['p_close'][mask] - y[mask]) ** 2):.4f}")
    has_open = df["p_open"].notna().values
    if has_open.any():
        print(f"  openers ({has_open.sum()} games): open {log_loss(df['p_open'][has_open], y[has_open]):.4f}  "
              f"close {log_loss(df['p_close'][has_open], y[has_open]):.4f}  model {log_loss(df['p_model'][has_open], y[has_open]):.4f}")

    print("\n=== 2. DOES THE MODEL KNOW ANYTHING THE MARKET DOESN'T? ===")
    X = np.column_stack([np.ones(len(df)), logit(df["p_close"]), logit(df["p_model"])])
    beta, se = fit_logistic(X[train], y[train])
    print(f"  fit on 2021-24: logit(win) = {beta[0]:+.3f} + {beta[1]:.3f} * market + {beta[2]:.3f} * model")
    print(f"  model coefficient {beta[2]:.3f} +/- {1.96 * se[2]:.3f} (95%)  -> "
          f"{'adds information' if beta[2] - 1.96 * se[2] > 0 else 'no reliable information beyond the market'}")
    df["p_blend"] = 1 / (1 + np.exp(-(X @ beta)))
    print(f"  held-out 2025-26 log loss: market {log_loss(df['p_close'][test], y[test]):.4f}  "
          f"blend {log_loss(df['p_blend'][test], y[test]):.4f}  model {log_loss(df['p_model'][test], y[test]):.4f}")
    beta_test, se_test = fit_logistic(X[test], y[test])
    print(f"  (same fit on 2025-26 alone: model coefficient {beta_test[2]:.3f} +/- {1.96 * se_test[2]:.3f})")

    print("\n=== 3. BETTING THE MODEL'S DISAGREEMENTS AT THE CLOSING PRICE ===")
    print(f"  baseline, every home team: ", end="")
    home_all = df.assign(p_all=1.0)
    describe_bets("", bets(home_all, "p_all", "p_close", "home_close", "away_close", 0.0))
    for name, p_col, mask in (("model, 2021-24", "p_model", train), ("model, 2025-26", "p_model", test),
                              ("blend, 2025-26", "p_blend", test)):
        print(f"  {name}:")
        for t in THRESHOLDS:
            describe_bets(f"edge >= {t:.0%}", bets(df[mask], p_col, "p_close", "home_close", "away_close", t))

    opened = df[df["p_open"].notna() & df["home_open"].notna() & df["away_open"].notna()]
    if len(opened):
        print(f"\n=== 4. BETTING AT THE OPENER, AND WHERE THE LINE WENT ({len(opened)} games with openers) ===")
        print("  CLV = how many no-vig points the side we bet gained from open to close")
        print("  Closing-market blend excluded: its inputs were unavailable at the opener.")
        print("  Model results below are retrospective: model inputs are pregame, not timestamped at the opener.")
        for name, p_col in (("model", "p_model"),):
            print(f"  {name}:")
            for t in THRESHOLDS:
                describe_bets(f"edge >= {t:.0%}", bets(opened, p_col, "p_open", "home_open", "away_open", t), with_clv=True)

    print(f"\nWrote {RAW_DIR / 'market_joined.csv'}")


if __name__ == "__main__":
    main()
