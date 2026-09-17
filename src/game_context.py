"""Match prices to a specific game and keep pregame picks out of past/live games."""
from datetime import datetime, timezone

import pandas as pd

PREGAME_STATES = ("Scheduled", "Pre-Game", "Warmup")
START_TOLERANCE = pd.Timedelta(minutes=60)


def eligible_pregame(game, now=None) -> bool:
    now = now or datetime.now(timezone.utc)
    try:
        start = datetime.fromisoformat(game.game_date.replace("Z", "+00:00"))
        return game.detailed_state in PREGAME_STATES and start > now
    except (ValueError, TypeError):
        return False


def book_price(quote_df, away, home, market, selection, bookmaker, game_date):
    if quote_df.empty:
        return None
    start = pd.to_datetime(game_date, utc=True, errors="coerce")
    # Feeds can list the same first pitch a few minutes apart; doubleheader games are hours apart.
    gap = (pd.to_datetime(quote_df["commence_time"], utc=True, errors="coerce") - start).abs()
    matches = quote_df[
        quote_df["game"].eq(f"{away} @ {home}")
        & gap.le(START_TOLERANCE)
        & quote_df["market"].eq(market)
        & quote_df["selection"].eq(selection)
        & quote_df["bookmaker"].eq(bookmaker)
    ]
    if len(matches) != 1:
        return None
    row = matches.iloc[0]
    return row["point"], int(row["american_odds"])
