# MLB Edge Board

Streamlit dashboard that prices every game on the MLB slate: a win probability and projected
score from a regressed Pythagorean / log5 model with a starting-pitcher adjustment, shown next
to DraftKings' no-vig moneyline, the public betting splits, weather, hot streaks, and the
playoff picture.

## Run it

```powershell
cd 'C:\MLB Handicap'
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

`Launch Dashboard.vbs` starts it with no console window (for a desktop shortcut using
`assets/app_icon.ico`); `Launch Dashboard.bat` does the same with the console visible.

## Sportsbook prices

DraftKings moneylines come from ESPN's public scoreboard feed: one request per date, cached
10 minutes, no key and no monthly quota. ESPN carries the line until first pitch, which is the
only time the board compares against it. The app no longer calls The Odds API at all, so it
cannot use up the key the prop board shares. The bets/money split bars are scraped from
DraftKings Network and need no key either.

## Layout

- `app.py` -- the dashboard
- `src/team_model.py` -- the win/run model. `game_projection()` is the one function both the
  app and the backtest call, so what is backtested is exactly what is on screen.
- `src/mlb_stats.py`, `src/live_odds.py`, `src/betting_splits.py`, `src/weather.py` -- feeds
  (MLB Stats API, ESPN scoreboard, DraftKings Network, Open-Meteo)
- `scripts/backtest.py` -- no-lookahead backtest of a season; caches API pulls under `data/raw/`
- `scripts/tune_win_model.py` -- refits the model constants on 2021-2024, scores 2025-2026
- `scripts/market_benchmark.py` -- model vs. the closing line (needs `scripts/fetch_espn_odds.py`)
- `scripts/check_app.py` -- regression checks; `--live` also drives the app against real feeds
- `config/sources.json` -- reference list of the sites and feeds behind the board

## Reading a card

- Win % and the predicted score are the model's. Confidence tiers come from the held-out
  backtest (see `confidence_tier()` in `app.py`).
- "DK xx% / edge +y" compares our win % with DraftKings' no-vig price, pregame only. Edges of
  3+ points light up green.
- The Bets / Money bars are DraftKings' public splits. Money running 10+ points ahead of bets
  on a side is tagged as sharp money: fewer bettors, bigger bets.
- A TBD starter means that side is priced off its whole staff; treat the number as rough.

These are research estimates. Model-market differences are not a validated betting edge.
