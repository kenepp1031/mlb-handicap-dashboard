# MLB Handicap

Local dashboard for MLB player-prop research: pitcher strikeouts, batter hits, and home runs.

## Start here

1. Install Python 3.11+ from https://www.python.org/downloads/.
2. In PowerShell, run:
   ```powershell
   cd 'C:\MLB Handicap'
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Open the local URL Streamlit displays.

The dashboard includes transparent baseline projections and an optional live market board.

## Live odds setup

The live board uses [The Odds API](https://the-odds-api.com/) for current MLB moneyline, spread, and total quotes. Create a provider account, then copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and insert your key. The key remains local and should never be committed. Without a key, the dashboard still shows the ESPN schedule and consensus context.

Live prices are a market-data input, not a recommendation. Confirm lineups, starters, injury status, weather, and the price immediately before making any decision.

## Project layout

- `app.py` — dashboard entry point
- `src/` — probability models and odds utilities
- `data/ballparks.csv` — editable ballpark context table
- `config/sources.json` — source registry, purpose, and refresh rules

## Important modeling rules

- Model probability and betting price separately. A likely event is not automatically a value play.
- Remove sportsbook vig before comparing model probability to the market.
- Do not generate a final projection without confirmed lineups, a named starter, and a current injury check.
- Save every input snapshot and final result before backtesting.

## Build summary

This project is now an MLB research dashboard with a Live Board, an optional live sportsbook-odds feed, and transparent player-prop tools for pitcher strikeouts, batter hits, and home runs.

- **Live Board:** Displays the MLB slate and available consensus context through ESPN. When `ODDS_API_KEY` is configured, it also retrieves current moneyline, spread, and total prices from The Odds API, compares books, and highlights the best available price for each market selection.
- **Pitcher Ks:** Produces an expected-strikeout total, an over probability, fair odds, no-vig market probability, estimated model edge, and expected value per $1 using the entered market prices.
- **Hits and HRs:** Produces expected hits plus probabilities and fair odds for 1+ hit and 1+ home run outcomes.
- **Decision safeguards:** The dashboard keeps lineup, starter, injury, and weather checks visible, and labels all projections as research estimates rather than guarantees.

To run the dashboard locally, use a Python installation visible to your terminal, create a virtual environment, install `requirements.txt`, and run `streamlit run app.py`. The current workspace session cannot locate a Python executable, so the app has not been launched here yet. If `python --version` works in your own terminal, run the commands in **Start here**; otherwise, provide the output of `where python` to identify the correct executable path.
