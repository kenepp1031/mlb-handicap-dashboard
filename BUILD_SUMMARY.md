# MLB Handicap — Build Summary

## What existed before this session
A working Streamlit dashboard (`app.py`) with:
- **Live Board** tab — ESPN schedule + optional live sportsbook odds via The Odds API
- **Pitcher Ks** tab — strikeout projection from manually-set sliders
- **Hits & HRs** tab — hit/HR probability from manually-set sliders
- **Ballparks** tab — static park-factor table
- **Sources** tab — data source registry
- Support modules: `src/models.py` (projection math), `src/odds.py` (vig removal, EV, fair odds), `src/live_odds.py` (ESPN + Odds API clients)

Verified it actually ran (venv + `pip install -r requirements.txt` + headless launch → HTTP 200, no errors).

## What was added this session

### `src/mlb_stats.py` (new)
Client for the free, keyless official MLB Stats API (`statsapi.mlb.com`):
- `fetch_teams()` — all 30 teams
- `fetch_team_season_stats(team_id)` — season runs scored/allowed, games played
- `fetch_probable_pitchers(date)` — today's probable starters per game
- `search_players(name)` — player lookup by name
- `fetch_player_season_stats(player_id, group)` — pitching K rate, batting hit rate, HR rate

Verified live against the real API: team list, a team's season R/RA, a player search, a batter's hit/HR rate, and a pitcher's K rate all returned correctly-shaped data.

### `src/team_model.py` (new)
Transparent baseline game-outcome model:
- `pythagorean_win_pct(runs_scored, runs_allowed)` — Bill James Pythagorean expectation
- `log5_win_probability(team_pyth, opponent_pyth, home_field_edge)` — head-to-head win probability
- `project_total_runs(...)` — combined total-runs projection, park-adjustable

Sanity-checked: two evenly-matched teams → 50% before home edge, 54% with it; a stronger team (5.5 RS/3.8 RA) beats an average one ~70% of the time — all in line with expectations.

### New "Game Outcome" tab in `app.py`
Team dropdowns (auto-filled from the MLB Stats API) → win probability for each side, fair moneyline, projected total runs, and edge vs. a no-vig market price you enter. All auto-filled numbers remain editable.

### Auto-fill added to existing tabs
- **Pitcher Ks** — search any pitcher by name, or pick from today's probable starters, to pre-fill the season K-rate slider (still fully adjustable).
- **Hits & HRs** — search any batter by name to pre-fill season hit-rate and HR-rate sliders (still fully adjustable).

## Still open
- **Odds API key** — not yet added to `.streamlit/secrets.toml`. Live sportsbook prices on the Live Board won't appear until that's set (paste the key and I'll write it in, or add it yourself by copying `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`).
- The game-outcome model doesn't yet adjust for a specific starting pitcher beyond season team averages — noted as a limitation in the app itself.

## How to run it
```powershell
cd 'C:\MLB Handicap'
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
(The venv and dependencies are already installed from this session's verification pass.)
