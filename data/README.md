# Data

- `ballparks.csv` -- park HR factors (LHB/RHB) used by the run projection.
- `stadiums.csv` -- park name, city, roof type, and coordinates for the weather lookup.
- `raw/` -- gitignored cache of MLB Stats API and ESPN pulls used by `scripts/backtest.py`,
  `scripts/tune_win_model.py`, and `scripts/market_benchmark.py`. Safe to delete; the scripts
  re-fetch what they need.
