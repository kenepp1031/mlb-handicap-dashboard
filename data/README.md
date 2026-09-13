# Data folders

Store immutable, timestamped imports in `raw/` (odds, confirmed lineups, injury snapshots, weather, and final results). Do not overwrite an earlier snapshot.

Suggested filename pattern:

`YYYY-MM-DD_HHMM_source_dataset.csv`

The MVP deliberately has no automatic web scraper. Before adding one, choose a source with permitted API or export access and capture the source terms plus the data timestamp.
