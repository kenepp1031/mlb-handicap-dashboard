from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.betting_splits import SplitsFeedError, fetch_mlb_betting_splits
from src.live_odds import OddsFeedError, fetch_espn_moneylines
from src.mlb_stats import (
    CURRENT_SEASON,
    StatsFeedError,
    fetch_league_runs,
    fetch_player_recent_stats,
    fetch_player_season_stats,
    fetch_probable_pitchers,
    fetch_schedule,
    fetch_standings,
    fetch_team_season_stats,
    fetch_teams,
)
from src.odds import remove_vig
from src.team_model import StarterLine, TeamLine, game_projection, league_runs_per_game, park_hr_factor
from src.weather import WeatherFeedError, fetch_stadium_weather
from src.game_context import PREGAME_STATES, book_price, eligible_pregame
from src.save_image import save_day_image_button

ROOT = Path(__file__).parent
BOOK = "DraftKings"
PITCHER_K_FLAG = 0.24
BATTER_K_FLAG = 0.23
HR_ALLOWED_FLAG = 0.035
PARK_HR_FLAG = 1.05
EDGE_FLAG_PTS = 3.0  # highlight when our win% beats DK's no-vig price by this many points
SHARP_GAP_PTS = 10.0  # money% this far above bets% on a side = fewer, bigger bets: the usual sharp-money read

st.set_page_config(page_title="MLB Edge Board", page_icon="⚾", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(0,234,255,0.08), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(157,0,255,0.07), transparent 40%),
        repeating-linear-gradient(180deg, rgba(0,234,255,0.015) 0px, rgba(0,234,255,0.015) 1px, transparent 1px, transparent 3px),
        #03060d;
}
header[data-testid="stHeader"] { display: none; }
.block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 2rem; animation: hud-fade-in 0.4s ease; }

@keyframes hud-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.04em; }

h1 {
    color: #00eaff !important;
    text-shadow: 0 0 10px rgba(0,234,255,0.6), 0 0 22px rgba(0,234,255,0.3);
    font-size: 1.9rem !important;
}
.hud-title-row {
    display: flex; align-items: center; gap: 14px;
}
.hud-logo { height: 44px; width: auto; }
.hud-logo-ball, .hud-logo-badge { filter: drop-shadow(0 0 6px rgba(0,234,255,0.35)); }

/* HUD ticker */
.hud-ticker {
    display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 16px 0;
}
.hud-chip {
    flex: 1 1 120px; background: linear-gradient(160deg, rgba(0,234,255,0.08), rgba(10,22,40,0.6));
    border: 1px solid rgba(0,234,255,0.35); border-radius: 6px; padding: 6px 10px;
    box-shadow: 0 0 12px rgba(0,234,255,0.08) inset;
}
.hud-chip .label { font-size: 0.62rem; color: #6fd8ff; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.85; }
.hud-chip .value { font-family: 'Orbitron', sans-serif; font-size: 1.15rem; color: #eafcff; text-shadow: 0 0 8px rgba(0,234,255,0.5); }
.hud-chip.alert { border-color: rgba(255,90,90,0.6); }
.hud-chip.alert .value { color: #ff8a8a; text-shadow: 0 0 8px rgba(255,60,60,0.6); }

/* Matchup card */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(155deg, rgba(10,22,40,0.85), rgba(4,8,16,0.9)) !important;
    border: 1px solid rgba(0,234,255,0.25) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 18px rgba(0,234,255,0.06), 0 0 1px rgba(0,234,255,0.4) inset !important;
    animation: hud-fade-in 0.35s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 10px 14px !important; gap: 0.35rem !important; }

.matchup-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
.matchup-side { display: flex; align-items: center; gap: 6px; }
.team-logo { height: 22px; width: 22px; object-fit: contain; opacity: 0.75; }
.team-name { font-family: 'Orbitron', sans-serif; font-size: 1.05rem; color: #eafcff; }
.team-score { font-family: 'Orbitron', sans-serif; font-size: 1.05rem; color: #00eaff; text-shadow: 0 0 6px rgba(0,234,255,0.5); }
.vs-sep { color: #3a6b80; font-size: 0.75rem; letter-spacing: 0.2em; }
.flame { font-size: 0.85rem; margin: 0 2px; filter: drop-shadow(0 0 4px rgba(255,140,0,0.8)); display: inline-flex; align-items: center; gap: 3px; }
.hot-stat { font-size: 0.6rem; color: #ff9d4d; letter-spacing: 0.02em; }
.playoff-row { display: flex; justify-content: space-between; align-items: center; margin: -2px 0 4px 0; }
.playoff-badge {
    font-size: 0.62rem; color: #ffcf7a; background: rgba(255,170,40,0.08);
    border: 1px solid rgba(255,170,40,0.3); border-radius: 4px; padding: 1px 6px;
    letter-spacing: 0.03em;
}
.hud-caption { color: #6fd8ff; font-size: 0.72rem; opacity: 0.85; margin-bottom: 4px; }

.model-row {
    display: flex; justify-content: space-between; align-items: center;
    margin: 6px 0 8px 0; padding: 6px 4px;
    border-top: 1px solid rgba(0,234,255,0.15); border-bottom: 1px solid rgba(0,234,255,0.15);
}
.model-row .side { display: flex; flex-direction: column; align-items: center; gap: 1px; flex: 1; }
.model-row .side .ml { font-family: 'Orbitron', sans-serif; font-size: 0.95rem; color: #eafcff; }
.model-row .side .winpct { font-size: 0.68rem; color: #6fd8ff; }

/* DraftKings public splits: bets% and money% as bars */
.split-bars { width: 100%; max-width: 150px; margin-top: 3px; display: flex; flex-direction: column; gap: 2px; }
.split-bar { display: grid; grid-template-columns: 34px 1fr 30px; align-items: center; gap: 4px; font-size: 0.58rem; color: #6fd8ff; }
.split-bar .lbl { text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.8; }
.split-bar .val { text-align: right; color: #cdeff9; }
.split-track { display: block; height: 6px; border-radius: 3px; background: rgba(0,234,255,0.08); border: 1px solid rgba(0,234,255,0.18); overflow: hidden; }
.split-fill { display: block; height: 100%; border-radius: 3px; }
.split-fill.bets { background: linear-gradient(90deg, rgba(0,234,255,0.45), #00eaff); }
.split-fill.money { background: linear-gradient(90deg, rgba(255,184,77,0.45), #ffb84d); }
.split-fill.sharp { background: linear-gradient(90deg, rgba(125,255,154,0.45), #7dff9a); box-shadow: 0 0 6px rgba(80,255,120,0.6); }
.sharp-tag { font-size: 0.6rem; color: #7dff9a; text-shadow: 0 0 6px rgba(80,255,120,0.45); letter-spacing: 0.04em; margin-top: 1px; }
.model-row .side .edge { font-size: 0.6rem; color: #6fd8ff; opacity: 0.8; }
.model-row .side .edge.pos { color: #7dff9a; opacity: 1; text-shadow: 0 0 6px rgba(80,255,120,0.45); }
.model-row .total { display: flex; flex-direction: column; align-items: center; gap: 1px; flex: 1; }
.model-row .total .label { font-size: 0.6rem; color: #6fd8ff; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.8; }
.model-row .total .value { font-family: 'Orbitron', sans-serif; font-size: 1.15rem; color: #00eaff; text-shadow: 0 0 8px rgba(0,234,255,0.5); }

.stadium-line {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
    gap: 4px 10px; color: #6fd8ff; font-size: 0.68rem; opacity: 0.85;
    background: rgba(0,234,255,0.04); border: 1px solid rgba(0,234,255,0.15);
    border-radius: 5px; padding: 4px 8px; margin-bottom: 6px;
}
.stadium-line .park { color: #9fe6ff; }
.stadium-line .wx { display: flex; gap: 8px; flex-wrap: wrap; }

.section-label {
    font-family: 'Orbitron', sans-serif; font-size: 0.62rem; color: #6fd8ff;
    letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7; margin: 8px 0 3px 0;
}
.player-row {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.74rem; color: #cdeff9; padding: 2px 0;
    border-bottom: 1px dashed rgba(0,234,255,0.12);
}
.player-row .who { display: flex; align-items: center; gap: 6px; }
.player-row .stat { color: #6fd8ff; font-size: 0.7rem; }
.player-row .tbd { color: #ffb84d; font-weight: 600; }

[data-testid="stAlert"] { padding: 6px 10px !important; font-size: 0.8rem !important; border-radius: 6px !important; }
.stCaption, [data-testid="stCaptionContainer"] { font-size: 0.72rem !important; }

/* Day-tab strip */
.day-tab-row div[data-testid="stButton"] button {
    width: 100%; border-radius: 6px 6px 0 0; border: 1px solid rgba(0,234,255,0.25);
    border-bottom: none; background: rgba(10,22,40,0.5); color: #6fd8ff;
    font-family: 'Orbitron', sans-serif; font-size: 0.72rem; letter-spacing: 0.05em;
    padding: 6px 2px; transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.day-tab-row div[data-testid="stButton"] button:hover {
    background: rgba(0,234,255,0.12); color: #eafcff; border-color: rgba(0,234,255,0.5);
}
.day-tab-row div[data-testid="stButton"] button:focus:not(:active) {
    border-color: rgba(0,234,255,0.5); color: #eafcff;
}
.day-tab-row .active-day div[data-testid="stButton"] button {
    background: linear-gradient(160deg, rgba(0,234,255,0.22), rgba(0,234,255,0.06));
    color: #eafcff; border-color: #00eaff;
    box-shadow: 0 0 12px rgba(0,234,255,0.35) inset;
    text-shadow: 0 0 6px rgba(0,234,255,0.6);
}
.day-tab-strip-underline { border-bottom: 1px solid rgba(0,234,255,0.25); margin: -1px 0 16px 0; }
.week-range-label {
    font-family: 'Orbitron', sans-serif; font-size: 0.8rem; color: #6fd8ff;
    letter-spacing: 0.08em; text-align: center; padding-top: 6px;
}

.top-pick-bubble {
    margin: 22px auto 4px auto; max-width: 640px;
    background: linear-gradient(160deg, rgba(255,170,40,0.12), rgba(10,22,40,0.75));
    border: 1px solid rgba(255,170,40,0.45); border-radius: 12px;
    padding: 12px 18px; box-shadow: 0 0 18px rgba(255,170,40,0.1);
}
.top-pick-title {
    font-family: 'Orbitron', sans-serif; font-size: 0.85rem; color: #ffcf7a;
    text-shadow: 0 0 8px rgba(255,170,40,0.4); margin-bottom: 3px;
}
.top-pick-meta { font-size: 0.72rem; color: #eafcff; opacity: 0.9; margin-bottom: 6px; }
.top-pick-reasons { margin: 0; padding-left: 18px; }
.top-pick-reasons li { font-size: 0.72rem; color: #cdeff9; line-height: 1.5; }
.top-pick-banner { margin: 4px 0 14px 0; max-width: none; }
.top-pick-banner .top-pick-meta { margin-bottom: 0; }

/* Scroll the whole document instead of an inner container, so a
   full-page screenshot doesn't clip anything below the fold. */
html, body, #root,
.stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"] {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def live_quotes(game_date: date):
    """DraftKings moneylines via ESPN's free scoreboard: no key and no request quota."""
    return fetch_espn_moneylines(game_date)


@st.cache_data(ttl=300, show_spinner=False)
def betting_splits():
    return fetch_mlb_betting_splits()


@st.cache_data(ttl=60, show_spinner=False)
def mlb_schedule(game_date: date):
    return fetch_schedule(game_date)


@st.cache_data(ttl=300, show_spinner=False)
def probable_pitchers(game_date: date):
    return fetch_probable_pitchers(game_date)


@st.cache_data(ttl=3600, show_spinner=False)
def team_season_stats(team_id: int, season: int = CURRENT_SEASON):
    return fetch_team_season_stats(team_id, season)


@st.cache_data(ttl=3600, show_spinner=False)
def player_season_stats(player_id: int, full_name: str, group: str, season: int = CURRENT_SEASON):
    return fetch_player_season_stats(player_id, full_name, group, season)


@st.cache_data(ttl=3600, show_spinner=False)
def league_run_environment() -> tuple[float, float | None]:
    """(this season's league runs per team-game, leaning on last season's early on; last season's)."""
    prev_runs, prev_games = fetch_league_runs(CURRENT_SEASON - 1)
    prev_rpg = prev_runs / prev_games if prev_games else None
    runs, games = fetch_league_runs(CURRENT_SEASON)
    return league_runs_per_game(runs, games, prev_rpg), prev_rpg


@st.cache_data(ttl=1800, show_spinner=False)
def player_recent_stats(player_id: int, full_name: str, group: str):
    return fetch_player_recent_stats(player_id, full_name, group)


@st.cache_data(ttl=1800, show_spinner=False)
def team_standings():
    return fetch_standings()


@st.cache_data(ttl=600, show_spinner=False)
def stadium_weather(lat: float, lon: float):
    return fetch_stadium_weather(lat, lon)


@st.cache_data(ttl=3600, show_spinner=False)
def stadiums() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "stadiums.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def park_factors() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "ballparks.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def team_ids() -> dict[str, int]:
    return {t.name: t.team_id for t in fetch_teams()}


@st.cache_data(show_spinner=False)
def image_b64(filename: str) -> str | None:
    logo_path = ROOT / "assets" / filename
    if not logo_path.exists():
        return None
    return base64.b64encode(logo_path.read_bytes()).decode("ascii")


def team_logo_url(team_id: int | None) -> str | None:
    if not team_id:
        return None
    return f"https://www.mlbstatic.com/team-logos/{team_id}.svg"


ET = ZoneInfo("America/New_York")


def game_time_label(game_date_iso: str) -> tuple[datetime, str]:
    if not game_date_iso:
        return datetime.max.replace(tzinfo=ET), "Time TBD"
    dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00")).astimezone(ET)
    hour12 = dt.hour % 12 or 12
    return dt, f"{hour12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'} ET"


def confidence_tier(pick_probability: float) -> tuple[str, int]:
    """Simple, plain-language confidence read for a pick, with a 1-10 gut-check score.

    Tiers follow the held-out 2025-2026 backtest (scripts/tune_win_model.py, 4,531
    regular-season games never used in fitting): 60-65% picks hit 61.9%, 65-70% hit 66.4%,
    and 70%+ hit 78.1%. The 70%+ group is only 73 games, so it shares the top tier rather
    than getting a 9/10 of its own. (The previous model's top buckets flattened out at
    56-65%, which is why this used to be capped at 7/10.)"""
    if pick_probability < 0.55:
        return "Leave alone", 3
    if pick_probability < 0.60:
        return "Iffy", 5
    if pick_probability < 0.65:
        return "Favorable", 7
    return "Strong", 8


def pitcher_rating(era: float | None) -> str:
    if era is None:
        return ""
    if era < 3.50:
        return "Great"
    if era < 4.20:
        return "Good"
    if era < 5.00:
        return "Average"
    return "Rough"


def hitting_rating(ops: float | None) -> str:
    if ops is None:
        return ""
    if ops >= 0.760:
        return "Hot bats"
    if ops >= 0.700:
        return "Average bats"
    return "Cold bats"


def pitcher_is_hot(season_stats, recent_stats) -> bool:
    """Flame-worthy: ERA at least a run better over the trailing 14 days than
    the season line, on a big enough recent sample to mean something."""
    if not season_stats or not recent_stats:
        return False
    if recent_stats.plate_appearances_or_batters < 12:
        return False
    if season_stats.era is None or recent_stats.era is None:
        return False
    return (season_stats.era - recent_stats.era) >= 1.00


def build_top_pick_note(summaries) -> dict | None:
    """Picks the day's single highest-confidence game and writes out, in plain English,
    what's actually driving that number -- not just the win% itself, but the offense/
    pitching gap behind it, any hot streak, playoff stakes, a starter running hot, and
    any batter-vs-pitcher matchup edge already flagged for the card."""
    # A game with a TBD starter is priced off a staff average, so it can't be the top pick.
    candidates = [s for s in summaries if not s["tbd_teams"] and eligible_pregame(s["game"])]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: max(s["home_win_prob"], s["away_win_prob"]))
    g = best["game"]
    is_home = best["home_win_prob"] >= best["away_win_prob"]
    pick_team = g.home_team if is_home else g.away_team
    opp_team = g.away_team if is_home else g.home_team
    pick_prob = max(best["home_win_prob"], best["away_win_prob"])
    tier_label, tier_score = confidence_tier(pick_prob)

    pick_stats = best.get("home_stats") if is_home else best.get("away_stats")
    opp_stats = best.get("away_stats") if is_home else best.get("home_stats")
    standing = best.get("home_standing") if is_home else best.get("away_standing")
    pitcher_hot = best.get("home_pitcher_hot") if is_home else best.get("away_pitcher_hot")
    pitcher_season = best.get("home_pitcher_stats") if is_home else best.get("away_pitcher_stats")
    pitcher_recent = best.get("home_pitcher_recent") if is_home else best.get("away_pitcher_recent")
    probable = best.get("probable")
    pitcher_name = (probable.home_pitcher if is_home else probable.away_pitcher) if probable else None
    projected_pick_runs = best.get("home_projected_runs") if is_home else best.get("away_projected_runs")
    projected_opp_runs = best.get("away_projected_runs") if is_home else best.get("home_projected_runs")

    reasons = []

    # Lead with the actual offense-vs-pitching case, not just the headline number.
    if pick_stats and opp_stats and pick_stats.runs_scored_per_game and opp_stats.runs_allowed_per_game:
        reasons.append(
            f"{pick_team} score {pick_stats.runs_scored_per_game:.1f} runs/game on the season, and {opp_team} "
            f"give up {opp_stats.runs_allowed_per_game:.1f}/game — that gap, along with the starting-pitcher matchup, drives the "
            f"model's {pick_prob:.0%} win probability here, projecting to roughly {round(projected_pick_runs)}-"
            f"{round(projected_opp_runs)}."
        )
    else:
        reasons.append(f"The model gives {pick_team} a {pick_prob:.0%} win probability ({tier_label}, {tier_score}/10).")

    if pitcher_hot and pitcher_name and pitcher_season and pitcher_recent and pitcher_season.era and pitcher_recent.era:
        reasons.append(
            f"{pitcher_name} has been nearly unhittable lately — a {pitcher_recent.era:.2f} ERA over his last 14 "
            f"days, way down from his {pitcher_season.era:.2f} season mark, so he's throwing better than his "
            f"full-season stat line even suggests."
        )

    if standing and standing.is_hot:
        streak_phrase = (
            f"on a {standing.streak_number}-game win streak" if standing.streak_type == "wins" and standing.streak_number >= 3
            else f"{standing.last10_wins}-{standing.last10_losses} in their last 10"
        )
        reasons.append(f"{pick_team} are {streak_phrase} — the whole team is playing above its season average right now.")

    if standing and standing.playoff_status:
        reasons.append(f"There's added motivation too: {pick_team} are {standing.playoff_status.lower()} in the playoff race.")

    split = team_split(pick_team)
    if is_sharp(split):
        reasons.append(
            f"DraftKings' money agrees: {pick_team} have {split.pct_handle:.0%} of the money on only "
            f"{split.pct_bets:.0%} of the bets, so the bigger bets are landing on this side."
        )

    for note in best["notes"]:
        # These are already written as full sentences (e.g. "favorable K matchup",
        # "power bats have a favorable matchup") -- just strip the leading emoji.
        reasons.append(note.split(" ", 1)[-1] if note[:1] in "🎯💥" else note)

    return {
        "game_pk": g.game_pk,
        "pick_team": pick_team, "opp_team": opp_team, "pick_prob": pick_prob,
        "tier_label": tier_label, "tier_score": tier_score, "reasons": reasons,
    }


def edge_html(model_prob: float, market_prob: float | None, flag: bool = True) -> str:
    """DK's no-vig probability next to ours, and the gap in percentage points. flag=False
    never highlights the edge (a TBD starter means our number is a staff average)."""
    if market_prob is None:
        return ""
    edge = (model_prob - market_prob) * 100
    cls = "edge pos" if flag and edge >= EDGE_FLAG_PTS else "edge"
    return f'<span class="{cls}">DK {market_prob:.0%} · edge {edge:+.1f}</span>'


def starter_line(season_stats, prev_stats) -> StarterLine:
    """A probable starter's runs/outs this season and last (zeros where he has no line)."""
    return StarterLine(
        runs=season_stats.runs_allowed or 0.0, outs=season_stats.outs or 0.0,
        prev_runs=prev_stats.runs_allowed or 0.0, prev_outs=prev_stats.outs or 0.0,
    )


def team_line(stats, prev_stats, starter: StarterLine | None) -> TeamLine:
    return TeamLine(
        runs_scored=stats.runs_scored, runs_allowed=stats.runs_allowed, games=stats.games_played,
        prev_rs_pg=prev_stats.runs_scored_per_game if prev_stats.games_played else None,
        prev_ra_pg=prev_stats.runs_allowed_per_game if prev_stats.games_played else None,
        starter=starter,
    )


def compute_game(g, probables, teams_by_name, standings):
    """Returns a dict of everything a matchup card needs, or None if season stats aren't available."""
    home_id = teams_by_name.get(g.home_team)
    away_id = teams_by_name.get(g.away_team)
    if not (home_id and away_id):
        return None

    home_standing = standings.get(home_id)
    away_standing = standings.get(away_id)

    home_stats = team_season_stats(home_id)
    away_stats = team_season_stats(away_id)

    notes = []
    home_pitcher_stats = away_pitcher_stats = None
    home_pitcher_recent = away_pitcher_recent = None
    home_pitcher_hot = away_pitcher_hot = False
    home_starter = away_starter = None
    probable = probables.get(g.game_pk)
    if probable:
        if probable.home_pitcher_id:
            home_pitcher_stats = player_season_stats(probable.home_pitcher_id, probable.home_pitcher, "pitching")
            home_pitcher_recent = player_recent_stats(probable.home_pitcher_id, probable.home_pitcher, "pitching")
            home_pitcher_hot = pitcher_is_hot(home_pitcher_stats, home_pitcher_recent)
            home_starter = starter_line(
                home_pitcher_stats,
                player_season_stats(probable.home_pitcher_id, probable.home_pitcher, "pitching", CURRENT_SEASON - 1),
            )
            if home_pitcher_stats.k_rate and home_pitcher_stats.k_rate >= PITCHER_K_FLAG and away_stats.batting_k_rate and away_stats.batting_k_rate >= BATTER_K_FLAG:
                notes.append(f"🎯 {probable.home_pitcher} ({home_pitcher_stats.k_rate:.1%} K rate) faces a {g.away_team} lineup that strikes out often ({away_stats.batting_k_rate:.1%}) — strikeout upside.")
        if probable.away_pitcher_id:
            away_pitcher_stats = player_season_stats(probable.away_pitcher_id, probable.away_pitcher, "pitching")
            away_pitcher_recent = player_recent_stats(probable.away_pitcher_id, probable.away_pitcher, "pitching")
            away_pitcher_hot = pitcher_is_hot(away_pitcher_stats, away_pitcher_recent)
            away_starter = starter_line(
                away_pitcher_stats,
                player_season_stats(probable.away_pitcher_id, probable.away_pitcher, "pitching", CURRENT_SEASON - 1),
            )
            if away_pitcher_stats.k_rate and away_pitcher_stats.k_rate >= PITCHER_K_FLAG and home_stats.batting_k_rate and home_stats.batting_k_rate >= BATTER_K_FLAG:
                notes.append(f"🎯 {probable.away_pitcher} ({away_pitcher_stats.k_rate:.1%} K rate) faces a {g.home_team} lineup that strikes out often ({home_stats.batting_k_rate:.1%}) — strikeout upside.")
    if home_stats.pitching_hr_rate_allowed and home_stats.pitching_hr_rate_allowed >= HR_ALLOWED_FLAG and park_hr_factor(park_factors(), g.home_team) >= PARK_HR_FLAG:
        notes.append(f"💥 {g.home_team} staff allows HRs at an elevated rate ({home_stats.pitching_hr_rate_allowed:.1%}/PA) in a HR-friendly park — {g.away_team} power bats have a favorable matchup.")
    if away_stats.pitching_hr_rate_allowed and away_stats.pitching_hr_rate_allowed >= HR_ALLOWED_FLAG:
        notes.append(f"💥 {g.away_team} staff allows HRs at an elevated rate ({away_stats.pitching_hr_rate_allowed:.1%}/PA) — {g.home_team} power bats have a favorable matchup.")

    # A TBD side gets no starter adjustment -- game_projection() falls back to that team's
    # whole-staff run prevention -- so the card has to say its number is rough.
    tbd_teams = [
        team for team, pitcher_id in (
            (g.home_team, probable.home_pitcher_id if probable else None),
            (g.away_team, probable.away_pitcher_id if probable else None),
        )
        if not pitcher_id
    ]

    league_rpg, prev_league_rpg = league_run_environment()
    hr_factor = (park_hr_factor(park_factors(), g.home_team) + 1.0) / 2
    projection = game_projection(
        team_line(home_stats, team_season_stats(home_id, CURRENT_SEASON - 1), home_starter),
        team_line(away_stats, team_season_stats(away_id, CURRENT_SEASON - 1), away_starter),
        league_rpg, prev_league_rpg, hr_factor,
    )
    home_win_prob, away_win_prob = projection.home_win_probability, projection.away_win_probability
    home_projected_runs, away_projected_runs = projection.home_runs, projection.away_runs
    projected_total = projection.projected_total_runs

    weather = None
    stadium_row = stadiums()[stadiums()["team"] == g.home_team]
    stadium = stadium_row.iloc[0] if not stadium_row.empty else None
    if stadium is not None and stadium["roof_type"] != "indoor":
        try:
            weather = stadium_weather(stadium["lat"], stadium["lon"])
        except WeatherFeedError:
            weather = None

    return {
        "game": g, "stadium": stadium, "weather": weather,
        "home_win_prob": home_win_prob, "away_win_prob": away_win_prob,
        "projected_total": projected_total,
        "home_projected_runs": home_projected_runs, "away_projected_runs": away_projected_runs,
        "notes": notes,
        "home_stats": home_stats, "away_stats": away_stats,
        "probable": probable, "tbd_teams": tbd_teams,
        "home_pitcher_stats": home_pitcher_stats, "away_pitcher_stats": away_pitcher_stats,
        "home_pitcher_recent": home_pitcher_recent, "away_pitcher_recent": away_pitcher_recent,
        "home_pitcher_hot": home_pitcher_hot, "away_pitcher_hot": away_pitcher_hot,
        "home_id": home_id, "away_id": away_id,
        "home_standing": home_standing, "away_standing": away_standing,
    }


if "week_start" not in st.session_state:
    today = date.today()
    st.session_state.week_start = today - timedelta(days=today.weekday())
if "selected_day_idx" not in st.session_state:
    st.session_state.selected_day_idx = date.today().weekday()


def load_day(day, teams_by_name):
    try:
        games = mlb_schedule(day)
    except StatsFeedError as error:
        st.error(f"Could not load schedule for {day.strftime('%b %d')}: {error}")
        return []

    if not games:
        return []

    try:
        probables = {p.game_pk: p for p in probable_pitchers(day)}
    except StatsFeedError:
        probables = {}

    try:
        standings = team_standings()
    except StatsFeedError:
        standings = {}

    # Each game's stats/weather/pitcher fetches are independent network calls, so run them
    # concurrently -- on a cold cache this is a ~10-game day making ~50+ sequential API
    # round trips otherwise, which is most of what makes first load feel slow.
    order = {g.game_pk: i for i, g in enumerate(games)}
    day_summaries: list = [None] * len(games)
    with ThreadPoolExecutor(max_workers=min(8, len(games))) as pool:
        futures = {
            pool.submit(compute_game, g, probables, teams_by_name, standings): g
            for g in games
        }
        for future in as_completed(futures):
            g = futures[future]
            try:
                summary = future.result()
            except StatsFeedError as error:
                st.warning(f"Could not load {g.game}: {error}")
                continue
            if summary:
                day_summaries[order[g.game_pk]] = summary
    return [s for s in day_summaries if s is not None]


top_left, top_right = st.columns([3, 1], vertical_alignment="bottom")
with top_left:
    ball_b64 = image_b64("neon_baseball.png")
    badge_b64 = image_b64("neon_badge.png")
    ball_img = f'<img src="data:image/png;base64,{ball_b64}" class="hud-logo hud-logo-ball" />' if ball_b64 else "⚾"
    mlb_img = f'<img src="data:image/png;base64,{badge_b64}" class="hud-logo hud-logo-badge" />' if badge_b64 else ""
    st.markdown(
        f'<h1 class="hud-title-row">{ball_img}<span>MLB EDGE BOARD</span>{mlb_img}</h1>',
        unsafe_allow_html=True,
    )
with top_right:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button(":material/refresh:", help="Refresh live data"):
            live_quotes.clear()
            mlb_schedule.clear()
            probable_pitchers.clear()
            stadium_weather.clear()
            betting_splits.clear()
            team_standings.clear()
            player_recent_stats.clear()
            team_season_stats.clear()
            player_season_stats.clear()
            league_run_environment.clear()

splits_by_team = {}
try:
    splits_by_team = betting_splits()
except SplitsFeedError:
    splits_by_team = {}


def team_mascot(team_name: str) -> str:
    words = team_name.split()
    return " ".join(words[-2:]) if words[-1] == "Sox" else words[-1]


def team_split(team_name: str):
    """DK's splits page uses short team names (e.g. 'NY Yankees'); match on the
    mascot, which lines up with our full MLB names ('New York Yankees')."""
    mascot = team_mascot(team_name)
    for short_name, game in splits_by_team.items():
        if team_mascot(short_name) == mascot:
            return game.teams[short_name]
    return None


def is_sharp(split) -> bool:
    """Money% running SHARP_GAP_PTS or more ahead of bets% on a side: fewer bettors but
    more dollars, so the average bet is bigger -- the usual public read for sharp money."""
    return (
        split is not None and split.pct_bets is not None and split.pct_handle is not None
        and (split.pct_handle - split.pct_bets) * 100 >= SHARP_GAP_PTS
    )


def split_bars_html(split) -> str:
    """DK's bets% and money% for one side as two bars; the money bar turns green with a
    tag when that side is drawing sharp money."""
    if not split or split.pct_bets is None or split.pct_handle is None:
        return ""
    sharp = is_sharp(split)
    money_cls = "money sharp" if sharp else "money"
    html = (
        '<div class="split-bars">'
        f'<div class="split-bar"><span class="lbl">Bets</span><span class="split-track">'
        f'<span class="split-fill bets" style="width:{split.pct_bets:.0%}"></span></span>'
        f'<span class="val">{split.pct_bets:.0%}</span></div>'
        f'<div class="split-bar"><span class="lbl">Money</span><span class="split-track">'
        f'<span class="split-fill {money_cls}" style="width:{split.pct_handle:.0%}"></span></span>'
        f'<span class="val">{split.pct_handle:.0%}</span></div>'
    )
    if sharp:
        html += '<div class="sharp-tag">💎 Sharp money</div>'
    return html + "</div>"

try:
    teams_by_name = team_ids()
except StatsFeedError as error:
    st.error(f"Could not load MLB teams: {error}")
    st.stop()

week_start = st.session_state.week_start
week_dates = [week_start + timedelta(days=i) for i in range(7)]

prev_col, strip_col, next_col = st.columns([1, 10, 1], vertical_alignment="center")
with prev_col:
    if st.button("◀", help="Previous week", key="prev_week"):
        st.session_state.week_start -= timedelta(days=7)
        st.session_state.selected_day_idx = 0
        st.rerun()
with next_col:
    if st.button("▶", help="Next week", key="next_week"):
        st.session_state.week_start += timedelta(days=7)
        st.session_state.selected_day_idx = 0
        st.rerun()
with strip_col:
    st.markdown(
        f'<div class="week-range-label">{week_dates[0].strftime("%b %d")} – {week_dates[-1].strftime("%b %d, %Y")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="day-tab-row">', unsafe_allow_html=True)
    tab_cols = st.columns(7)
    for i, day in enumerate(week_dates):
        with tab_cols[i]:
            is_active = i == st.session_state.selected_day_idx
            st.markdown(f'<div class="{"active-day" if is_active else ""}">', unsafe_allow_html=True)
            label = f"{day.strftime('%a').upper()} {day.month}/{day.day}"
            if st.button(label, key=f"day_tab_{i}"):
                st.session_state.selected_day_idx = i
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div><div class="day-tab-strip-underline"></div>', unsafe_allow_html=True)

selected_day = week_dates[st.session_state.selected_day_idx]

quote_df = pd.DataFrame()
try:
    quote_df = pd.DataFrame([q.__dict__ for q in live_quotes(selected_day)])
except OddsFeedError as error:
    st.warning(f"Odds feed unavailable: {error}")
st.caption("Research estimates using current season statistics. Historical cards are not saved pregame predictions; live-game estimates do not adjust for the score. Model-market differences are not a validated betting edge.")
with st.spinner(f"Loading {selected_day.strftime('%A, %b %d')}…"):
    summaries = load_day(selected_day, teams_by_name)


def render_game_card(s, top_pick=None):
    g = s["game"]
    with st.container(border=True):
        score_html = f'<span class="team-score">{g.home_score} — {g.away_score}</span>' if g.detailed_state not in PREGAME_STATES else '<span class="vs-sep">VS</span>'
        home_logo = team_logo_url(s.get("home_id"))
        away_logo = team_logo_url(s.get("away_id"))
        home_logo_html = f'<img class="team-logo" src="{home_logo}" />' if home_logo else ""
        away_logo_html = f'<img class="team-logo" src="{away_logo}" />' if away_logo else ""
        home_standing, away_standing = s.get("home_standing"), s.get("away_standing")

        def team_flame_html(standing) -> str:
            if not standing or not standing.is_hot:
                return ""
            record = f"{standing.last10_wins}-{standing.last10_losses} L10"
            return f'<span class="flame" title="{record}">🔥<span class="hot-stat">{record}</span></span>'

        home_flame = team_flame_html(home_standing)
        away_flame = team_flame_html(away_standing)
        st.markdown(
            f'<div class="matchup-row">'
            f'<span class="matchup-side">{home_logo_html}<span class="team-name">{g.home_team}</span>{home_flame}</span>'
            f'{score_html}'
            f'<span class="matchup-side">{away_flame}<span class="team-name">{g.away_team}</span>{away_logo_html}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        home_playoff = home_standing.playoff_status if home_standing else None
        away_playoff = away_standing.playoff_status if away_standing else None
        if home_playoff or away_playoff:
            home_badge = f'<span class="playoff-badge">🏁 {home_playoff}</span>' if home_playoff else '<span></span>'
            away_badge = f'<span class="playoff-badge">🏁 {away_playoff}</span>' if away_playoff else '<span></span>'
            st.markdown(
                f'<div class="playoff-row">{home_badge}{away_badge}</div>',
                unsafe_allow_html=True,
            )

        ml_home = book_price(quote_df, g.away_team, g.home_team, "h2h", g.home_team, BOOK, g.game_date) if not quote_df.empty else None
        ml_away = book_price(quote_df, g.away_team, g.home_team, "h2h", g.away_team, BOOK, g.game_date) if not quote_df.empty else None
        home_ml_html = f'<span class="ml">{ml_home[1]:+d}</span>' if ml_home else ""
        away_ml_html = f'<span class="ml">{ml_away[1]:+d}</span>' if ml_away else ""
        # Only compare before first pitch: once a game starts, DK's line is a live in-game price.
        market_home = (
            remove_vig(ml_home[1], ml_away[1])[0]
            if ml_home and ml_away and eligible_pregame(g) else None
        )
        tbd_teams = s["tbd_teams"] if g.detailed_state in PREGAME_STATES else []
        home_edge_html = edge_html(s["home_win_prob"], market_home, flag=not tbd_teams)
        away_edge_html = edge_html(s["away_win_prob"], None if market_home is None else 1 - market_home, flag=not tbd_teams)

        home_split_html = split_bars_html(team_split(g.home_team))
        away_split_html = split_bars_html(team_split(g.away_team))

        st.markdown(
            f'<div class="model-row">'
            f'<div class="side">{home_ml_html}<span class="winpct">{s["home_win_prob"]:.0%} win</span>{home_edge_html}{home_split_html}</div>'
            f'<div class="total">'
            f'<span class="label">Our predicted score</span>'
            f'<span class="value">{round(s["home_projected_runs"])} – {round(s["away_projected_runs"])}</span>'
            f'</div>'
            f'<div class="side">{away_ml_html}<span class="winpct">{s["away_win_prob"]:.0%} win</span>{away_edge_html}{away_split_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        stadium = s["stadium"]
        if stadium is not None:
            w = s["weather"]
            if w:
                wx_html = (
                    f'<span class="wx">Current weather: {w.temperature_f:.0f}°F · {w.wind_mph:.0f} mph wind · '
                    f'{w.precipitation_prob:.0f}% precip · {w.condition}</span>'
                )
            elif stadium["roof_type"] == "indoor":
                wx_html = '<span class="wx">Indoor — weather not a factor</span>'
            else:
                wx_html = ""
            st.markdown(
                f'<div class="stadium-line"><span class="park">📍 {stadium["park"]} · {stadium["city"]}</span>{wx_html}</div>',
                unsafe_allow_html=True,
            )
            if w and w.is_inclement:
                roof_note = " (retractable roof may close)" if stadium["roof_type"] == "retractable" else ""
                st.warning(f"Inclement weather currently reported{roof_note}; check the game-time forecast.")

        pick_team = g.home_team if s["home_win_prob"] >= s["away_win_prob"] else g.away_team
        pick_prob = max(s["home_win_prob"], s["away_win_prob"])
        tier_label, tier_score = confidence_tier(pick_prob)
        st.markdown(f'<div class="hud-caption">Confidence: {tier_label} ({tier_score}/10) — {pick_team} to win</div>', unsafe_allow_html=True)
        if tbd_teams:
            lead = "Both starters are TBD" if len(tbd_teams) == 2 else f"{tbd_teams[0]} starter is TBD"
            st.warning(
                f"⚠️ {lead}. The win % is using whole-staff averages in place of a named pitcher, so "
                f"treat the confidence and DK edge as rough — they can move several points once the "
                f"starter is announced."
            )

        probable = s.get("probable")
        home_p = s.get("home_pitcher_stats")
        away_p = s.get("away_pitcher_stats")
        if probable or tbd_teams:
            st.markdown('<div class="section-label">Starting pitchers</div>', unsafe_allow_html=True)
            tbd_html = '<span class="tbd">TBD ⚠️</span>'
            home_name = (probable.home_pitcher if probable else None) or tbd_html
            away_name = (probable.away_pitcher if probable else None) or tbd_html
            home_era = home_p.era if home_p else None
            away_era = away_p.era if away_p else None
            home_rating = pitcher_rating(home_era)
            away_rating = pitcher_rating(away_era)
            home_stat = f"{home_era:.2f} ERA · {home_rating}" if home_era is not None else ""
            away_stat = f"{away_era:.2f} ERA · {away_rating}" if away_era is not None else ""
            home_recent = s.get("home_pitcher_recent")
            away_recent = s.get("away_pitcher_recent")
            if s.get("home_pitcher_hot") and home_recent and home_recent.era is not None:
                home_stat += f' <span class="hot-stat">🔥 {home_recent.era:.2f} ERA last 14d</span>'
            if s.get("away_pitcher_hot") and away_recent and away_recent.era is not None:
                away_stat += f' <span class="hot-stat">🔥 {away_recent.era:.2f} ERA last 14d</span>'
            home_pitcher_flame = '<span class="flame" title="Hot over last 14 days">🔥</span>' if s.get("home_pitcher_hot") else ""
            away_pitcher_flame = '<span class="flame" title="Hot over last 14 days">🔥</span>' if s.get("away_pitcher_hot") else ""
            st.markdown(
                f'<div class="player-row"><span class="who">{home_logo_html} {home_name}{home_pitcher_flame}</span><span class="stat">{home_stat}</span></div>'
                f'<div class="player-row"><span class="who">{away_logo_html} {away_name}{away_pitcher_flame}</span><span class="stat">{away_stat}</span></div>',
                unsafe_allow_html=True,
            )

        home_stats, away_stats = s.get("home_stats"), s.get("away_stats")
        if home_stats is not None and away_stats is not None:
            st.markdown('<div class="section-label">Team hitting</div>', unsafe_allow_html=True)
            home_bat = f"{home_stats.ops:.3f} OPS · {hitting_rating(home_stats.ops)}" if home_stats.ops else ""
            away_bat = f"{away_stats.ops:.3f} OPS · {hitting_rating(away_stats.ops)}" if away_stats.ops else ""
            st.markdown(
                f'<div class="player-row"><span class="who">{home_logo_html} {g.home_team}</span><span class="stat">{home_bat}</span></div>'
                f'<div class="player-row"><span class="who">{away_logo_html} {g.away_team}</span><span class="stat">{away_bat}</span></div>',
                unsafe_allow_html=True,
            )

        for note in s["notes"]:
            st.info(note)

        if top_pick and top_pick["game_pk"] == g.game_pk:
            reasons_html = "".join(f"<li>{r}</li>" for r in top_pick["reasons"])
            st.markdown(
                f'<div class="top-pick-bubble">'
                f'<div class="top-pick-title">⭐ Why this is our top pick</div>'
                f'<ul class="top-pick-reasons">{reasons_html}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_grid(items, render_fn, max_cols=3, **kwargs):
    cols_count = min(max_cols, max(2, -(-len(items) // 3))) if len(items) > 4 else 2
    cols = st.columns(cols_count)
    for i, item in enumerate(items):
        with cols[i % cols_count]:
            render_fn(item, **kwargs)


live_now = sum(1 for s in summaries if s["game"].detailed_state in ("In Progress", "Manager Challenge", "Umpire Review"))
alerts = sum(1 for s in summaries if s["weather"] and s["weather"].is_inclement)

tbd_games = sum(1 for s in summaries if s["tbd_teams"] and s["game"].detailed_state in PREGAME_STATES)

picks = []
for s in summaries:
    if s["tbd_teams"] or not eligible_pregame(s["game"]):
        continue  # priced off a staff average -- not a "most confident" pick
    if s["home_win_prob"] >= s["away_win_prob"]:
        picks.append((s["game"].home_team, s["home_win_prob"]))
    else:
        picks.append((s["game"].away_team, s["away_win_prob"]))
picks.sort(key=lambda p: p[1], reverse=True)
top_picks_text = " · ".join(f"{name.split()[-1]} {prob:.0%}" for name, prob in picks[:3]) if picks else "—"

with st.container(horizontal=True, horizontal_alignment="right"):
    save_day_image_button(filename=f"mlb-picks-{selected_day.isoformat()}.png", key="save_day_image")

ticker_html = f"""<div class="hud-ticker">
<div class="hud-chip"><div class="label">Day</div><div class="value">{selected_day.strftime('%a %b %d')} · {len(summaries)} games</div></div>
<div class="hud-chip"><div class="label">Live now</div><div class="value">{live_now}</div></div>
<div class="hud-chip{' alert' if alerts else ''}"><div class="label">Weather alerts</div><div class="value">{alerts}</div></div>
<div class="hud-chip{' alert' if tbd_games else ''}"><div class="label">TBD starters</div><div class="value">{tbd_games}</div></div>
<div class="hud-chip"><div class="label">Most confident</div><div class="value" style="font-size:0.9rem">{top_picks_text}</div></div>
</div>"""
st.markdown(ticker_html, unsafe_allow_html=True)

top_pick = build_top_pick_note(summaries)
if top_pick:
    st.markdown(
        f'<div class="top-pick-bubble top-pick-banner">'
        f'<div class="top-pick-title">⭐ Most confident pick — {top_pick["pick_team"]} over {top_pick["opp_team"]}</div>'
        f'<div class="top-pick-meta">{top_pick["pick_prob"]:.0%} win probability · {top_pick["tier_label"]} ({top_pick["tier_score"]}/10) '
        f'— reasoning noted under the game card below</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not summaries:
    st.info("No game cards available for this day. Check any data-feed warnings above.")
else:
    grouped: dict[str, list] = {}
    sort_key: dict[str, datetime] = {}
    for s in summaries:
        dt, label = game_time_label(s["game"].game_date)
        grouped.setdefault(label, []).append(s)
        sort_key[label] = dt

    for label in sorted(grouped, key=lambda l: sort_key[l]):
        st.markdown(f'<div class="hud-caption">🕒 {label}</div>', unsafe_allow_html=True)
        render_grid(grouped[label], render_game_card, top_pick=top_pick)
