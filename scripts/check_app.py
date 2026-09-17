"""Regression checks; add --live to exercise the dashboard with real feeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.game_context import book_price, eligible_pregame
from src.mlb_stats import StatsFeedError, _get, fetch_probable_pitchers
from src.live_odds import OddsFeedError, fetch_espn_moneylines
from src.weather import WeatherFeedError, fetch_stadium_weather
from src.team_model import TeamLine, game_projection


class RegressionChecks(unittest.TestCase):
    def test_feed_timeouts_are_recoverable(self):
        calls = [(_get, ("/teams",), StatsFeedError),
                 (fetch_espn_moneylines, (date(2026, 9, 11),), OddsFeedError),
                 (fetch_stadium_weather, (40, -74), WeatherFeedError)]
        for fn, args, error_type in calls:
            with self.subTest(feed=fn.__name__), patch("src.feed.requests.get", side_effect=requests.Timeout("private-test-key")):
                with self.assertRaises(error_type) as caught:
                    fn(*args)
                self.assertNotIn("private-test-key", str(caught.exception))

    def test_invalid_json_is_recoverable(self):
        response = Mock()
        response.json.side_effect = ValueError("bad JSON")
        with patch("src.feed.requests.get", return_value=response):
            with self.assertRaises(StatsFeedError):
                _get("/teams")

    def test_doubleheader_pitchers_keep_game_ids(self):
        games = [{"gamePk": pk, "teams": {
            "home": {"team": {"name": "Boston Red Sox"}, "probablePitcher": {"id": pitcher}},
            "away": {"team": {"name": "Chicago White Sox"}}}}
                 for pk, pitcher in [(1, 101), (2, 202)]]
        with patch("src.mlb_stats._get", return_value={"dates": [{"games": games}]}):
            lookup = {p.game_pk: p for p in fetch_probable_pitchers(date(2026, 9, 11))}
        self.assertEqual(lookup[1].home_pitcher_id, 101)
        self.assertEqual(lookup[2].home_pitcher_id, 202)

    def test_odds_are_specific_to_teams_and_start(self):
        base = dict(game="Chicago White Sox @ Boston Red Sox", market="h2h",
                    selection="Boston Red Sox", bookmaker="DraftKings", point=None)
        first, second = "2026-09-11T17:00:00Z", "2026-09-11T23:00:00Z"
        quotes = pd.DataFrame([{**base, "commence_time": first, "american_odds": -110},
                               {**base, "commence_time": second, "american_odds": -180}])
        args = ("Chicago White Sox", "Boston Red Sox", "h2h", "Boston Red Sox", "DraftKings")
        self.assertEqual(book_price(quotes, *args, second)[1], -180)
        self.assertIsNone(book_price(quotes, *args, "2026-09-12T23:00:00Z"))
        self.assertIsNone(book_price(quotes, "New York Yankees", *args[1:], second))
        self.assertIsNone(book_price(pd.concat([quotes, quotes]), *args, second))

    def test_picks_exclude_live_completed_and_past_games(self):
        now = datetime(2026, 9, 11, 20, tzinfo=timezone.utc)
        for state, start, expected in [("Scheduled", "23:00", True),
                                       ("Final", "23:00", False),
                                       ("In Progress", "23:00", False),
                                       ("Scheduled", "17:00", False)]:
            game = SimpleNamespace(detailed_state=state, game_date=f"2026-09-11T{start}:00Z")
            self.assertEqual(eligible_pregame(game, now), expected)

    def test_weather_uses_current_hour_not_midnight(self):
        data = {"current": {"time": "2026-09-11T20:15", "weather_code": 0,
                            "temperature_2m": 75, "wind_speed_10m": 5},
                "hourly": {"time": ["2026-09-11T00:00", "2026-09-11T20:00"],
                           "precipitation_probability": [100, 10]}}
        with patch("src.weather.get_json", return_value=data):
            weather = fetch_stadium_weather(40, -74)
        self.assertEqual(weather.precipitation_prob, 10)
        self.assertFalse(weather.is_inclement)

    def test_model_probabilities_and_direction(self):
        average = TeamLine(450, 450, 100)
        stronger = TeamLine(550, 350, 100)
        neutral = game_projection(average, average, 4.5)
        improved = game_projection(stronger, average, 4.5)
        self.assertAlmostEqual(neutral.home_win_probability + neutral.away_win_probability, 1)
        self.assertGreater(improved.home_win_probability, neutral.home_win_probability)
        self.assertAlmostEqual(improved.projected_total_runs, improved.home_runs + improved.away_runs)


def live_check():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180)
    for label, run in [("initial load", lambda: app.run()),
                       ("previous week", lambda: app.button(key="prev_week").click().run()),
                       ("next week", lambda: app.button(key="next_week").click().run())]:
        run()
        failures = [e.message for e in app.exception]
        print(label, "exceptions:", failures, "errors:", [e.value for e in app.error], flush=True)
        assert not failures and not app.error, label
    with patch("src.mlb_stats.fetch_teams", side_effect=StatsFeedError("Test outage")):
        import streamlit as st
        st.cache_data.clear()
        outage = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
        assert not outage.exception
        assert any("Could not load MLB teams" in e.value for e in outage.error)
        print("team-feed outage: handled without crash", flush=True)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegressionChecks))
    if not result.wasSuccessful():
        sys.exit(1)
    if "--live" in sys.argv:
        live_check()
