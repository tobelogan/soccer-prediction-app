from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from soccer_predictor.autosource import StatsBombOpenSource, GoogleNewsAvailabilitySource
from soccer_predictor.advanced import AdvancedDataBundle, AdvancedFeatureStore


class FakeHTTP:
    def get_json(self, url, params=None, ttl_hours=None):
        if url.endswith("competitions.json"):
            return [{"competition_id": 1, "season_id": 10, "season_name": "2024/2025"}]
        if "/matches/1/10.json" in url:
            return [{
                "match_id": 123,
                "match_date": "2024-08-10",
                "home_team": {"home_team_name": "Arsenal"},
                "away_team": {"away_team_name": "Chelsea"},
                "home_score": 2,
                "away_score": 1,
            }]
        if "/events/123.json" in url:
            return [
                {
                    "id": "xi-h", "minute": 0,
                    "type": {"name": "Starting XI"}, "team": {"name": "Arsenal"},
                    "tactics": {"lineup": [
                        {"player": {"name": "A Keeper"}, "position": {"name": "Goalkeeper"}},
                        {"player": {"name": "A Striker"}, "position": {"name": "Center Forward"}},
                    ]},
                },
                {
                    "id": "xi-a", "minute": 0,
                    "type": {"name": "Starting XI"}, "team": {"name": "Chelsea"},
                    "tactics": {"lineup": [
                        {"player": {"name": "C Keeper"}, "position": {"name": "Goalkeeper"}},
                        {"player": {"name": "C Striker"}, "position": {"name": "Center Forward"}},
                    ]},
                },
                {
                    "id": "shot-1", "minute": 15, "type": {"name": "Shot"},
                    "team": {"name": "Arsenal"}, "player": {"name": "A Striker"},
                    "shot": {"statsbomb_xg": 0.42, "type": {"name": "Open Play"}},
                },
                {
                    "id": "pass-1", "minute": 15, "type": {"name": "Pass"},
                    "team": {"name": "Arsenal"}, "player": {"name": "A Keeper"},
                    "pass": {"assisted_shot_id": "shot-1"},
                },
                {
                    "id": "shot-2", "minute": 55, "type": {"name": "Shot"},
                    "team": {"name": "Chelsea"}, "player": {"name": "C Striker"},
                    "shot": {"statsbomb_xg": 0.25, "type": {"name": "Open Play"}},
                },
            ]
        raise AssertionError(url)


def test_statsbomb_open_sync(tmp_path):
    base = pd.DataFrame([{
        "Date": pd.Timestamp("2024-08-10"), "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
        "FTHG": 2, "FTAG": 1,
    }])
    report = StatsBombOpenSource(FakeHTTP()).sync(base, tmp_path)
    assert report.matched_fixtures == 1
    events = pd.read_csv(tmp_path / "events.csv")
    lineups = pd.read_csv(tmp_path / "lineups.csv")
    players = pd.read_csv(tmp_path / "player_stats.csv")
    assert len(events) == 2
    assert abs(events.loc[events.Player == "A Striker", "xG"].iloc[0] - 0.42) < 1e-9
    assert lineups["Confirmed"].astype(str).str.lower().isin(["true", "1"]).all()
    assert players.loc[players.Player == "A Keeper", "xA"].iloc[0] == 0.42


def test_news_classifier_is_conservative():
    assert GoogleNewsAvailabilitySource._classify("Player ruled out with hamstring injury")[0] == "out"
    assert GoogleNewsAvailabilitySource._classify("Player suspended for one match")[0] == "suspended"
    status, _, conf = GoogleNewsAvailabilitySource._classify("Player remains a doubt")
    assert status == "doubtful" and conf < 0.78


def test_availability_confidence_and_doubt_weighting():
    date = pd.Timestamp("2025-01-10")
    availability = pd.DataFrame([
        {"Date": date, "HomeTeam": "A", "AwayTeam": "B", "Team": "A", "Player": "P1", "Status": "out", "Reason": "injury", "ExpectedMinutes": 90, "PlayerXG90": 0.5, "PlayerXA90": 0.2, "SourceConfidence": 1.0},
        {"Date": date, "HomeTeam": "A", "AwayTeam": "B", "Team": "A", "Player": "P2", "Status": "doubtful", "Reason": "injury", "ExpectedMinutes": 90, "PlayerXG90": 0.5, "PlayerXA90": 0.2, "SourceConfidence": 0.5},
    ])
    store = AdvancedFeatureStore(AdvancedDataBundle(availability=availability))
    f = store.features_for({"Date": date, "HomeTeam": "A", "AwayTeam": "B"})
    # confirmed out = 1.0 injury; doubtful = 0.5 confidence * 0.4 status weight = 0.2
    assert abs(f["home_injuries"] - 1.2) < 1e-9
    assert abs(f["home_unavailable_xg90"] - 0.6) < 1e-9
