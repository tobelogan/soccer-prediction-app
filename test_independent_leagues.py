import pandas as pd

from soccer_predictor import data as data_mod
from soccer_predictor.bundle import LeagueModelBundle


def test_target_applies_to_each_league(monkeypatch):
    calls = []

    def fake_download(code, seasons, target_matches=0, min_start_year=0, max_workers=0):
        calls.append((code, target_matches))
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2026-01-01"]),
            "HomeTeam": [f"{code} Home"],
            "AwayTeam": [f"{code} Away"],
            "FTHG": [1], "FTAG": [0], "FTR": ["H"],
            "League": [code], "Season": ["2526"],
        })
        return frame, ["2526"], []

    monkeypatch.setattr(data_mod, "download_history_target", fake_download)
    frames, _, _ = data_mod.download_independent_league_histories(
        ["E0", "SP1", "D1", "I1", "F1"], ["2526"], target_matches_per_league=5000,
        league_workers=1, season_workers=1,
    )
    assert set(frames) == {"E0", "SP1", "D1", "I1", "F1"}
    assert calls == [("E0", 5000), ("SP1", 5000), ("D1", 5000), ("I1", 5000), ("F1", 5000)]


class DummyEngine:
    def __init__(self, league, teams):
        self.team_leagues_ = {team: [league] for team in teams}
        self.metrics = {"league": league}
        self.league = league

    def predict(self, fixtures, advanced_data=None):
        out = fixtures[["Date", "League", "HomeTeam", "AwayTeam"]].copy()
        out["Prediction"] = self.league
        return out


def test_bundle_routes_each_fixture_to_its_own_model():
    bundle = LeagueModelBundle(engines={
        "E0": DummyEngine("E0", ["Arsenal", "Chelsea"]),
        "SP1": DummyEngine("SP1", ["Barcelona", "Real Madrid"]),
    })
    fixtures = pd.DataFrame([
        {"Date": "20/09/2026", "League": "SP1", "HomeTeam": "Barcelona", "AwayTeam": "Real Madrid"},
        {"Date": "21/09/2026", "League": "E0", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"},
    ])
    pred = bundle.predict(fixtures)
    assert pred["Prediction"].tolist() == ["SP1", "E0"]
    assert pred["League"].tolist() == ["SP1", "E0"]
