import numpy as np
import pandas as pd

from soccer_predictor.features import FeatureBuilder
from soccer_predictor.engine import SoccerPredictionEngine
from soccer_predictor.model import MatchModel


def _row(date, home, away, hg, ag, league="SP1"):
    return {
        "Date": pd.Timestamp(date), "League": league,
        "HomeTeam": home, "AwayTeam": away,
        "FTHG": hg, "FTAG": ag,
        "HS": 15, "AS": 9, "HST": 8, "AST": 3, "HC": 6, "AC": 4,
    }


def test_last_five_scoring_surge_is_visible_in_fixture_features():
    b = FeatureBuilder()
    rows = []
    # Strong has a five-game 5-goal scoring surge; Weak has a low-scoring run.
    for i in range(5):
        rows.append(_row(f"2026-09-{1+i:02d}", "Strong", f"Opp{i}", 5, 1))
        rows.append(_row(f"2026-09-{1+i:02d}", f"Other{i}", "Weak", 1, 0))
    hist = pd.DataFrame(rows).sort_values("Date")
    b.build_training_frame(hist)
    fx = pd.DataFrame([{"Date": "2026-09-10", "League": "SP1", "HomeTeam": "Strong", "AwayTeam": "Weak"}])
    feat = b.build_fixture_frame(fx).iloc[0]
    assert feat["home_gf_5"] >= 4.9
    assert feat["away_gf_5"] <= 0.1
    assert feat["home_goal_diff_5"] > feat["away_goal_diff_5"]
    assert feat["form_home_xg"] > feat["form_away_xg"]


def test_engine_refresh_advances_saved_form_state():
    teams = ["A", "B", "C", "D", "E", "F"]
    rows = []
    date = pd.Timestamp("2025-01-01")
    for i in range(220):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        rows.append(_row(date + pd.Timedelta(days=i), home, away, 1 + (i % 2), i % 2, league="E0"))
    eng = SoccerPredictionEngine(backend="sklearn", accelerator="cpu").fit(pd.DataFrame(rows))
    cutoff = eng.state_asof_date
    fresh = pd.DataFrame([_row(cutoff + pd.Timedelta(days=1), "A", "B", 6, 0, league="E0")])
    assert eng.refresh_completed_matches(fresh) == 1
    assert eng.state_asof_date > cutoff
    audit = eng.fixture_feature_audit(pd.DataFrame([{"Date": cutoff + pd.Timedelta(days=2), "League": "E0", "HomeTeam": "A", "AwayTeam": "B"}]))
    assert audit.loc[0, "home_gf_5"] > 1.0
    assert audit.loc[0, "away_ga_5"] > 0.0


def test_goal_form_blend_can_prefer_informative_recent_form():
    actual = np.array([4, 5, 3, 4, 5, 0, 1, 0, 1, 0] * 5, dtype=float)
    base = np.full_like(actual, 1.4)
    form = np.clip(actual * 0.9 + 0.2, 0.1, None)
    w, _ = MatchModel._best_goal_form_blend(actual, base, form)
    assert w > 0.0
