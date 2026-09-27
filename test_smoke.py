from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
from soccer_predictor.engine import SoccerPredictionEngine


def make_data(n=220, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"Team {i}" for i in range(12)]
    rows = []
    date = pd.Timestamp("2023-08-01")
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        he = rng.normal(1.55, 0.35)
        ae = rng.normal(1.20, 0.30)
        hg, ag = rng.poisson(max(0.3, he)), rng.poisson(max(0.25, ae))
        rows.append({
            "Date": date + pd.Timedelta(days=i), "HomeTeam": h, "AwayTeam": a,
            "FTHG": hg, "FTAG": ag, "HS": rng.integers(6, 20), "AS": rng.integers(5, 18),
            "HST": rng.integers(1, 9), "AST": rng.integers(1, 8), "HC": rng.integers(1, 10), "AC": rng.integers(1, 10),
        })
    return pd.DataFrame(rows)


def test_end_to_end():
    eng = SoccerPredictionEngine().fit(make_data())
    fx = pd.DataFrame([{"Date": "2026-09-20", "HomeTeam": "Team 1", "AwayTeam": "Team 2"}])
    pred = eng.predict(fx)
    assert len(pred) == 1
    assert abs(pred.loc[0, ["P_Home", "P_Draw", "P_Away"]].sum() - 1) < 1e-6
    assert pred.loc[0, "Home_xG"] > 0
    assert pred.loc[0, "Away_xG"] > 0
