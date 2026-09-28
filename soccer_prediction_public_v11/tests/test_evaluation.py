from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from soccer_predictor.engine import SoccerPredictionEngine


def make_market_data(n=260, seed=44):
    rng = np.random.default_rng(seed)
    teams = [f"Side {i}" for i in range(12)]
    strengths = {t: rng.normal(0, 0.35) for t in teams}
    rows = []
    start = pd.Timestamp("2022-08-01")
    for i in range(n):
        home, away = rng.choice(teams, 2, replace=False)
        hx = np.exp(0.25 + strengths[home] - 0.45 * strengths[away])
        ax = np.exp(-0.05 + strengths[away] - 0.35 * strengths[home])
        hg, ag = rng.poisson(hx), rng.poisson(ax)
        # A deliberately sensible but noisy market proxy.
        diff = strengths[home] - strengths[away] + 0.18
        ph = 1 / (1 + np.exp(-1.25 * diff))
        pd_ = 0.24
        pa = max(0.05, 1 - ph - pd_)
        p = np.array([ph, pd_, pa], dtype=float)
        p = np.clip(p, 0.05, None); p /= p.sum()
        margin = 1.055
        odds = 1 / (p * margin)
        rows.append({
            "Date": start + pd.Timedelta(days=i), "HomeTeam": home, "AwayTeam": away,
            "FTHG": hg, "FTAG": ag, "HS": rng.integers(7, 20), "AS": rng.integers(6, 18),
            "HST": rng.integers(1, 9), "AST": rng.integers(1, 8), "HC": rng.integers(1, 10), "AC": rng.integers(1, 10),
            "AvgH": odds[0], "AvgD": odds[1], "AvgA": odds[2],
        })
    return pd.DataFrame(rows)


def test_three_block_evaluation_and_market_ensemble():
    data = make_market_data()
    eng = SoccerPredictionEngine().fit(data)
    m = eng.metrics
    assert m["train_matches"] + m["calibration_matches"] + m["test_matches"] == len(data)
    assert m["train_matches"] > m["calibration_matches"] >= 30
    assert m["test_matches"] >= 30
    assert 0.0 <= m["poisson_blend_weight"] <= 1.0
    assert 0.0 <= m["market_blend_weight"] <= 1.0
    assert 0.70 <= m["probability_temperature"] <= 1.80
    assert -0.151 <= m["dixon_coles_rho"] <= 0.151
    assert np.isfinite(m["log_loss"])
    assert m["market_baseline_matches"] == m["test_matches"]

    fx = data.tail(1)[["Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"]].copy()
    fx["Date"] = pd.Timestamp("2027-01-01")
    pred = eng.predict(fx)
    assert pred.loc[0, "MarketSource"] == "BASE"
    assert abs(pred.loc[0, ["P_Home", "P_Draw", "P_Away"]].sum() - 1) < 1e-8
