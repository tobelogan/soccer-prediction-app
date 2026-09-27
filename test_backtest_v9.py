import numpy as np
import pandas as pd

from soccer_predictor.features import FeatureBuilder
from soccer_predictor.model import MatchModel, POST_ENTRY_MARKET_FEATURES, FEATURE_COLUMNS


def synthetic_matches(n=360):
    rng = np.random.default_rng(42)
    teams = [f"Team{i}" for i in range(12)]
    rows = []
    d = pd.Timestamp("2020-08-01")
    for i in range(n):
        h = teams[i % len(teams)]
        a = teams[(i * 5 + 3) % len(teams)]
        if h == a:
            a = teams[(teams.index(a) + 1) % len(teams)]
        hg = int(rng.poisson(1.55))
        ag = int(rng.poisson(1.15))
        # Plausible opening and closing prices. Close deliberately differs so CLV is testable.
        open_odds = [2.15, 3.35, 3.55]
        close_odds = [2.05, 3.30, 3.75]
        rows.append({
            "Date": d + pd.Timedelta(days=i), "League": "E0", "HomeTeam": h, "AwayTeam": a,
            "FTHG": hg, "FTAG": ag, "HS": 12 + rng.normal(), "AS": 10 + rng.normal(),
            "HST": 4.5 + rng.normal(), "AST": 3.8 + rng.normal(), "HC": 5, "AC": 4,
            "B365H": open_odds[0], "B365D": open_odds[1], "B365A": open_odds[2],
            "B365CH": close_odds[0], "B365CD": close_odds[1], "B365CA": close_odds[2],
        })
    return pd.DataFrame(rows)


def test_closing_market_fields_excluded_from_ml_features():
    assert POST_ENTRY_MARKET_FEATURES.isdisjoint(FEATURE_COLUMNS)


def test_training_creates_untouched_betting_ledger():
    frame = FeatureBuilder().build_training_frame(synthetic_matches())
    model = MatchModel(backend="sklearn", accelerator="cpu").fit(frame, refit_full=False)
    bt = model.metrics_["betting_backtest"]
    assert "bets" in bt and "roi" in bt and "max_drawdown_pct" in bt and "mean_clv" in bt
    assert model.metrics_["bet_edge_threshold"] >= 0.02
    assert set(["OpenOdds", "CloseOdds", "CLV", "ProfitUnits"]).issubset(model.backtest_ledger_.columns) or model.backtest_ledger_.empty
