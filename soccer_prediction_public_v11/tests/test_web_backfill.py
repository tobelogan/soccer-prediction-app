from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
from soccer_predictor.autosource import FootballDataBackfillSource
from soccer_predictor.data import load_advanced_bundle
from soccer_predictor.advanced import AdvancedFeatureStore


def test_football_data_web_backfills_schedule_and_market(tmp_path):
    date = pd.Timestamp("2025-01-10")
    base = pd.DataFrame([{ 
        "Date": date, "HomeTeam": "A", "AwayTeam": "B", "League": "E0",
        "FTHG": 2, "FTAG": 1,
        "B365H": 2.0, "B365D": 3.5, "B365A": 4.0,
        "B365CH": 1.9, "B365CD": 3.6, "B365CA": 4.2,
        "PSCH": 1.95, "PSCD": 3.55, "PSCA": 4.1,
    }])
    report = FootballDataBackfillSource().sync(base, tmp_path)
    assert report.rows_written["schedule"] == 2
    assert report.rows_written["markets"] == 1
    bundle = load_advanced_bundle(tmp_path)
    store = AdvancedFeatureStore(bundle)
    f = store.features_for({"Date": date, "HomeTeam": "A", "AwayTeam": "B"})
    assert f["market_data_available"] == 1.0
    assert f["schedule_data_available"] == 1.0
    assert f["sharp_home"] > 0
    assert abs(f["open_home"] + f["open_draw"] + f["open_away"] - 1.0) < 1e-9
