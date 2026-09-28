from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import soccer_predictor.data as data_mod


def _season_frame(season: str, n: int = 380):
    start = data_mod._season_start_year(season)
    teams = [f"T{i}" for i in range(20)]
    rows = []
    d = pd.Timestamp(f"{start}-08-01")
    for i in range(n):
        h = teams[i % 20]
        a = teams[(i * 7 + 3) % 20]
        if h == a:
            a = teams[(i * 7 + 4) % 20]
        rows.append({"Date": d + pd.Timedelta(days=i), "HomeTeam": h, "AwayTeam": a, "FTHG": i % 4, "FTAG": (i + 1) % 3, "League": "E0", "Season": season})
    return pd.DataFrame(rows)


def test_download_history_target_expands_backwards(monkeypatch):
    calls = []
    def fake_download(league, season):
        calls.append(season)
        return _season_frame(season)
    monkeypatch.setattr(data_mod, "download_season", fake_download)
    frame, used, errors = data_mod.download_history_target("E0", ["2223", "2324"], target_matches=5000, min_start_year=2000)
    assert len(frame) >= 5000
    assert len(used) >= 14
    assert "2122" in used
    assert not errors
    assert frame.Date.is_monotonic_increasing
