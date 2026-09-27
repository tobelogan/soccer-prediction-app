from pathlib import Path
import pandas as pd

from soccer_predictor.autosource import DerivedPlayerMovementSource, OpenMeteoHistoricalWeatherSource, NOMINATIM_SEARCH, OPEN_METEO_ARCHIVE


class FakeHTTP:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, url, params, suffix):
        return self.root / ("cached" + suffix)
    def get_json(self, url, params=None, ttl_hours=None):
        if url == NOMINATIM_SEARCH:
            return [{"lat": "51.5", "lon": "-0.1"}]
        if url == OPEN_METEO_ARCHIVE:
            return {"daily": {"time": ["2024-01-01", "2024-01-08"], "temperature_2m_mean": [8.0, 9.0], "precipitation_sum": [1.2, 0.0], "wind_speed_10m_max": [18.0, 12.0]}}
        raise AssertionError(url)


def test_derived_player_change_writes_transfer_rows(tmp_path):
    pd.DataFrame([
        {"Date": "2024-01-01", "Team": "Alpha", "Player": "Test Player", "Minutes": 90},
        {"Date": "2024-02-01", "Team": "Beta", "Player": "Test Player", "Minutes": 90},
    ]).to_csv(tmp_path / "player_stats.csv", index=False)
    report = DerivedPlayerMovementSource().sync(tmp_path)
    out = pd.read_csv(tmp_path / "transfers.csv")
    assert report.rows_written["transfers"] == 2
    assert set(out["Direction"]) == {"IN", "OUT"}


def test_open_meteo_backfill_writes_locations_and_weather(tmp_path):
    matches = pd.DataFrame([
        {"Date": "2024-01-01", "HomeTeam": "Alpha", "AwayTeam": "Beta"},
        {"Date": "2024-01-08", "HomeTeam": "Alpha", "AwayTeam": "Gamma"},
    ])
    src = OpenMeteoHistoricalWeatherSource(FakeHTTP(tmp_path / "cache"), geocode_delay_seconds=0)
    report = src.sync(matches, tmp_path)
    assert report.rows_written["team_locations"] == 1
    assert report.rows_written["weather"] == 2
    weather = pd.read_csv(tmp_path / "weather.csv")
    assert weather["TempC"].tolist() == [8.0, 9.0]
