import pandas as pd

from soccer_predictor import data


def test_download_upcoming_fixtures_filters_league_date_and_limit(monkeypatch, tmp_path):
    raw = pd.DataFrame([
        {"Div": "E0", "Date": "15/09/2026", "HomeTeam": "Old", "AwayTeam": "Game"},
        {"Div": "E0", "Date": "18/09/2026", "HomeTeam": "Brentford", "AwayTeam": "Chelsea"},
        {"Div": "SP1", "Date": "18/09/2026", "HomeTeam": "Barcelona", "AwayTeam": "Valencia"},
        {"Div": "E0", "Date": "19/09/2026", "HomeTeam": "Brighton", "AwayTeam": "Arsenal"},
        {"Div": "E0", "Date": "30/09/2026", "HomeTeam": "Too", "AwayTeam": "Late"},
    ])
    monkeypatch.setattr(data, "_get_csv", lambda url: raw.copy())

    got = data.download_upcoming_fixtures("E0", days=10, limit=2, now="2026-09-16", cache_path=tmp_path / "fixtures.csv")
    assert list(got["HomeTeam"]) == ["Brentford", "Brighton"]
    assert set(got["Div"]) == {"E0"}
    assert got["Date"].min() >= pd.Timestamp("2026-09-16")
    assert got["Date"].max() <= pd.Timestamp("2026-09-26")


def test_clean_fixtures_accepts_provider_alias_columns():
    raw = pd.DataFrame([
        {"leagueCode": "E0", "dateEvent": "2026-09-18", "strHomeTeam": "Brentford", "strAwayTeam": "Chelsea"}
    ])
    got = data.clean_fixtures(raw)
    assert got.loc[0, "HomeTeam"] == "Brentford"
    assert got.loc[0, "AwayTeam"] == "Chelsea"
    assert got.loc[0, "League"] == "E0"


def test_multi_fixture_download_falls_back_when_football_data_is_bad(monkeypatch, tmp_path):
    # Simulates the current failure mode: the Football-Data URL returns a web page
    # that pandas can technically parse but which has no fixture schema.
    bad = pd.DataFrame({"<html><body>download page</body></html>": ["x"]})
    backup = pd.DataFrame([
        {"Div": "E0", "League": "E0", "Date": "2026-09-18", "HomeTeam": "Brentford", "AwayTeam": "Chelsea", "FixtureSource": "TheSportsDB"}
    ])
    monkeypatch.setattr(data, "_get_csv", lambda url: bad.copy())
    monkeypatch.setattr(data, "_download_espn_fixtures", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("espn unavailable")))
    monkeypatch.setattr(data, "_download_thesportsdb_fixtures", lambda leagues: backup.copy())

    got = data.download_upcoming_fixtures_multi(["E0"], days=3, now="2026-09-17", cache_path=tmp_path / "fixtures.csv")
    assert len(got) == 1
    assert got.loc[0, "HomeTeam"] == "Brentford"
    assert got.loc[0, "FixtureSource"] == "TheSportsDB"


def test_multi_fixture_download_uses_espn_full_window(monkeypatch, tmp_path):
    bad = pd.DataFrame({"<html>download page</html>": ["x"]})
    espn = pd.DataFrame([
        {"Div": "E0", "League": "E0", "Date": "2026-09-18", "HomeTeam": "Brentford", "AwayTeam": "Chelsea", "FixtureSource": "ESPN"},
        {"Div": "E0", "League": "E0", "Date": "2026-09-19", "HomeTeam": "Brighton & Hove Albion", "AwayTeam": "Arsenal", "FixtureSource": "ESPN"},
        {"Div": "E0", "League": "E0", "Date": "2026-09-20", "HomeTeam": "Fulham", "AwayTeam": "Manchester United", "FixtureSource": "ESPN"},
        {"Div": "SP1", "League": "SP1", "Date": "2026-09-20", "HomeTeam": "Barcelona", "AwayTeam": "Valencia", "FixtureSource": "ESPN"},
    ])
    monkeypatch.setattr(data, "_get_csv", lambda url: bad.copy())
    monkeypatch.setattr(data, "_download_espn_fixtures", lambda leagues, date_from, date_to: espn.copy())
    monkeypatch.setattr(data, "_download_thesportsdb_fixtures", lambda leagues: (_ for _ in ()).throw(AssertionError("should not reach TheSportsDB")))

    got = data.download_upcoming_fixtures_multi(["E0", "SP1"], days=10, now="2026-09-17", cache_path=tmp_path / "fixtures.csv")
    assert len(got) == 4
    assert set(got["League"]) == {"E0", "SP1"}
    assert set(got["FixtureSource"]) == {"ESPN"}


def test_espn_parser_returns_all_events(monkeypatch):
    payload = {
        "events": [
            {
                "id": "1", "date": "2026-09-19T14:00Z",
                "status": {"type": {"name": "STATUS_SCHEDULED"}},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "team": {"displayName": "Brighton & Hove Albion"}},
                    {"homeAway": "away", "team": {"displayName": "Arsenal"}},
                ]}],
            },
            {
                "id": "2", "date": "2026-09-20T15:30Z",
                "status": {"type": {"name": "STATUS_SCHEDULED"}},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "team": {"displayName": "Fulham"}},
                    {"homeAway": "away", "team": {"displayName": "Manchester United"}},
                ]}],
            },
        ]
    }

    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return payload

    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response())
    got = data._download_espn_fixtures(["E0"], "2026-09-17", "2026-09-27")
    assert len(got) == 2
    assert list(got["HomeTeam"]) == ["Brighton & Hove Albion", "Fulham"]
    assert set(got["FixtureSource"]) == {"ESPN"}


def test_fixture_download_parser_returns_full_slate(monkeypatch):
    sample = pd.DataFrame([
        {"Round Number": 5, "Date": "18/09/2026 20:00", "Location": "A", "Home Team": "Brentford", "Away Team": "Chelsea", "Result": "-"},
        {"Round Number": 5, "Date": "19/09/2026 12:30", "Location": "B", "Home Team": "Spurs", "Away Team": "Aston Villa", "Result": "-"},
        {"Round Number": 5, "Date": "19/09/2026 15:00", "Location": "C", "Home Team": "Brighton", "Away Team": "Arsenal", "Result": "-"},
        {"Round Number": 5, "Date": "20/09/2026 16:30", "Location": "D", "Home Team": "Fulham", "Away Team": "Man Utd", "Result": "-"},
    ])
    monkeypatch.setattr(data, "_get_csv", lambda url: sample.copy())
    got = data._download_fixturedownload_fixtures(["E0"], now="2026-09-18")
    assert len(got) == 4
    assert set(got["League"]) == {"E0"}
    assert set(got["FixtureSource"]) == {"FixtureDownload"}
    assert got.loc[0, "HomeTeam"] == "Brentford"


def test_multi_fixture_download_prefers_fixture_download_before_limited_fallback(monkeypatch, tmp_path):
    bad = pd.DataFrame({"<html>download page</html>": ["x"]})
    full = pd.DataFrame([
        {"Div": "E0", "League": "E0", "Date": "2026-09-18", "HomeTeam": "Brentford", "AwayTeam": "Chelsea", "FixtureSource": "FixtureDownload"},
        {"Div": "E0", "League": "E0", "Date": "2026-09-19", "HomeTeam": "Tottenham", "AwayTeam": "Aston Villa", "FixtureSource": "FixtureDownload"},
        {"Div": "E0", "League": "E0", "Date": "2026-09-19", "HomeTeam": "Brighton", "AwayTeam": "Arsenal", "FixtureSource": "FixtureDownload"},
        {"Div": "SP1", "League": "SP1", "Date": "2026-09-20", "HomeTeam": "Barcelona", "AwayTeam": "Valencia", "FixtureSource": "FixtureDownload"},
        {"Div": "SP1", "League": "SP1", "Date": "2026-09-20", "HomeTeam": "Sevilla", "AwayTeam": "Getafe", "FixtureSource": "FixtureDownload"},
    ])

    def fake_get_csv(url, *args, **kwargs):
        if "football-data" in str(url):
            return bad.copy()
        raise RuntimeError("fixture direct download mocked separately")

    monkeypatch.setattr(data, "_get_csv", fake_get_csv)
    monkeypatch.setattr(data, "_download_fixturedownload_fixtures", lambda leagues, now=None: full.copy())
    monkeypatch.setattr(data, "_download_espn_fixtures", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not reach ESPN")))
    monkeypatch.setattr(data, "_download_thesportsdb_fixtures", lambda leagues: (_ for _ in ()).throw(AssertionError("should not reach TheSportsDB")))

    got = data.download_upcoming_fixtures_multi(["E0", "SP1"], days=10, now="2026-09-17", cache_path=tmp_path / "fixtures.csv")
    assert len(got) == 5
    assert got.groupby("League").size().to_dict() == {"E0": 3, "SP1": 2}
    assert set(got["FixtureSource"]) == {"FixtureDownload"}
