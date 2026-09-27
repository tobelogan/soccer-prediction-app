from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from soccer_predictor.advanced import AdvancedDataBundle
from soccer_predictor.engine import SoccerPredictionEngine


def make_matches(n=220, seed=12):
    rng = np.random.default_rng(seed)
    teams = [f"Club {i}" for i in range(10)]
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hs, ass = rng.integers(7, 19), rng.integers(6, 17)
        hg = rng.poisson(max(0.25, 0.8 + hs / 18))
        ag = rng.poisson(max(0.20, 0.65 + ass / 20))
        rows.append({
            "MatchID": f"m{i}", "Date": start + pd.Timedelta(days=i),
            "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag,
            "HS": hs, "AS": ass, "HST": max(1, hs // 3), "AST": max(1, ass // 3),
            "HC": rng.integers(1, 9), "AC": rng.integers(1, 9),
            "AvgH": 2.0 + rng.random(), "AvgD": 3.0 + rng.random(), "AvgA": 2.2 + rng.random(),
        })
    return pd.DataFrame(rows)


def make_advanced(matches):
    rng = np.random.default_rng(33)
    teams = sorted(set(matches.HomeTeam).union(matches.AwayTeam))
    events, players, lineups, availability, schedule, weather, markets, tactics = [], [], [], [], [], [], [], []
    for _, r in matches.iterrows():
        date, h, a, mid = r.Date, r.HomeTeam, r.AwayTeam, r.MatchID
        for team, opp in [(h, a), (a, h)]:
            schedule.append({"Date": date, "Team": team, "Competition": "League", "IsContinental": False})
            tactics.append({"Date": date, "Team": team, "Possession": 45 + rng.random()*10, "PPDA": 8+rng.random()*5, "FieldTilt": 45+rng.random()*10, "xT": 1+rng.random(), "xTAgainst": 1+rng.random()})
            for p in range(3):
                player = f"{team}-P{p}"
                xg = float(rng.random() * 0.35)
                xa = float(rng.random() * 0.20)
                events.append({"MatchID": mid, "Date": date, "Team": team, "Opponent": opp, "Player": player, "xG": xg, "npxG": xg*0.95, "BigChance": int(xg > 0.25)})
                players.append({"Date": date, "MatchID": mid, "Team": team, "Player": player, "Minutes": 90, "xG": xg, "xA": xa, "PSxG": 0.0, "GoalsAllowed": 0.0, "Role": "FW"})
                lineups.append({"Date": date, "HomeTeam": h, "AwayTeam": a, "Team": team, "Player": player, "IsStarter": True, "Confirmed": True, "Role": "FW"})
            gk = f"{team}-GK"
            players.append({"Date": date, "MatchID": mid, "Team": team, "Player": gk, "Minutes": 90, "xG": 0.0, "xA": 0.0, "PSxG": 1.3, "GoalsAllowed": 1.0, "Role": "GK"})
            lineups.append({"Date": date, "HomeTeam": h, "AwayTeam": a, "Team": team, "Player": gk, "IsStarter": True, "Confirmed": True, "Role": "GK"})
        if rng.random() < 0.35:
            availability.append({"Date": date, "HomeTeam": h, "AwayTeam": a, "Team": h, "Player": f"{h}-P2", "Status": "out", "Reason": "injury", "ExpectedMinutes": 75})
        weather.append({"Date": date, "HomeTeam": h, "AwayTeam": a, "TempC": 12+rng.random()*14, "WindKph": rng.random()*25, "PrecipMm": rng.random()*4, "HumidityPct": 45+rng.random()*40})
        markets.append({
            "Date": date, "HomeTeam": h, "AwayTeam": a,
            "OpenH": 2.4, "OpenD": 3.2, "OpenA": 2.9,
            "CloseH": 2.3, "CloseD": 3.25, "CloseA": 3.0,
            "SharpH": 2.32, "SharpD": 3.28, "SharpA": 2.98,
            "ExchangeH": 2.34, "ExchangeD": 3.30, "ExchangeA": 3.02,
            "ExchangeLiquidity": 50000 + rng.random()*200000,
        })
    managers = pd.DataFrame([{"Date": matches.Date.min() - pd.Timedelta(days=100), "Team": t, "Manager": f"Coach-{t}"} for t in teams])
    locations = pd.DataFrame([{"Team": t, "Lat": 50+i*0.12, "Lon": -1+i*0.11} for i, t in enumerate(teams)])
    transfers = pd.DataFrame([{"Date": matches.Date.min() + pd.Timedelta(days=20), "Team": teams[0], "Player": f"{teams[0]}-P0", "Direction": "IN", "PlayerXG90": 0.2, "PlayerXA90": 0.1, "PriorMinutes": 1000}])
    return AdvancedDataBundle(
        events=pd.DataFrame(events), player_stats=pd.DataFrame(players), lineups=pd.DataFrame(lineups),
        availability=pd.DataFrame(availability), transfers=transfers, managers=managers,
        team_locations=locations, schedule=pd.DataFrame(schedule), weather=pd.DataFrame(weather),
        markets=pd.DataFrame(markets), tactics=pd.DataFrame(tactics),
    )


def test_advanced_pipeline():
    matches = make_matches()
    rich = make_advanced(matches)
    eng = SoccerPredictionEngine().fit(matches, advanced_data=rich)
    assert eng.metrics["advanced_coverage"]["event_data_available"] > 0.5
    assert eng.metrics["advanced_coverage"]["lineup_data_available"] > 0.5
    assert -0.2 <= eng.metrics["dixon_coles_rho"] <= 0.2

    fixture_date = matches.Date.max() + pd.Timedelta(days=3)
    h, a = matches.HomeTeam.iloc[-1], matches.AwayTeam.iloc[-1]
    fx = pd.DataFrame([{"Date": fixture_date, "HomeTeam": h, "AwayTeam": a, "AvgH": 2.2, "AvgD": 3.3, "AvgA": 3.1}])
    pred = eng.predict(fx)
    assert len(pred) == 1
    assert pred.loc[0, "RichDataScore"] > 0
    assert abs(pred.loc[0, ["P_Home", "P_Draw", "P_Away"]].sum() - 1.0) < 1e-6
