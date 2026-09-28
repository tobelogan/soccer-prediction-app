from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from soccer_predictor.advanced import AdvancedDataBundle, AdvancedFeatureStore
from soccer_predictor.data import download_recent_cross_competition_matches
from soccer_predictor.engine import SoccerPredictionEngine


def _history(n=220):
    rows=[]
    teams=["Alpha","Beta","Gamma","Delta"]
    start=pd.Timestamp("2025-01-01")
    for i in range(n):
        h=teams[i%4]; a=teams[(i+1)%4]
        rows.append({
            "League":"E0","Date":start+pd.Timedelta(days=i),"HomeTeam":h,"AwayTeam":a,
            "FTHG":1+(i%3==0),"FTAG":int(i%4==0),"HS":12,"AS":10,"HST":5,"AST":4,"HC":5,"AC":4,
        })
    return pd.DataFrame(rows)


def test_cross_competition_result_changes_allcomp_form_without_changing_league_form():
    hist=_history()
    eng=SoccerPredictionEngine(backend="sklearn", accelerator="cpu").fit(hist)
    future=hist["Date"].max()+pd.Timedelta(days=2)
    fixture=pd.DataFrame([{"League":"E0","Date":future,"HomeTeam":"Alpha","AwayTeam":"Beta"}])
    before=eng.fixture_feature_audit(fixture).iloc[0]
    cross=pd.DataFrame([{
        "Date":future-pd.Timedelta(days=1),"HomeTeam":"Alpha","AwayTeam":"Outside Club",
        "FTHG":5,"FTAG":0,"Competition":"UCL","CompetitionWeight":1.0,
    }])
    eng.refresh_competitive_matches(cross, opponent_elos={"Outside Club":1900.0})
    after=eng.fixture_feature_audit(fixture).iloc[0]
    assert after["home_gf_5"] == before["home_gf_5"]
    assert after["home_allcomp_gf_5"] > before["home_allcomp_gf_5"]
    assert after["home_opp_adj_gf_5"] > after["home_allcomp_gf_5"]


def test_lineup_and_unavailable_attack_strength_features():
    date=pd.Timestamp("2026-09-30")
    ps=[]
    for i in range(12):
        for j in range(4):
            ps.append({
                "Date":date-pd.Timedelta(days=7*(j+1)),"Team":"Alpha","Player":f"P{i}",
                "Minutes":90,"xG":0.55 if i==0 else 0.08,"xA":0.25 if i==0 else 0.05,
                "Role":"GK" if i==11 else "FW","PSxG":0.3 if i==11 else np.nan,"GoalsAllowed":0.2 if i==11 else np.nan,
            })
    lineups=pd.DataFrame([{
        "Date":date,"HomeTeam":"Alpha","AwayTeam":"Beta","Team":"Alpha","Player":f"P{i}",
        "Confirmed":True,"IsStarter":True,"Role":"GK" if i==11 else "FW"
    } for i in range(1,12)])
    availability=pd.DataFrame([{
        "Date":date,"HomeTeam":"Alpha","AwayTeam":"Beta","Team":"Alpha","Player":"P0",
        "Status":"out","Reason":"injury","SourceConfidence":1.0,"ExpectedMinutes":85,
    }])
    store=AdvancedFeatureStore(AdvancedDataBundle(player_stats=pd.DataFrame(ps), lineups=lineups, availability=availability))
    f=store.features_for({"Date":date,"HomeTeam":"Alpha","AwayTeam":"Beta"})
    assert f["home_xi_completeness"] == 1.0
    assert np.isfinite(f["home_xi_attack_delta_pct"])
    assert f["home_unavailable_attack_share"] > 0.0
    assert f["home_unavailable_xg90"] > 0.0


def test_cross_competition_espn_parser(monkeypatch):
    payload={"events":[{
        "id":"1","date":"2026-09-20T19:00:00Z","status":{"type":{"state":"post","name":"STATUS_FINAL"}},
        "competitions":[{"competitors":[
            {"homeAway":"home","score":"4","team":{"displayName":"Alpha"}},
            {"homeAway":"away","score":"1","team":{"displayName":"Beta"}},
        ]}]
    }]}
    class Resp:
        def raise_for_status(self): pass
        def json(self): return payload
    monkeypatch.setattr("soccer_predictor.data.requests.get", lambda *a, **k: Resp())
    out=download_recent_cross_competition_matches(now="2026-09-21", lookback_days=10)
    assert len(out) >= 1
    assert {"Competition","CompetitionWeight","FTHG","FTAG"}.issubset(out.columns)
    assert (out["FTHG"] == 4).any()
