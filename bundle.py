from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .advanced import AdvancedDataBundle
from .data import clean_fixtures, resolve_team_name
from .engine import SoccerPredictionEngine


@dataclass
class LeagueModelBundle:
    """A collection of fully independent league-specific prediction engines."""

    engines: dict[str, SoccerPredictionEngine] = field(default_factory=dict)
    target_matches_per_league: int = 5000
    version: str = "11"

    @property
    def trained_leagues(self) -> list[str]:
        return list(self.engines.keys())

    @property
    def metrics(self) -> dict[str, dict]:
        return {code: engine.metrics for code, engine in self.engines.items()}

    def _team_league_candidates(self, home: str, away: str) -> list[str]:
        found = []
        for code, engine in self.engines.items():
            team_map = getattr(engine, "team_leagues_", {}) or {}
            if code in team_map.get(str(home), []) and code in team_map.get(str(away), []):
                found.append(code)
        return found

    def prepare_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        out = clean_fixtures(fixtures).copy()
        if "League" not in out.columns:
            out["League"] = pd.NA
        leagues = []
        allowed = set(self.trained_leagues)
        for idx, row in out.iterrows():
            raw = row.get("League")
            code = "" if pd.isna(raw) else str(raw).strip().upper()
            if code in allowed:
                engine = self.engines[code]
                candidate_teams = list((getattr(engine, "team_leagues_", {}) or {}).keys())
                out.at[idx, "HomeTeam"] = resolve_team_name(row["HomeTeam"], candidate_teams)
                out.at[idx, "AwayTeam"] = resolve_team_name(row["AwayTeam"], candidate_teams)
                row = out.loc[idx]
            if not code or code == "NAN":
                candidates = self._team_league_candidates(row["HomeTeam"], row["AwayTeam"])
                if len(candidates) != 1:
                    raise ValueError(
                        f"Cannot uniquely infer league for {row['HomeTeam']} vs {row['AwayTeam']}. "
                        f"Add a League column using one of: {', '.join(self.trained_leagues)}"
                    )
                code = candidates[0]
            if code not in allowed:
                raise ValueError(
                    f"Fixture {row['HomeTeam']} vs {row['AwayTeam']} belongs to {code!r}, "
                    f"but this model bundle contains only: {', '.join(self.trained_leagues)}"
                )
            leagues.append(code)
        out["League"] = leagues
        return out


    @property
    def state_asof_dates(self) -> dict[str, object]:
        return {code: engine.state_asof_date for code, engine in self.engines.items()}

    def refresh_completed_matches(self, matches: pd.DataFrame) -> dict[str, int]:
        """Route fresh completed results to the corresponding league state."""
        if matches is None or len(matches) == 0:
            return {code: 0 for code in self.trained_leagues}
        out = matches.copy()
        if "League" not in out.columns and "Div" in out.columns:
            out["League"] = out["Div"].astype(str).str.upper()
        counts = {}
        for code, engine in self.engines.items():
            part = out[out["League"].astype(str).str.upper() == code].copy() if "League" in out.columns else pd.DataFrame()
            counts[code] = engine.refresh_completed_matches(part) if len(part) else 0
        return counts

    def refresh_competitive_matches(self, matches: pd.DataFrame) -> dict[str, int]:
        """Refresh all-competition recent form across the independent engines."""
        if matches is None or len(matches) == 0:
            return {code: 0 for code in self.trained_leagues}
        all_teams = []
        owner: dict[str, str] = {}
        opponent_elos: dict[str, float] = {}
        for code, engine in self.engines.items():
            for team in (getattr(engine, "team_leagues_", {}) or {}).keys():
                all_teams.append(team)
                owner.setdefault(str(team), code)
                try:
                    opponent_elos[str(team)] = float(engine.builder.teams[(code, str(team))].elo)
                except Exception:
                    pass
        all_teams = list(dict.fromkeys(all_teams))
        resolved = matches.copy()
        for idx, row in resolved.iterrows():
            resolved.at[idx, "HomeTeam"] = resolve_team_name(row.get("HomeTeam"), all_teams)
            resolved.at[idx, "AwayTeam"] = resolve_team_name(row.get("AwayTeam"), all_teams)
        counts = {}
        for code, engine in self.engines.items():
            counts[code] = engine.refresh_competitive_matches(resolved, team_leagues=owner, opponent_elos=opponent_elos)
        return counts

    def fixture_feature_audit(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare_fixtures(fixtures).reset_index(drop=True)
        prepared["__input_order"] = range(len(prepared))
        chunks = []
        for code in self.trained_leagues:
            part = prepared[prepared["League"] == code].copy()
            if part.empty:
                continue
            order = part["__input_order"].to_numpy()
            audit = self.engines[code].fixture_feature_audit(part.drop(columns=["__input_order"]))
            audit["__input_order"] = order
            chunks.append(audit)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True, sort=False).sort_values("__input_order").drop(columns=["__input_order"]).reset_index(drop=True)

    def predict(self, fixtures: pd.DataFrame, advanced_data: AdvancedDataBundle | None = None) -> pd.DataFrame:
        prepared = self.prepare_fixtures(fixtures).reset_index(drop=True)
        prepared["__input_order"] = range(len(prepared))
        chunks = []
        for code in self.trained_leagues:
            part = prepared[prepared["League"] == code].copy()
            if part.empty:
                continue
            order = part["__input_order"].to_numpy()
            pred = self.engines[code].predict(part.drop(columns=["__input_order"]), advanced_data=advanced_data)
            pred["__input_order"] = order
            chunks.append(pred)
        if not chunks:
            return pd.DataFrame()
        return (
            pd.concat(chunks, ignore_index=True, sort=False)
            .sort_values("__input_order")
            .drop(columns=["__input_order"])
            .reset_index(drop=True)
        )

    @property
    def backtest_ledgers(self) -> dict[str, pd.DataFrame]:
        out = {}
        for code, engine in self.engines.items():
            ledger = getattr(engine.model, "backtest_ledger_", None)
            if ledger is not None:
                out[code] = ledger.copy()
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "LeagueModelBundle":
        obj = joblib.load(path)
        if not isinstance(obj, LeagueModelBundle):
            raise TypeError("Saved artifact is not a LeagueModelBundle")
        return obj


def load_model_artifact(path: str | Path):
    """Load v9/v8 bundles and retain compatibility with older single-league engines."""
    obj = joblib.load(path)
    if isinstance(obj, (LeagueModelBundle, SoccerPredictionEngine)):
        return obj
    raise TypeError(f"Unsupported model artifact type: {type(obj).__name__}")
