from __future__ import annotations

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from .advanced import AdvancedDataBundle, AdvancedFeatureStore
from .data import clean_fixtures, resolve_team_name, clean_matches
from .features import FeatureBuilder
from .model import MatchModel


def _merge_frames(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame | None:
    if a is None:
        return None if b is None else b.copy()
    if b is None:
        return a.copy()
    out = pd.concat([a, b], ignore_index=True, sort=False)
    return out.drop_duplicates().reset_index(drop=True)


def merge_bundles(base: AdvancedDataBundle | None, extra: AdvancedDataBundle | None) -> AdvancedDataBundle | None:
    if base is None and extra is None:
        return None
    base = base or AdvancedDataBundle()
    extra = extra or AdvancedDataBundle()
    kwargs = {}
    for name in AdvancedDataBundle.__dataclass_fields__:
        kwargs[name] = _merge_frames(getattr(base, name), getattr(extra, name))
    return AdvancedDataBundle(**kwargs)


class SoccerPredictionEngine:
    def __init__(self, rolling_window: int = 8, backend: str = "auto", accelerator: str = "auto"):
        self.rolling_window = rolling_window
        self.backend = backend
        self.accelerator = accelerator
        self.advanced_bundle_: AdvancedDataBundle | None = None
        self.builder = FeatureBuilder(rolling_window=rolling_window)
        self.model = MatchModel(backend=backend, accelerator=accelerator)
        self.training_frame_: pd.DataFrame | None = None
        self.trained_leagues_: list[str] = []
        self.team_leagues_: dict[str, list[str]] = {}
        self.state_asof_date_: pd.Timestamp | None = None
        self.recent_completed_log_: pd.DataFrame = pd.DataFrame()

    def fit(self, matches: pd.DataFrame, advanced_data: AdvancedDataBundle | None = None):
        matches = clean_matches(matches)
        if "League" not in matches.columns:
            matches = matches.copy()
            matches["League"] = "CUSTOM"
        else:
            matches = matches.copy()
            matches["League"] = matches["League"].astype(str).str.upper()

        self.trained_leagues_ = sorted(matches["League"].dropna().astype(str).str.upper().unique().tolist())
        mapping: dict[str, set[str]] = defaultdict(set)
        for _, row in matches[["League", "HomeTeam", "AwayTeam"]].iterrows():
            league = str(row["League"]).upper()
            mapping[str(row["HomeTeam"])].add(league)
            mapping[str(row["AwayTeam"])].add(league)
        self.team_leagues_ = {team: sorted(vals) for team, vals in mapping.items()}

        self.advanced_bundle_ = advanced_data
        store = AdvancedFeatureStore(advanced_data, rolling_matches=self.rolling_window) if advanced_data is not None else None
        self.builder = FeatureBuilder(rolling_window=self.rolling_window, advanced_store=store)
        frame = self.builder.build_training_frame(matches)
        self.training_frame_ = frame
        self.state_asof_date_ = pd.to_datetime(matches["Date"], errors="coerce").max()
        self.model.fit(frame)
        return self


    @property
    def state_asof_date(self):
        value = getattr(self, "state_asof_date_", None)
        if value is not None and not pd.isna(value):
            return pd.Timestamp(value)
        frame = getattr(self, "training_frame_", None)
        if frame is not None and len(frame) and "Date" in frame.columns:
            value = pd.to_datetime(frame["Date"], errors="coerce").max()
            if not pd.isna(value):
                return pd.Timestamp(value)
        return None

    def refresh_completed_matches(self, matches: pd.DataFrame) -> int:
        """Advance saved Elo/recent-form state without refitting the ML trees."""
        if matches is None or len(matches) == 0:
            return 0
        frame = clean_matches(matches).copy()
        scope = set(self._legacy_scope())
        if "League" not in frame.columns:
            if len(scope) != 1:
                raise ValueError("League is required when refreshing a multi-league engine")
            frame["League"] = next(iter(scope))
        frame["League"] = frame["League"].astype(str).str.upper()
        frame = frame[frame["League"].isin(scope)].copy()
        cutoff = self.state_asof_date
        if cutoff is not None:
            frame = frame[frame["Date"] > cutoff].copy()
        if frame.empty:
            return 0

        candidate_teams = list((getattr(self, "team_leagues_", {}) or {}).keys())
        for idx, row in frame.iterrows():
            frame.at[idx, "HomeTeam"] = resolve_team_name(row["HomeTeam"], candidate_teams)
            frame.at[idx, "AwayTeam"] = resolve_team_name(row["AwayTeam"], candidate_teams)
        # Do not inject truly unknown clubs into a trained league state. Newly
        # promoted teams present in the current training season are already in
        # team_leagues_; provider spelling differences are reconciled above.
        known = set(candidate_teams)
        frame = frame[frame["HomeTeam"].isin(known) & frame["AwayTeam"].isin(known)].copy()
        frame = frame.sort_values("Date").drop_duplicates(
            subset=["League", "Date", "HomeTeam", "AwayTeam"], keep="first"
        )
        if frame.empty:
            return 0
        count = self.builder.update_completed_matches(frame)
        if count:
            self.state_asof_date_ = pd.to_datetime(frame["Date"], errors="coerce").max()
            self.recent_completed_log_ = pd.concat([getattr(self, "recent_completed_log_", pd.DataFrame()), frame], ignore_index=True, sort=False)
            self.recent_completed_log_ = self.recent_completed_log_.drop_duplicates(
                subset=["League", "Date", "HomeTeam", "AwayTeam"], keep="last"
            ).sort_values("Date").tail(500)
        return int(count)

    def refresh_competitive_matches(self, matches: pd.DataFrame, *, team_leagues: dict[str, str] | None = None, opponent_elos: dict[str, float] | None = None) -> int:
        """Rebuild recent all-competition state from league + cup/Europe results.

        This does not refit the ML trees and does not change league Elo or league
        scoring priors. It only refreshes the all-competition recency features.
        """
        team_leagues = team_leagues or {t: leagues[0] for t, leagues in (getattr(self, "team_leagues_", {}) or {}).items() if leagues}
        candidate_teams = list(team_leagues)
        ext = pd.DataFrame() if matches is None else matches.copy()
        if len(ext):
            for idx, row in ext.iterrows():
                ext.at[idx, "HomeTeam"] = resolve_team_name(row.get("HomeTeam"), candidate_teams)
                ext.at[idx, "AwayTeam"] = resolve_team_name(row.get("AwayTeam"), candidate_teams)
            ext["Date"] = pd.to_datetime(ext["Date"], errors="coerce")
            if "CompetitionWeight" not in ext.columns:
                ext["CompetitionWeight"] = 0.85

        base = getattr(self, "training_frame_", None)
        league_rows = []
        if base is not None and len(base):
            tmp = base[[c for c in ["Date", "League", "HomeTeam", "AwayTeam", "home_goals", "away_goals"] if c in base.columns]].copy()
            if {"home_goals", "away_goals"}.issubset(tmp.columns):
                tmp = tmp.rename(columns={"home_goals": "FTHG", "away_goals": "FTAG"})
                tmp["Competition"] = "Domestic league"
                tmp["CompetitionWeight"] = 1.0
                league_rows.append(tmp)
        recent = getattr(self, "recent_completed_log_", pd.DataFrame())
        if recent is not None and len(recent):
            tmp = recent.copy()
            tmp["Competition"] = "Domestic league"
            tmp["CompetitionWeight"] = 1.0
            league_rows.append(tmp)
        frames = league_rows + ([ext] if len(ext) else [])
        if not frames:
            return 0
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined = combined.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        latest = combined["Date"].max()
        combined = combined[combined["Date"] >= latest - pd.Timedelta(days=120)].copy()
        combined = combined.sort_values(["Date", "Competition", "HomeTeam", "AwayTeam"]).drop_duplicates(
            subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last"
        )
        self.builder.reset_competitive_state()
        return int(self.builder.update_external_competitive_matches(combined, team_leagues=team_leagues, opponent_elos=opponent_elos))

    def fixture_feature_audit(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Expose the live form inputs used for a fixture prediction."""
        fixtures = self.prepare_fixtures(fixtures)
        feat = self.builder.build_fixture_frame(fixtures)
        cols = [
            "Date", "League", "HomeTeam", "AwayTeam", "home_elo", "away_elo",
            "home_gf_3", "away_gf_3", "home_gf_5", "away_gf_5",
            "home_ga_5", "away_ga_5", "home_ppg_5", "away_ppg_5",
            "home_goal_diff_5", "away_goal_diff_5",
            "home_attack_ewm", "away_attack_ewm", "home_defense_ewm", "away_defense_ewm",
            "home_allcomp_gf_5", "away_allcomp_gf_5", "home_allcomp_ga_5", "away_allcomp_ga_5",
            "home_allcomp_ppg_5", "away_allcomp_ppg_5",
            "home_opp_adj_gf_5", "away_opp_adj_gf_5", "home_opp_adj_ga_5", "away_opp_adj_ga_5",
            "home_allcomp_attack_ewm", "away_allcomp_attack_ewm",
            "home_allcomp_defense_ewm", "away_allcomp_defense_ewm",
            "home_xi_completeness", "away_xi_completeness",
            "home_xi_attack_delta_pct", "away_xi_attack_delta_pct",
            "home_unavailable_attack_share", "away_unavailable_attack_share",
            "home_gk_delta", "away_gk_delta",
            "form_home_xg", "form_away_xg", "league_home_goal_prior", "league_away_goal_prior",
        ]
        return feat[[c for c in cols if c in feat.columns]].copy()

    def _legacy_scope(self) -> list[str]:
        leagues = getattr(self, "trained_leagues_", None)
        if leagues:
            return [str(x).upper() for x in leagues]
        # Old v6 single-league artifacts did not persist league metadata. Keep
        # them usable as EPL artifacts, but v7 models should always be retrained.
        return ["E0"]

    @property
    def trained_leagues(self) -> list[str]:
        return self._legacy_scope()

    def prepare_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Normalize fixture league labels and enforce the model's league scope."""
        fixtures = clean_fixtures(fixtures)
        allowed = set(self._legacy_scope())
        team_map = getattr(self, "team_leagues_", {}) or {}
        out = fixtures.copy()

        if "League" not in out.columns:
            out["League"] = pd.NA

        inferred: list[str] = []
        for idx, row in out.iterrows():
            raw = row.get("League")
            league = "" if pd.isna(raw) else str(raw).strip().upper()
            if league in allowed:
                candidate_teams = list(team_map.keys())
                out.at[idx, "HomeTeam"] = resolve_team_name(row["HomeTeam"], candidate_teams)
                out.at[idx, "AwayTeam"] = resolve_team_name(row["AwayTeam"], candidate_teams)
                row = out.loc[idx]
            if not league or league == "NAN":
                home_set = set(team_map.get(str(row["HomeTeam"]), []))
                away_set = set(team_map.get(str(row["AwayTeam"]), []))
                candidates = sorted((home_set & away_set) & allowed)
                if len(candidates) != 1:
                    raise ValueError(
                        f"Cannot infer league for {row['HomeTeam']} vs {row['AwayTeam']}. "
                        f"Add a League column using one of the trained codes: {', '.join(sorted(allowed))}."
                    )
                league = candidates[0]
            if league not in allowed:
                raise ValueError(
                    f"Fixture {row['HomeTeam']} vs {row['AwayTeam']} is league {league!r}, "
                    f"but this model was trained only on: {', '.join(sorted(allowed))}."
                )
            inferred.append(league)
        out["League"] = inferred
        return out

    def predict(self, fixtures: pd.DataFrame, advanced_data: AdvancedDataBundle | None = None) -> pd.DataFrame:
        fixtures = self.prepare_fixtures(fixtures)
        merged = merge_bundles(self.advanced_bundle_, advanced_data)
        if merged is not None:
            self.builder.advanced_store = AdvancedFeatureStore(merged, rolling_matches=self.rolling_window)
        feat = self.builder.build_fixture_frame(fixtures)
        pred = self.model.predict_frame(feat)

        for side, pcol, aliases in [
            ("Home", "P_Home", ["AvgH", "B365H", "MaxH", "CloseH"]),
            ("Draw", "P_Draw", ["AvgD", "B365D", "MaxD", "CloseD"]),
            ("Away", "P_Away", ["AvgA", "B365A", "MaxA", "CloseA"]),
        ]:
            odds = None
            for c in aliases:
                if c in fixtures.columns:
                    odds = pd.to_numeric(fixtures[c], errors="coerce")
                    break
            if odds is not None:
                pred[f"Odds_{side}"] = odds.values
                pred[f"Edge_{side}"] = pred[pcol] * odds.values - 1.0
                b = odds.values - 1.0
                q = 1.0 - pred[pcol].values
                with np.errstate(divide="ignore", invalid="ignore"):
                    full_kelly = np.where(b > 0, (b * pred[pcol].values - q) / b, 0.0)
                pred[f"QuarterKelly_{side}"] = np.clip(np.nan_to_num(full_kelly, nan=0.0), 0.0, 0.025)
        return pred

    @property
    def metrics(self):
        return self.model.metrics_

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
