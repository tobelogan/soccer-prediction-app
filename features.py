from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import pow, sqrt

import numpy as np
import pandas as pd

from .advanced import AdvancedFeatureStore, empty_advanced_features
from .config import ELO_HOME_ADVANTAGE, ELO_K, INITIAL_ELO, LEAGUES, ROLLING_WINDOW

STAT_FIELDS = ["gf", "ga", "shots", "sot", "corners", "points"]
COMP_STAT_FIELDS = ["gf", "ga", "points", "adj_gf", "adj_ga"]
LEAGUE_ONE_HOT_COLUMNS = [f"league_is_{code}" for code in LEAGUES]
FORM_HISTORY_WINDOW = max(20, ROLLING_WINDOW)
LEAGUE_PRIOR_WINDOW = 380


@dataclass
class TeamState:
    elo: float = INITIAL_ELO
    stats: dict = field(default_factory=lambda: {k: deque(maxlen=FORM_HISTORY_WINDOW) for k in STAT_FIELDS})
    # Separate all-competition state. League matches enter with weight 1.0;
    # cups/continental matches can be injected at prediction time with a
    # competition-quality weight. This avoids contaminating league priors/Elo.
    comp_stats: dict = field(default_factory=lambda: {k: deque(maxlen=FORM_HISTORY_WINDOW) for k in COMP_STAT_FIELDS})
    dates: deque = field(default_factory=lambda: deque(maxlen=80))

    def mean(self, name: str, default: float, n: int | None = None) -> float:
        values = list(self.stats[name])
        if n is not None:
            values = values[-int(n):]
        return float(np.mean(values)) if values else float(default)

    def ewma(self, name: str, default: float, n: int = 10, decay: float = 0.72) -> float:
        values = np.asarray(list(self.stats[name])[-int(n):], dtype=float)
        if not len(values):
            return float(default)
        age = np.arange(len(values) - 1, -1, -1, dtype=float)
        weights = np.power(float(decay), age)
        return float(np.average(values, weights=weights))

    def comp_mean(self, name: str, default: float, n: int | None = None) -> float:
        store = getattr(self, "comp_stats", None) or {}
        values = list(store.get(name, []))
        if n is not None:
            values = values[-int(n):]
        return float(np.mean(values)) if values else float(default)

    def comp_ewma(self, name: str, default: float, n: int = 10, decay: float = 0.72) -> float:
        store = getattr(self, "comp_stats", None) or {}
        values = np.asarray(list(store.get(name, []))[-int(n):], dtype=float)
        if not len(values):
            return float(default)
        age = np.arange(len(values) - 1, -1, -1, dtype=float)
        weights = np.power(float(decay), age)
        return float(np.average(values, weights=weights))


@dataclass
class LeagueState:
    matches: int = 0
    home_goals: float = 0.0
    away_goals: float = 0.0
    draws: int = 0
    recent_home_goals: deque = field(default_factory=lambda: deque(maxlen=LEAGUE_PRIOR_WINDOW))
    recent_away_goals: deque = field(default_factory=lambda: deque(maxlen=LEAGUE_PRIOR_WINDOW))
    recent_draws: deque = field(default_factory=lambda: deque(maxlen=LEAGUE_PRIOR_WINDOW))

    def priors(self) -> tuple[float, float, float, float]:
        recent_h = getattr(self, "recent_home_goals", None)
        recent_a = getattr(self, "recent_away_goals", None)
        recent_d = getattr(self, "recent_draws", None)
        if recent_h:
            h = float(np.mean(recent_h))
            a = float(np.mean(recent_a))
            d = float(np.mean(recent_d))
            return h, a, h + a, d
        if self.matches <= 0:
            return 1.45, 1.15, 2.60, 0.27
        h = self.home_goals / self.matches
        a = self.away_goals / self.matches
        return float(h), float(a), float(h + a), float(self.draws / self.matches)


class FeatureBuilder:
    def __init__(self, rolling_window: int = ROLLING_WINDOW, advanced_store: AdvancedFeatureStore | None = None):
        self.rolling_window = rolling_window
        self.advanced_store = advanced_store
        self.teams: dict[tuple[str, str], TeamState] = defaultdict(self._new_state)
        self.leagues: dict[str, LeagueState] = defaultdict(LeagueState)
        self.league_defaults = {
            "gf": 1.35,
            "ga": 1.35,
            "shots": 12.0,
            "sot": 4.2,
            "corners": 5.0,
            "points": 1.35,
        }

    def _new_state(self):
        state = TeamState()
        for k in state.stats:
            state.stats[k] = deque(maxlen=max(FORM_HISTORY_WINDOW, self.rolling_window))
        for k in state.comp_stats:
            state.comp_stats[k] = deque(maxlen=max(FORM_HISTORY_WINDOW, self.rolling_window))
        state.dates = deque(maxlen=max(80, self.rolling_window * 6))
        return state

    @staticmethod
    def _league(row) -> str:
        value = row.get("League", row.get("Div", "UNKNOWN"))
        if pd.isna(value):
            return "UNKNOWN"
        return str(value).strip().upper() or "UNKNOWN"

    @staticmethod
    def _num(row, key, default=0.0):
        value = row.get(key, default)
        try:
            if pd.isna(value):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _market_probs(row) -> tuple[float, float, float]:
        triplets = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("MaxH", "MaxD", "MaxA")]
        for h, d, a in triplets:
            try:
                odds = np.array([float(row[h]), float(row[d]), float(row[a])], dtype=float)
                if np.all(np.isfinite(odds)) and np.all(odds > 1.0):
                    inv = 1.0 / odds
                    probs = inv / inv.sum()
                    return tuple(probs.tolist())
            except Exception:
                pass
        return (np.nan, np.nan, np.nan)

    @staticmethod
    def _first_odds(row, triplets) -> tuple[float, float, float]:
        for h, d, a in triplets:
            try:
                odds = np.array([float(row[h]), float(row[d]), float(row[a])], dtype=float)
                if np.all(np.isfinite(odds)) and np.all(odds > 1.0):
                    return tuple(map(float, odds))
            except Exception:
                pass
        return (np.nan, np.nan, np.nan)

    @classmethod
    def _betting_metadata(cls, row) -> dict[str, float]:
        opening = cls._first_odds(row, [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("MaxH", "MaxD", "MaxA")])
        closing = cls._first_odds(row, [("B365CH", "B365CD", "B365CA"), ("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA")])
        return {
            "raw_open_home_odds": opening[0], "raw_open_draw_odds": opening[1], "raw_open_away_odds": opening[2],
            "raw_close_home_odds": closing[0], "raw_close_draw_odds": closing[1], "raw_close_away_odds": closing[2],
        }

    @staticmethod
    def _schedule_state(state: TeamState, date) -> tuple[float, float, float]:
        d = pd.Timestamp(date)
        if not state.dates:
            return (np.nan, 0.0, 0.0)
        dates = list(state.dates)
        last = dates[-1]
        rest = max(0.0, (d - last).total_seconds() / 86400.0)
        m7 = float(sum(x >= d - pd.Timedelta(days=7) for x in dates))
        m14 = float(sum(x >= d - pd.Timedelta(days=14) for x in dates))
        return (rest, m7, m14)

    @staticmethod
    def _form_goal_rate(attacking_gf: float, opponent_ga: float, league_goal_rate: float, venue_rate: float) -> float:
        base = max(float(league_goal_rate), 0.25)
        attack_ratio = max(float(attacking_gf), 0.10) / base
        defense_ratio = max(float(opponent_ga), 0.10) / base
        estimate = max(float(venue_rate), 0.25) * sqrt(max(attack_ratio * defense_ratio, 0.01))
        return float(np.clip(estimate, 0.10, 7.50))

    def _features_for(self, home: str, away: str, row) -> dict:
        league = self._league(row)
        h = self.teams[(league, home)]
        a = self.teams[(league, away)]
        hp, dp, ap = self._market_probs(row)
        date = pd.to_datetime(row.get("Date"), errors="coerce")
        month = float(date.month) if not pd.isna(date) else 6.0
        hrest, hm7, hm14 = self._schedule_state(h, date) if not pd.isna(date) else (np.nan, 0.0, 0.0)
        arest, am7, am14 = self._schedule_state(a, date) if not pd.isna(date) else (np.nan, 0.0, 0.0)
        lhome, laway, ltotal, ldraw = self.leagues[league].priors()
        league_team_goal = max((lhome + laway) / 2.0, 0.25)

        # Multi-horizon recent form. The old model used one flat 8-match mean;
        # these features let the learner distinguish a sudden scoring surge from
        # a merely good long-run average.
        hgf3, hgf5, hgf10 = h.mean("gf", self.league_defaults["gf"], 3), h.mean("gf", self.league_defaults["gf"], 5), h.mean("gf", self.league_defaults["gf"], 10)
        hga3, hga5, hga10 = h.mean("ga", self.league_defaults["ga"], 3), h.mean("ga", self.league_defaults["ga"], 5), h.mean("ga", self.league_defaults["ga"], 10)
        agf3, agf5, agf10 = a.mean("gf", self.league_defaults["gf"], 3), a.mean("gf", self.league_defaults["gf"], 5), a.mean("gf", self.league_defaults["gf"], 10)
        aga3, aga5, aga10 = a.mean("ga", self.league_defaults["ga"], 3), a.mean("ga", self.league_defaults["ga"], 5), a.mean("ga", self.league_defaults["ga"], 10)
        hgf_ewm, hga_ewm = h.ewma("gf", self.league_defaults["gf"]), h.ewma("ga", self.league_defaults["ga"])
        agf_ewm, aga_ewm = a.ewma("gf", self.league_defaults["gf"]), a.ewma("ga", self.league_defaults["ga"])

        # All-competition + opponent-strength adjusted form. During historical
        # training these start from league games, so the model learns the scale;
        # prediction-time cup/UCL/UEL results can then update the same state.
        hcgf5 = h.comp_mean("gf", hgf5, 5); hcga5 = h.comp_mean("ga", hga5, 5)
        acgf5 = a.comp_mean("gf", agf5, 5); acga5 = a.comp_mean("ga", aga5, 5)
        hcadjgf5 = h.comp_mean("adj_gf", hgf5, 5); hcadjga5 = h.comp_mean("adj_ga", hga5, 5)
        acadjgf5 = a.comp_mean("adj_gf", agf5, 5); acadjga5 = a.comp_mean("adj_ga", aga5, 5)
        hcattack = h.comp_ewma("adj_gf", hgf_ewm); hcdef = h.comp_ewma("adj_ga", hga_ewm)
        acattack = a.comp_ewma("adj_gf", agf_ewm); acdef = a.comp_ewma("adj_ga", aga_ewm)
        hcppg5 = h.comp_mean("points", h.mean("points", self.league_defaults["points"], 5), 5)
        acppg5 = a.comp_mean("points", a.mean("points", self.league_defaults["points"], 5), 5)

        # Blend league form with opponent-adjusted all-competition form. This is
        # still calibrated later against actual goals, so it is not a hard-coded
        # big-club boost.
        h_attack_live = 0.55 * hgf_ewm + 0.45 * hcattack
        h_def_live = 0.55 * hga_ewm + 0.45 * hcdef
        a_attack_live = 0.55 * agf_ewm + 0.45 * acattack
        a_def_live = 0.55 * aga_ewm + 0.45 * acdef
        form_home_xg = self._form_goal_rate(h_attack_live, a_def_live, league_team_goal, lhome)
        form_away_xg = self._form_goal_rate(a_attack_live, h_def_live, league_team_goal, laway)

        feat = {
            "home_elo": h.elo,
            "away_elo": a.elo,
            "elo_diff": h.elo + ELO_HOME_ADVANTAGE - a.elo,
            "home_gf": h.mean("gf", self.league_defaults["gf"], self.rolling_window),
            "home_ga": h.mean("ga", self.league_defaults["ga"], self.rolling_window),
            "away_gf": a.mean("gf", self.league_defaults["gf"], self.rolling_window),
            "away_ga": a.mean("ga", self.league_defaults["ga"], self.rolling_window),
            "home_shots": h.mean("shots", self.league_defaults["shots"], self.rolling_window),
            "away_shots": a.mean("shots", self.league_defaults["shots"], self.rolling_window),
            "home_sot": h.mean("sot", self.league_defaults["sot"], self.rolling_window),
            "away_sot": a.mean("sot", self.league_defaults["sot"], self.rolling_window),
            "home_corners": h.mean("corners", self.league_defaults["corners"], self.rolling_window),
            "away_corners": a.mean("corners", self.league_defaults["corners"], self.rolling_window),
            "home_ppg": h.mean("points", self.league_defaults["points"], self.rolling_window),
            "away_ppg": a.mean("points", self.league_defaults["points"], self.rolling_window),
            "home_gf_3": hgf3, "home_gf_5": hgf5, "home_gf_10": hgf10,
            "home_ga_3": hga3, "home_ga_5": hga5, "home_ga_10": hga10,
            "away_gf_3": agf3, "away_gf_5": agf5, "away_gf_10": agf10,
            "away_ga_3": aga3, "away_ga_5": aga5, "away_ga_10": aga10,
            "home_ppg_5": h.mean("points", self.league_defaults["points"], 5),
            "away_ppg_5": a.mean("points", self.league_defaults["points"], 5),
            "home_ppg_10": h.mean("points", self.league_defaults["points"], 10),
            "away_ppg_10": a.mean("points", self.league_defaults["points"], 10),
            "home_sot_5": h.mean("sot", self.league_defaults["sot"], 5),
            "away_sot_5": a.mean("sot", self.league_defaults["sot"], 5),
            "home_goal_diff_5": hgf5 - hga5,
            "away_goal_diff_5": agf5 - aga5,
            "home_attack_ewm": hgf_ewm,
            "home_defense_ewm": hga_ewm,
            "away_attack_ewm": agf_ewm,
            "away_defense_ewm": aga_ewm,
            "home_allcomp_gf_5": hcgf5, "away_allcomp_gf_5": acgf5,
            "home_allcomp_ga_5": hcga5, "away_allcomp_ga_5": acga5,
            "home_allcomp_ppg_5": hcppg5, "away_allcomp_ppg_5": acppg5,
            "home_allcomp_goal_diff_5": hcgf5 - hcga5, "away_allcomp_goal_diff_5": acgf5 - acga5,
            "home_opp_adj_gf_5": hcadjgf5, "away_opp_adj_gf_5": acadjgf5,
            "home_opp_adj_ga_5": hcadjga5, "away_opp_adj_ga_5": acadjga5,
            "home_allcomp_attack_ewm": hcattack, "away_allcomp_attack_ewm": acattack,
            "home_allcomp_defense_ewm": hcdef, "away_allcomp_defense_ewm": acdef,
            "form_home_xg": form_home_xg,
            "form_away_xg": form_away_xg,
            "home_rest_days": hrest,
            "away_rest_days": arest,
            "home_matches_7d": hm7,
            "away_matches_7d": am7,
            "home_matches_14d": hm14,
            "away_matches_14d": am14,
            "market_home": hp,
            "market_draw": dp,
            "market_away": ap,
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "league_home_goal_prior": lhome,
            "league_away_goal_prior": laway,
            "league_total_goal_prior": ltotal,
            "league_draw_rate_prior": ldraw,
        }
        for code in LEAGUES:
            feat[f"league_is_{code}"] = 1.0 if league == code else 0.0
        if self.advanced_store is not None:
            feat.update(self.advanced_store.features_for(row))
        else:
            feat.update(empty_advanced_features())
        return feat

    @staticmethod
    def _strength_factor(opponent_elo: float) -> float:
        # Modest Elo adjustment: roughly 0.75..1.33 over practical club ranges.
        try:
            factor = pow(10.0, (float(opponent_elo) - INITIAL_ELO) / 1200.0)
        except Exception:
            factor = 1.0
        return float(np.clip(factor, 0.75, 1.33))

    def _append_comp_result(self, state: TeamState, gf: float, ga: float, points: float, opponent_elo: float, competition_weight: float = 1.0):
        if not hasattr(state, "comp_stats") or state.comp_stats is None:
            state.comp_stats = {k: deque(maxlen=max(FORM_HISTORY_WINDOW, self.rolling_window)) for k in COMP_STAT_FIELDS}
        w = float(np.clip(competition_weight, 0.20, 1.0))
        strength = self._strength_factor(opponent_elo)
        # Pull lower-weight competitions gently toward neutral rather than
        # multiplying goals toward zero.
        neutral_goal = self.league_defaults["gf"]
        state.comp_stats["gf"].append(w * float(gf) + (1.0 - w) * neutral_goal)
        state.comp_stats["ga"].append(w * float(ga) + (1.0 - w) * neutral_goal)
        state.comp_stats["points"].append(w * float(points) + (1.0 - w) * self.league_defaults["points"])
        state.comp_stats["adj_gf"].append(w * float(gf) * strength + (1.0 - w) * neutral_goal)
        state.comp_stats["adj_ga"].append(w * float(ga) / strength + (1.0 - w) * neutral_goal)

    def _update(self, row):
        league = self._league(row)
        home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        h, a = self.teams[(league, home)], self.teams[(league, away)]

        if hg > ag:
            hp, ap = 3.0, 0.0
            actual_h = 1.0
        elif hg == ag:
            hp, ap = 1.0, 1.0
            actual_h = 0.5
        else:
            hp, ap = 0.0, 3.0
            actual_h = 0.0

        # Snapshot opponent strength before this match changes Elo.
        self._append_comp_result(h, hg, ag, hp, a.elo, 1.0)
        self._append_comp_result(a, ag, hg, ap, h.elo, 1.0)
        h.stats["gf"].append(hg); h.stats["ga"].append(ag); h.stats["points"].append(hp)
        a.stats["gf"].append(ag); a.stats["ga"].append(hg); a.stats["points"].append(ap)
        h.stats["shots"].append(self._num(row, "HS", self.league_defaults["shots"]))
        a.stats["shots"].append(self._num(row, "AS", self.league_defaults["shots"]))
        h.stats["sot"].append(self._num(row, "HST", self.league_defaults["sot"]))
        a.stats["sot"].append(self._num(row, "AST", self.league_defaults["sot"]))
        h.stats["corners"].append(self._num(row, "HC", self.league_defaults["corners"]))
        a.stats["corners"].append(self._num(row, "AC", self.league_defaults["corners"]))
        d = pd.to_datetime(row.get("Date"), errors="coerce")
        if not pd.isna(d):
            h.dates.append(d); a.dates.append(d)

        expected_h = 1.0 / (1.0 + pow(10.0, -((h.elo + ELO_HOME_ADVANTAGE) - a.elo) / 400.0))
        delta = ELO_K * (actual_h - expected_h)
        h.elo += delta
        a.elo -= delta

        ls = self.leagues[league]
        ls.matches += 1
        ls.home_goals += hg
        ls.away_goals += ag
        ls.draws += int(hg == ag)
        # New rolling priors stop matches from 10-15 years ago defining today's
        # league scoring environment/home advantage.
        if not hasattr(ls, "recent_home_goals"):
            ls.recent_home_goals = deque(maxlen=LEAGUE_PRIOR_WINDOW)
            ls.recent_away_goals = deque(maxlen=LEAGUE_PRIOR_WINDOW)
            ls.recent_draws = deque(maxlen=LEAGUE_PRIOR_WINDOW)
        ls.recent_home_goals.append(hg)
        ls.recent_away_goals.append(ag)
        ls.recent_draws.append(int(hg == ag))

    def reset_competitive_state(self) -> None:
        for state in self.teams.values():
            state.comp_stats = {k: deque(maxlen=max(FORM_HISTORY_WINDOW, self.rolling_window)) for k in COMP_STAT_FIELDS}

    def update_external_competitive_matches(self, matches: pd.DataFrame, team_leagues: dict[str, str] | None = None, opponent_elos: dict[str, float] | None = None) -> int:
        """Update all-competition form without modifying league Elo/priors.

        Rows may contain teams from other leagues. Only teams mapped into this
        engine's league state are updated; the opponent can be unknown.
        """
        if matches is None or len(matches) == 0:
            return 0
        team_leagues = team_leagues or {}
        opponent_elos = opponent_elos or {}
        count = 0
        for _, row in matches.sort_values([c for c in ["Date", "Competition"] if c in matches.columns]).iterrows():
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                continue
            home, away = str(row.get("HomeTeam")), str(row.get("AwayTeam"))
            hg, ag = float(row.get("FTHG")), float(row.get("FTAG"))
            weight = self._num(row, "CompetitionWeight", 0.85)
            hp, ap = ((3.0, 0.0) if hg > ag else ((1.0, 1.0) if hg == ag else (0.0, 3.0)))
            touched = False
            home_league = team_leagues.get(home)
            away_league = team_leagues.get(away)
            if home_league and (home_league, home) in self.teams:
                self._append_comp_result(self.teams[(home_league, home)], hg, ag, hp, opponent_elos.get(away, INITIAL_ELO), weight)
                touched = True
            if away_league and (away_league, away) in self.teams:
                self._append_comp_result(self.teams[(away_league, away)], ag, hg, ap, opponent_elos.get(home, INITIAL_ELO), weight)
                touched = True
            count += int(touched)
        return count

    def update_completed_matches(self, matches: pd.DataFrame) -> int:
        """Advance team/Elo/form state with newly completed matches only."""
        if matches is None or len(matches) == 0:
            return 0
        count = 0
        sort_cols = [c for c in ["Date", "League", "HomeTeam", "AwayTeam"] if c in matches.columns]
        for _, row in matches.sort_values(sort_cols).iterrows():
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                continue
            self._update(row)
            count += 1
        return count

    def build_training_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in matches.sort_values(["Date", "League"] if "League" in matches.columns else ["Date"]).iterrows():
            league = self._league(row)
            feat = self._features_for(str(row["HomeTeam"]), str(row["AwayTeam"]), row)
            feat.update({
                "Date": row["Date"], "League": league, "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
                "home_goals": int(row["FTHG"]), "away_goals": int(row["FTAG"]),
                "result": 0 if row["FTHG"] > row["FTAG"] else (1 if row["FTHG"] == row["FTAG"] else 2),
            })
            feat.update(self._betting_metadata(row))
            rows.append(feat)
            self._update(row)
        return pd.DataFrame(rows)

    def build_fixture_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in fixtures.iterrows():
            league = self._league(row)
            feat = self._features_for(str(row["HomeTeam"]), str(row["AwayTeam"]), row)
            feat.update({"Date": row.get("Date"), "League": league, "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"]})
            rows.append(feat)
        return pd.DataFrame(rows)
