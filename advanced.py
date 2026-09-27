from __future__ import annotations

"""Optional pre-kickoff feature store for richer soccer models.

The module deliberately uses provider-neutral CSV/DataFrame schemas.  You can feed
StatsBomb/Opta/Wyscout/API-Football/Sportmonks/Betfair/etc. exports after mapping
columns to the names documented in README.md.  Every rolling statistic uses only
records strictly before the fixture timestamp, while same-match snapshots
(lineups, availability, weather and market prices) are only used when supplied.
"""

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Mapping

import numpy as np
import pandas as pd


ADVANCED_FEATURE_COLUMNS = [
    # Availability flags – useful because missing rich feeds are not equivalent to zero.
    "event_data_available", "lineup_data_available", "availability_data_available",
    "market_data_available", "exchange_data_available", "weather_data_available",
    "tactics_data_available", "transfer_data_available", "manager_data_available",
    "location_data_available", "schedule_data_available",
    # Genuine event-derived team attacking/defensive quality.
    "home_event_xgf", "home_event_xga", "away_event_xgf", "away_event_xga",
    "home_event_npxgf", "home_event_npxga", "away_event_npxgf", "away_event_npxga",
    "home_big_chances", "away_big_chances",
    # Confirmed XI / player-level xG-xA / goalkeeper.
    "home_xi_xg90", "home_xi_xa90", "away_xi_xg90", "away_xi_xa90",
    "home_xi_minutes", "away_xi_minutes", "home_xi_known", "away_xi_known",
    "home_xi_completeness", "away_xi_completeness",
    "home_xi_attack_delta_pct", "away_xi_attack_delta_pct",
    "home_gk_goals_prevented90", "away_gk_goals_prevented90",
    "home_gk_delta", "away_gk_delta",
    # Injuries, suspensions and availability impact.
    "home_injuries", "away_injuries", "home_suspensions", "away_suspensions",
    "home_unavailable_xg90", "away_unavailable_xg90",
    "home_unavailable_xa90", "away_unavailable_xa90",
    "home_unavailable_minutes", "away_unavailable_minutes",
    "home_unavailable_attack_share", "away_unavailable_attack_share",
    # Transfers and managers.
    "home_transfer_attack_delta", "away_transfer_attack_delta",
    "home_transfer_minutes_delta", "away_transfer_minutes_delta",
    "home_manager_days", "away_manager_days", "home_new_manager_30d", "away_new_manager_30d",
    # Travel / congestion / rest / European schedule.
    "home_travel_km", "away_travel_km",
    "home_rest_days_ext", "away_rest_days_ext",
    "home_matches_7d_ext", "away_matches_7d_ext", "home_matches_14d_ext", "away_matches_14d_ext",
    "home_continental_7d", "away_continental_7d", "home_next_continental_7d", "away_next_continental_7d",
    # Weather.
    "temperature_c", "wind_kph", "precip_mm", "humidity_pct",
    # Market movement, sharp close and exchange liquidity.
    "sharp_home", "sharp_draw", "sharp_away",
    "open_home", "open_draw", "open_away",
    "market_move_home", "market_move_draw", "market_move_away",
    "exchange_home", "exchange_draw", "exchange_away", "exchange_liquidity_log",
    # Tactical team ratings.
    "home_possession", "away_possession", "home_ppda", "away_ppda",
    "home_field_tilt", "away_field_tilt", "home_xt", "away_xt",
    "home_xt_against", "away_xt_against",
]


def empty_advanced_features() -> dict[str, float]:
    out = {c: np.nan for c in ADVANCED_FEATURE_COLUMNS}
    for c in ADVANCED_FEATURE_COLUMNS:
        if c.endswith("_available") or c.endswith("_known"):
            out[c] = 0.0
    # Neutral binary/default counts are safe even if feed is missing because availability flags distinguish them.
    for c in [
        "home_injuries", "away_injuries", "home_suspensions", "away_suspensions",
        "home_new_manager_30d", "away_new_manager_30d", "home_continental_7d", "away_continental_7d",
        "home_next_continental_7d", "away_next_continental_7d", "home_travel_km",
    ]:
        out[c] = 0.0
    return out


def _norm_date(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", format="mixed", dayfirst=True, utc=False)
    return out


def _num(series: pd.Series, name: str, default=np.nan) -> pd.Series:
    if name not in series.index:
        return pd.Series([default])
    return pd.to_numeric(series[name], errors="coerce")


def _row_num(row: pd.Series | Mapping, name: str, default=np.nan) -> float:
    try:
        v = row.get(name, default)
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _devig(h: float, d: float, a: float) -> tuple[float, float, float] | None:
    odds = np.asarray([h, d, a], dtype=float)
    if not np.isfinite(odds).all() or (odds <= 1.0).any():
        return None
    p = 1.0 / odds
    p /= p.sum()
    return tuple(map(float, p))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if not np.isfinite(vals).all():
        return np.nan
    r = 6371.0088
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    aa = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return float(2 * r * asin(sqrt(aa)))


@dataclass
class AdvancedDataBundle:
    events: pd.DataFrame | None = None
    player_stats: pd.DataFrame | None = None
    lineups: pd.DataFrame | None = None
    availability: pd.DataFrame | None = None
    transfers: pd.DataFrame | None = None
    managers: pd.DataFrame | None = None
    team_locations: pd.DataFrame | None = None
    schedule: pd.DataFrame | None = None
    weather: pd.DataFrame | None = None
    markets: pd.DataFrame | None = None
    tactics: pd.DataFrame | None = None

    @classmethod
    def from_paths(cls, **paths):
        kwargs = {}
        for name in cls.__dataclass_fields__:
            path = paths.get(name)
            kwargs[name] = pd.read_csv(path, on_bad_lines="skip") if path else None
        return cls(**kwargs)


class AdvancedFeatureStore:
    """Leakage-safe feature queries over optional rich-data tables."""

    def __init__(self, bundle: AdvancedDataBundle | None = None, rolling_matches: int = 8):
        self.bundle = bundle or AdvancedDataBundle()
        self.rolling_matches = int(rolling_matches)
        for name in self.bundle.__dataclass_fields__:
            setattr(self, name, _norm_date(getattr(self.bundle, name)))
        self._prepare_events()
        self._prepare_player_stats()
        self._prepare_locations()

    def _prepare_events(self):
        self.team_event_matches = None
        if self.events is None or self.events.empty or not {"Date", "Team"}.issubset(self.events.columns):
            return
        e = self.events.copy()
        for c in ["xG", "npxG", "BigChance"]:
            if c not in e.columns:
                e[c] = np.nan if c != "BigChance" else 0.0
            e[c] = pd.to_numeric(e[c], errors="coerce")
        # One row per team per match/date.  Opponent xG becomes xGA when opponent is identifiable.
        keys = ["Date", "Team"]
        if "MatchID" in e.columns:
            keys.insert(0, "MatchID")
        agg = e.groupby(keys, dropna=False).agg(
            xgf=("xG", "sum"), npxgf=("npxG", "sum"), big_chances=("BigChance", "sum")
        ).reset_index()
        if "Opponent" in e.columns:
            opp = e[keys + ["Opponent"]].drop_duplicates(keys)
            agg = agg.merge(opp, on=keys, how="left")
            # Self-join on same match/date and Team == Opponent.
            join_keys = [k for k in keys if k != "Team"]
            opp_lookup = agg[join_keys + ["Team", "xgf", "npxgf"]].copy()
            opp_lookup = opp_lookup.rename(columns={"Team": "Opponent", "xgf": "xga", "npxgf": "npxga"})
            agg = agg.merge(opp_lookup, on=join_keys + ["Opponent"], how="left")
        elif "MatchID" in agg.columns:
            # Provider exports often omit Opponent but have a stable MatchID. Derive the other team.
            opp_lookup = agg[["MatchID", "Team", "xgf", "npxgf"]].rename(
                columns={"Team": "Opponent", "xgf": "xga", "npxgf": "npxga"}
            )
            tmp = agg.merge(opp_lookup, on="MatchID", how="left")
            tmp = tmp[tmp["Team"].astype(str) != tmp["Opponent"].astype(str)]
            keep = [c for c in agg.columns] + ["Opponent", "xga", "npxga"]
            agg = tmp[keep].drop_duplicates(["MatchID", "Team"], keep="first")
        else:
            agg["xga"] = np.nan
            agg["npxga"] = np.nan
        self.team_event_matches = agg.sort_values("Date")

    def _prepare_player_stats(self):
        if self.player_stats is None or self.player_stats.empty:
            return
        p = self.player_stats
        for c in ["Minutes", "xG", "xA", "PSxG", "GoalsAllowed"]:
            if c not in p.columns:
                p[c] = np.nan
            p[c] = pd.to_numeric(p[c], errors="coerce")
        if "Role" not in p.columns:
            p["Role"] = ""
        self.player_stats = p.sort_values("Date")

    def _prepare_locations(self):
        self.location_map: dict[str, tuple[float, float]] = {}
        if self.team_locations is None or self.team_locations.empty:
            return
        if not {"Team", "Lat", "Lon"}.issubset(self.team_locations.columns):
            return
        for _, r in self.team_locations.iterrows():
            try:
                self.location_map[str(r["Team"])] = (float(r["Lat"]), float(r["Lon"]))
            except Exception:
                pass

    @staticmethod
    def _same_match(df: pd.DataFrame | None, date, home: str, away: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        x = df
        if "Date" in x.columns:
            # Exact timestamp if possible; fall back to calendar date for provider exports without kickoff time.
            d = pd.Timestamp(date)
            exact = x[x["Date"] == d]
            if not exact.empty:
                x = exact
            else:
                x = x[x["Date"].dt.date == d.date()]
        if {"HomeTeam", "AwayTeam"}.issubset(x.columns):
            x = x[(x["HomeTeam"].astype(str) == home) & (x["AwayTeam"].astype(str) == away)]
        return x

    def _rolling_team_event(self, team: str, date) -> dict[str, float]:
        if self.team_event_matches is None:
            return {}
        x = self.team_event_matches
        x = x[(x["Team"].astype(str) == team) & (x["Date"] < pd.Timestamp(date))].tail(self.rolling_matches)
        if x.empty:
            return {}
        return {
            "xgf": float(x["xgf"].mean()), "xga": float(x["xga"].mean()) if x["xga"].notna().any() else np.nan,
            "npxgf": float(x["npxgf"].mean()) if x["npxgf"].notna().any() else np.nan,
            "npxga": float(x["npxga"].mean()) if x["npxga"].notna().any() else np.nan,
            "big_chances": float(x["big_chances"].mean()),
        }

    def _player_form(self, player: str, date) -> dict[str, float]:
        if self.player_stats is None or self.player_stats.empty or "Player" not in self.player_stats.columns:
            return {}
        p = self.player_stats
        x = p[(p["Player"].astype(str) == str(player)) & (p["Date"] < pd.Timestamp(date))].tail(12)
        if x.empty:
            return {}
        mins = float(x["Minutes"].fillna(0).sum())
        scale = 90.0 / max(mins, 90.0)
        psxg = float(x["PSxG"].fillna(0).sum())
        ga = float(x["GoalsAllowed"].fillna(0).sum())
        return {
            "xg90": float(x["xG"].fillna(0).sum() * scale),
            "xa90": float(x["xA"].fillna(0).sum() * scale),
            "minutes": mins,
            "gk_gp90": float((psxg - ga) * scale),
        }

    def _team_expected_xi_baseline(self, team: str, date) -> dict[str, float]:
        if self.player_stats is None or self.player_stats.empty or not {"Team", "Player"}.issubset(self.player_stats.columns):
            return {}
        d = pd.Timestamp(date)
        p = self.player_stats
        x = p[(p["Team"].astype(str) == team) & (p["Date"] < d) & (p["Date"] >= d - pd.Timedelta(days=240))]
        if x.empty:
            return {}
        totals = x.groupby("Player", dropna=False)["Minutes"].sum().sort_values(ascending=False)
        players = [str(v) for v in totals.head(16).index]
        rows = []
        for player in players:
            f = self._player_form(player, d)
            if not f:
                continue
            role_rows = x[x["Player"].astype(str) == player]
            role = str(role_rows.iloc[-1].get("Role", "")) if len(role_rows) else ""
            rows.append((player, f, role))
        if not rows:
            return {}
        outfield = sorted(rows, key=lambda z: z[1].get("minutes", 0.0), reverse=True)[:11]
        xg = float(sum(z[1].get("xg90", 0.0) for z in outfield))
        xa = float(sum(z[1].get("xa90", 0.0) for z in outfield))
        gks = [z[1].get("gk_gp90", np.nan) for z in rows if z[2].upper() in {"GK", "GOALKEEPER"}]
        gks = [v for v in gks if np.isfinite(v)]
        return {"xg90": xg, "xa90": xa, "attack": xg + 0.70 * xa, "gk": float(np.mean(gks)) if gks else np.nan}

    def _lineup_features(self, team: str, date, home: str, away: str) -> dict[str, float]:
        if self.lineups is None or self.lineups.empty or not {"Date", "Team", "Player"}.issubset(self.lineups.columns):
            return {}
        x = self._same_match(self.lineups, date, home, away)
        if x.empty:
            return {}
        x = x[x["Team"].astype(str) == team]
        if "Confirmed" in x.columns:
            conf = x["Confirmed"].astype(str).str.lower().isin(["1", "true", "yes", "confirmed"])
            x = x[conf]
        if "IsStarter" in x.columns:
            starters = x["IsStarter"].astype(str).str.lower().isin(["1", "true", "yes", "starter"])
            x = x[starters]
        if x.empty:
            return {}
        forms = [self._player_form(str(p), date) for p in x["Player"]]
        forms = [f for f in forms if f]
        if not forms:
            return {"known": 1.0, "xg90": np.nan, "xa90": np.nan, "minutes": np.nan, "gk": np.nan}
        # Sum rates across XI: a lineup-level attacking/creative capacity feature.
        gk_vals = []
        for (_, r), f in zip(x.iterrows(), [self._player_form(str(p), date) for p in x["Player"]]):
            if not f:
                continue
            is_gk = str(r.get("Role", "")).upper() in {"GK", "GOALKEEPER"}
            if is_gk:
                gk_vals.append(f["gk_gp90"])
        xg90 = float(sum(f["xg90"] for f in forms))
        xa90 = float(sum(f["xa90"] for f in forms))
        gk = float(np.mean(gk_vals)) if gk_vals else np.nan
        baseline = self._team_expected_xi_baseline(team, date)
        attack = xg90 + 0.70 * xa90
        base_attack = baseline.get("attack", np.nan)
        attack_delta = (attack / base_attack - 1.0) if np.isfinite(base_attack) and base_attack > 0.05 else np.nan
        base_gk = baseline.get("gk", np.nan)
        gk_delta = (gk - base_gk) if np.isfinite(gk) and np.isfinite(base_gk) else np.nan
        return {
            "known": 1.0,
            "xg90": xg90, "xa90": xa90,
            "minutes": float(sum(f["minutes"] for f in forms)),
            "gk": gk, "gk_delta": gk_delta,
            "completeness": float(min(len(x), 11) / 11.0),
            "attack_delta_pct": float(np.clip(attack_delta, -1.5, 1.5)) if np.isfinite(attack_delta) else np.nan,
        }

    def _availability_features(self, team: str, date, home: str, away: str) -> dict[str, float]:
        if self.availability is None or self.availability.empty or not {"Date", "Team"}.issubset(self.availability.columns):
            return {}
        x = self._same_match(self.availability, date, home, away)
        x = x[x["Team"].astype(str) == team]
        if x.empty:
            return {}
        reason = x.get("Reason", pd.Series("", index=x.index)).astype(str).str.lower()
        status = x.get("Status", pd.Series("out", index=x.index)).astype(str).str.lower()
        confidence = pd.to_numeric(
            x.get("SourceConfidence", pd.Series(1.0, index=x.index)), errors="coerce"
        ).fillna(1.0).clip(0.0, 1.0)

        # Confirmed absences get full weight. Doubtful/questionable news is
        # deliberately down-weighted so a speculative article is not treated
        # like a confirmed suspension or medical ruling.
        status_weight = pd.Series(1.0, index=x.index, dtype=float)
        status_weight.loc[status.isin(["available", "fit", "active", "starting", "bench"])] = 0.0
        status_weight.loc[status.isin(["doubtful", "questionable", "50/50", "uncertain"])] = 0.40
        status_weight.loc[status.isin(["probable", "likely", "expected"])] = 0.15
        effective = (confidence * status_weight).clip(0.0, 1.0)
        keep = effective > 0
        x = x[keep]
        reason = reason.loc[x.index]
        effective = effective.loc[x.index]

        xg, xa, mins = 0.0, 0.0, 0.0
        for idx, r in x.iterrows():
            weight = float(effective.loc[idx])
            f = self._player_form(str(r.get("Player", "")), date)
            xg += weight * _row_num(r, "PlayerXG90", f.get("xg90", 0.0) if f else 0.0)
            xa += weight * _row_num(r, "PlayerXA90", f.get("xa90", 0.0) if f else 0.0)
            mins += weight * _row_num(r, "ExpectedMinutes", 0.0)
        injury_mask = reason.str.contains("injur|fitness|hamstring|ankle|knee|muscle|knock", regex=True)
        suspension_mask = reason.str.contains("susp|card|ban", regex=True)
        baseline = self._team_expected_xi_baseline(team, date)
        base_attack = baseline.get("attack", np.nan)
        lost_attack = float(xg + 0.70 * xa)
        share = lost_attack / base_attack if np.isfinite(base_attack) and base_attack > 0.05 else np.nan
        return {
            "injuries": float(effective[injury_mask].sum()),
            "suspensions": float(effective[suspension_mask].sum()),
            "xg90": float(xg), "xa90": float(xa), "minutes": float(mins),
            "attack_share": float(np.clip(share, 0.0, 2.0)) if np.isfinite(share) else np.nan,
        }

    def _transfer_features(self, team: str, date) -> dict[str, float]:
        if self.transfers is None or self.transfers.empty or not {"Date", "Team"}.issubset(self.transfers.columns):
            return {}
        d = pd.Timestamp(date)
        x = self.transfers[(self.transfers["Team"].astype(str) == team) & (self.transfers["Date"] < d) & (self.transfers["Date"] >= d - pd.Timedelta(days=120))]
        if x.empty:
            return {}
        attack, minutes = 0.0, 0.0
        for _, r in x.iterrows():
            direction = str(r.get("Direction", "IN")).upper()
            sign = 1.0 if direction in {"IN", "JOIN", "ARRIVAL"} else -1.0
            f = self._player_form(str(r.get("Player", "")), date)
            strength = _row_num(r, "PlayerXG90", f.get("xg90", 0.0) if f else 0.0) + _row_num(r, "PlayerXA90", f.get("xa90", 0.0) if f else 0.0)
            attack += sign * strength
            minutes += sign * _row_num(r, "PriorMinutes", f.get("minutes", 0.0) if f else 0.0)
        return {"attack": float(attack), "minutes": float(minutes)}

    def _manager_features(self, team: str, date) -> dict[str, float]:
        if self.managers is None or self.managers.empty or not {"Date", "Team"}.issubset(self.managers.columns):
            return {}
        d = pd.Timestamp(date)
        x = self.managers[(self.managers["Team"].astype(str) == team) & (self.managers["Date"] <= d)].sort_values("Date")
        if x.empty:
            return {}
        start = x.iloc[-1]["Date"]
        days = max(0.0, (d - start).total_seconds() / 86400.0)
        return {"days": float(days), "new30": float(days <= 30)}

    def _schedule_features(self, team: str, date) -> dict[str, float]:
        if self.schedule is None or self.schedule.empty or not {"Date", "Team"}.issubset(self.schedule.columns):
            return {}
        d = pd.Timestamp(date)
        x = self.schedule[self.schedule["Team"].astype(str) == team].copy()
        prev = x[x["Date"] < d].sort_values("Date")
        nxt = x[x["Date"] > d].sort_values("Date")
        rest = np.nan
        if not prev.empty:
            rest = max(0.0, (d - prev.iloc[-1]["Date"]).total_seconds() / 86400.0)
        m7 = float(((prev["Date"] >= d - pd.Timedelta(days=7))).sum())
        m14 = float(((prev["Date"] >= d - pd.Timedelta(days=14))).sum())
        if "IsContinental" in x.columns:
            cont = x["IsContinental"].astype(str).str.lower().isin(["1", "true", "yes", "ucl", "uel", "uecl"])
        elif "Competition" in x.columns:
            cont = x["Competition"].astype(str).str.upper().str.contains("UCL|CHAMPIONS|EUROPA|UEL|UECL|CONFERENCE")
        else:
            cont = pd.Series(False, index=x.index)
        prev_cont = x[(x["Date"] < d) & (x["Date"] >= d - pd.Timedelta(days=7)) & cont]
        next_cont = x[(x["Date"] > d) & (x["Date"] <= d + pd.Timedelta(days=7)) & cont]
        return {"rest": float(rest), "m7": m7, "m14": m14, "cont7": float(len(prev_cont) > 0), "next_cont7": float(len(next_cont) > 0)}

    def _tactical_features(self, team: str, date) -> dict[str, float]:
        if self.tactics is None or self.tactics.empty or not {"Date", "Team"}.issubset(self.tactics.columns):
            return {}
        x = self.tactics[(self.tactics["Team"].astype(str) == team) & (self.tactics["Date"] < pd.Timestamp(date))].tail(self.rolling_matches)
        if x.empty:
            return {}
        out = {}
        for src, dst in [("Possession", "possession"), ("PPDA", "ppda"), ("FieldTilt", "field_tilt"), ("xT", "xt"), ("xTAgainst", "xt_against")]:
            if src in x.columns:
                vals = pd.to_numeric(x[src], errors="coerce")
                out[dst] = float(vals.mean()) if vals.notna().any() else np.nan
        return out

    def _market_features(self, date, home: str, away: str) -> dict[str, float]:
        x = self._same_match(self.markets, date, home, away)
        if x.empty:
            return {}
        r = x.iloc[-1]
        out: dict[str, float] = {}
        for prefix, names in {
            "open": ("OpenH", "OpenD", "OpenA"),
            "sharp": ("SharpH", "SharpD", "SharpA"),
            "exchange": ("ExchangeH", "ExchangeD", "ExchangeA"),
        }.items():
            probs = _devig(*[_row_num(r, c) for c in names])
            if probs:
                out[f"{prefix}_home"], out[f"{prefix}_draw"], out[f"{prefix}_away"] = probs
        close_probs = _devig(*[_row_num(r, c) for c in ("CloseH", "CloseD", "CloseA")])
        if close_probs and all(k in out for k in ["open_home", "open_draw", "open_away"]):
            out["move_home"] = close_probs[0] - out["open_home"]
            out["move_draw"] = close_probs[1] - out["open_draw"]
            out["move_away"] = close_probs[2] - out["open_away"]
        liq = _row_num(r, "ExchangeLiquidity", np.nan)
        out["liq_log"] = float(np.log1p(liq)) if np.isfinite(liq) and liq >= 0 else np.nan
        return out

    def _weather_features(self, date, home: str, away: str) -> dict[str, float]:
        x = self._same_match(self.weather, date, home, away)
        if x.empty:
            return {}
        r = x.iloc[-1]
        return {
            "temp": _row_num(r, "TempC"), "wind": _row_num(r, "WindKph"),
            "precip": _row_num(r, "PrecipMm"), "humidity": _row_num(r, "HumidityPct"),
        }

    def features_for(self, row: pd.Series | Mapping) -> dict[str, float]:
        out = empty_advanced_features()
        date = pd.Timestamp(row.get("Date"))
        home, away = str(row.get("HomeTeam")), str(row.get("AwayTeam"))

        he, ae = self._rolling_team_event(home, date), self._rolling_team_event(away, date)
        if he or ae:
            out["event_data_available"] = 1.0
            for side, f in [("home", he), ("away", ae)]:
                out[f"{side}_event_xgf"] = f.get("xgf", np.nan)
                out[f"{side}_event_xga"] = f.get("xga", np.nan)
                out[f"{side}_event_npxgf"] = f.get("npxgf", np.nan)
                out[f"{side}_event_npxga"] = f.get("npxga", np.nan)
                out[f"{side}_big_chances"] = f.get("big_chances", np.nan)

        hl = self._lineup_features(home, date, home, away)
        al = self._lineup_features(away, date, home, away)
        if hl or al:
            out["lineup_data_available"] = 1.0
            for side, f in [("home", hl), ("away", al)]:
                out[f"{side}_xi_xg90"] = f.get("xg90", np.nan)
                out[f"{side}_xi_xa90"] = f.get("xa90", np.nan)
                out[f"{side}_xi_minutes"] = f.get("minutes", np.nan)
                out[f"{side}_xi_known"] = f.get("known", 0.0)
                out[f"{side}_xi_completeness"] = f.get("completeness", np.nan)
                out[f"{side}_xi_attack_delta_pct"] = f.get("attack_delta_pct", np.nan)
                out[f"{side}_gk_goals_prevented90"] = f.get("gk", np.nan)
                out[f"{side}_gk_delta"] = f.get("gk_delta", np.nan)

        ha = self._availability_features(home, date, home, away)
        aa = self._availability_features(away, date, home, away)
        if ha or aa:
            out["availability_data_available"] = 1.0
            for side, f in [("home", ha), ("away", aa)]:
                out[f"{side}_injuries"] = f.get("injuries", 0.0)
                out[f"{side}_suspensions"] = f.get("suspensions", 0.0)
                out[f"{side}_unavailable_xg90"] = f.get("xg90", 0.0)
                out[f"{side}_unavailable_xa90"] = f.get("xa90", 0.0)
                out[f"{side}_unavailable_minutes"] = f.get("minutes", 0.0)
                out[f"{side}_unavailable_attack_share"] = f.get("attack_share", np.nan)

        ht, at = self._transfer_features(home, date), self._transfer_features(away, date)
        if ht or at:
            out["transfer_data_available"] = 1.0
            out["home_transfer_attack_delta"] = ht.get("attack", np.nan); out["away_transfer_attack_delta"] = at.get("attack", np.nan)
            out["home_transfer_minutes_delta"] = ht.get("minutes", np.nan); out["away_transfer_minutes_delta"] = at.get("minutes", np.nan)

        hm, am = self._manager_features(home, date), self._manager_features(away, date)
        if hm or am:
            out["manager_data_available"] = 1.0
            out["home_manager_days"] = hm.get("days", np.nan); out["away_manager_days"] = am.get("days", np.nan)
            out["home_new_manager_30d"] = hm.get("new30", 0.0); out["away_new_manager_30d"] = am.get("new30", 0.0)

        hs, ass = self._schedule_features(home, date), self._schedule_features(away, date)
        if hs or ass:
            out["schedule_data_available"] = 1.0
            for side, f in [("home", hs), ("away", ass)]:
                out[f"{side}_rest_days_ext"] = f.get("rest", np.nan)
                out[f"{side}_matches_7d_ext"] = f.get("m7", np.nan)
                out[f"{side}_matches_14d_ext"] = f.get("m14", np.nan)
                out[f"{side}_continental_7d"] = f.get("cont7", 0.0)
                out[f"{side}_next_continental_7d"] = f.get("next_cont7", 0.0)

        if home in self.location_map and away in self.location_map:
            out["location_data_available"] = 1.0
            out["home_travel_km"] = 0.0
            out["away_travel_km"] = haversine_km(*self.location_map[away], *self.location_map[home])

        w = self._weather_features(date, home, away)
        if w:
            out["weather_data_available"] = 1.0
            out["temperature_c"] = w.get("temp", np.nan); out["wind_kph"] = w.get("wind", np.nan)
            out["precip_mm"] = w.get("precip", np.nan); out["humidity_pct"] = w.get("humidity", np.nan)

        mk = self._market_features(date, home, away)
        if mk:
            out["market_data_available"] = 1.0
            if any(k.startswith("exchange_") for k in mk):
                out["exchange_data_available"] = 1.0
            for dst, src in [
                ("sharp_home", "sharp_home"), ("sharp_draw", "sharp_draw"), ("sharp_away", "sharp_away"),
                ("open_home", "open_home"), ("open_draw", "open_draw"), ("open_away", "open_away"),
                ("market_move_home", "move_home"), ("market_move_draw", "move_draw"), ("market_move_away", "move_away"),
                ("exchange_home", "exchange_home"), ("exchange_draw", "exchange_draw"), ("exchange_away", "exchange_away"),
                ("exchange_liquidity_log", "liq_log"),
            ]:
                out[dst] = mk.get(src, np.nan)

        htac, atac = self._tactical_features(home, date), self._tactical_features(away, date)
        if htac or atac:
            out["tactics_data_available"] = 1.0
            for side, f in [("home", htac), ("away", atac)]:
                for name in ["possession", "ppda", "field_tilt", "xt", "xt_against"]:
                    out[f"{side}_{name}"] = f.get(name, np.nan)
        return out
