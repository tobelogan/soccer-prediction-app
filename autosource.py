from __future__ import annotations

"""Automatic enrichment of the predictor's optional advanced-data directory.

The design intentionally prefers structured sources over web/news text:

1. StatsBomb Open Data (free, historical event xG + actual starting XIs where covered).
2. Sportmonks API (token required; current/historical lineups, sidelined players,
   fixture xG, weather and pre-match news depending on subscription).
3. Google News RSS as a conservative *evidence fallback* for current
   availability. Only high-confidence phrases are promoted to availability.csv.

Nothing in this module bypasses paywalls, logins or provider licensing. Opta and
commercial Wyscout feeds must be accessed with the user's licensed API/export.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests


STATSBOMB_BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"
SPORTMONKS_BASE = "https://api.sportmonks.com/v3/football"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Public Wyscout research release. This is an old open dataset, not the live
# commercial Wyscout service and it does not contain provider xG values.
WYSCOUT_OPEN_FILES = {
    "players.json": "https://ndownloader.figshare.com/files/15073721",
    "teams.json": "https://ndownloader.figshare.com/files/15073697",
    "matches.zip": "https://ndownloader.figshare.com/files/14464622",
    "events.zip": "https://ndownloader.figshare.com/files/14464685",
}

_ALIAS = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "wolves": "wolverhampton wanderers",
    "nottm forest": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "brighton": "brighton and hove albion",
    "psg": "paris saint germain",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "ath madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    "bayern munich": "bayern munchen",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_team(value: Any) -> str:
    s = html.unescape(str(value or "")).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"\b(fc|afc|cf|sc|calcio|club de futbol|football club)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIAS.get(s, s)


def team_similarity(a: Any, b: Any) -> float:
    aa, bb = norm_team(a), norm_team(b)
    if aa == bb:
        return 1.0
    if aa and bb and (aa in bb or bb in aa):
        return 0.93
    return SequenceMatcher(None, aa, bb).ratio()


def teams_match(a: Any, b: Any, threshold: float = 0.78) -> bool:
    return team_similarity(a, b) >= threshold


class CachedHTTP:
    def __init__(self, cache_dir: str | Path, timeout: int = 30, user_agent: str = "SoccerPredictorAutoSource/4.0"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json,text/xml,text/plain,*/*"})

    @staticmethod
    def _cache_name(url: str, params: dict[str, Any] | None) -> str:
        raw = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, url: str, params: dict[str, Any] | None, suffix: str) -> Path:
        return self.cache_dir / f"{self._cache_name(url, params)}{suffix}"

    def get_bytes(self, url: str, params: dict[str, Any] | None = None, ttl_hours: float | None = None) -> bytes:
        path = self._path(url, params, ".bin")
        if path.exists() and ttl_hours is not None:
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            if age_h <= ttl_hours:
                return path.read_bytes()
        elif path.exists() and ttl_hours is None:
            return path.read_bytes()
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return r.content

    def get_json(self, url: str, params: dict[str, Any] | None = None, ttl_hours: float | None = None) -> Any:
        return json.loads(self.get_bytes(url, params=params, ttl_hours=ttl_hours).decode("utf-8"))


@dataclass
class SourceReport:
    provider: str
    matched_fixtures: int = 0
    rows_written: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_TABLE_KEYS = {
    "events": ["MatchID", "Team", "EventID"],
    "player_stats": ["Date", "MatchID", "Team", "Player"],
    "lineups": ["Date", "HomeTeam", "AwayTeam", "Team", "Player"],
    "availability": ["Date", "HomeTeam", "AwayTeam", "Team", "Player", "Status", "Reason"],
    "weather": ["Date", "HomeTeam", "AwayTeam"],
    "news_evidence": ["Date", "HomeTeam", "AwayTeam", "Team", "Title", "SourceURL"],
    "schedule": ["Date", "Team", "Opponent", "Competition"],
    "markets": ["Date", "HomeTeam", "AwayTeam"],
    "managers": ["Date", "Team", "Manager"],
    "transfers": ["Date", "Team", "Player", "Direction"],
    "team_locations": ["Team"],
    "tactics": ["Date", "Team"],
}


def _append_table(out_dir: Path, stem: str, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    new = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        try:
            old = pd.read_csv(path, on_bad_lines="skip")
            frame = pd.concat([old, new], ignore_index=True, sort=False)
        except Exception:
            frame = new
    else:
        frame = new
    keys = [c for c in _TABLE_KEYS.get(stem, []) if c in frame.columns]
    if stem == "events" and "EventID" in frame.columns:
        # Never collapse a user's event-level rows merely because they predate
        # this auto-source module and have no EventID column/value. Deduplicate
        # only rows that carry a provider-stable EventID.
        identified = frame[frame["EventID"].notna() & (frame["EventID"].astype(str) != "")].copy()
        unidentified = frame[~(frame["EventID"].notna() & (frame["EventID"].astype(str) != ""))].copy()
        if keys:
            identified = identified.drop_duplicates(keys, keep="last")
        frame = pd.concat([unidentified, identified], ignore_index=True, sort=False)
    elif keys:
        frame = frame.drop_duplicates(keys, keep="last")
    frame.to_csv(path, index=False)
    return len(new)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "confirmed", "available"}


def _find_nested(obj: Any, keys: set[str]) -> Any:
    """Return first nested value whose normalized key is in keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = re.sub(r"[^a-z0-9]", "", str(k).lower())
            if nk in keys:
                return v
        for v in obj.values():
            found = _find_nested(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_nested(v, keys)
            if found is not None:
                return found
    return None


class FootballDataBackfillSource:
    """Turn already-downloaded Football-Data web rows into advanced tables.

    This is the first backfill layer because it is timestamped, reproducible and
    available for the exact historical matches being trained. It fills schedule
    context and market snapshots whenever the source CSV contains those fields.
    """

    @staticmethod
    def _first_odds(row: pd.Series, candidates: list[tuple[str, str, str]]) -> tuple[float, float, float] | None:
        for cols in candidates:
            if not all(c in row.index for c in cols):
                continue
            try:
                vals = tuple(float(row[c]) for c in cols)
            except Exception:
                continue
            if all(np.isfinite(v) and v > 1.0 for v in vals):
                return vals
        return None

    def sync(self, base: pd.DataFrame, out_dir: str | Path) -> SourceReport:
        report = SourceReport("football_data_web_backfill")
        if base is None or base.empty:
            report.notes.append("No base match/fixture rows supplied.")
            return report
        out_dir = Path(out_dir)
        schedule_rows: list[dict[str, Any]] = []
        market_rows: list[dict[str, Any]] = []
        fetched = utc_now_iso()
        for _, r in base.iterrows():
            if pd.isna(r.get("Date")) or not r.get("HomeTeam") or not r.get("AwayTeam"):
                continue
            date = pd.Timestamp(r["Date"]).date().isoformat()
            home, away = str(r["HomeTeam"]), str(r["AwayTeam"])
            comp = str(r.get("League", "League"))
            schedule_rows.extend([
                {"Date": date, "Team": home, "Opponent": away, "Competition": comp, "IsContinental": False, "Source": "Football-Data", "FetchedAt": fetched},
                {"Date": date, "Team": away, "Opponent": home, "Competition": comp, "IsContinental": False, "Source": "Football-Data", "FetchedAt": fetched},
            ])

            # Prefer explicit opening/closing columns when Football-Data supplies
            # them; otherwise preserve the ordinary market as the snapshot.
            opening = self._first_odds(r, [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("MaxH", "MaxD", "MaxA")])
            closing = self._first_odds(r, [("B365CH", "B365CD", "B365CA"), ("AvgCH", "AvgCD", "AvgCA"), ("MaxCH", "MaxCD", "MaxCA"), ("AvgH", "AvgD", "AvgA")])
            sharp = self._first_odds(r, [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")])
            if opening or closing or sharp:
                m = {"Date": date, "HomeTeam": home, "AwayTeam": away, "Source": "Football-Data", "FetchedAt": fetched, "SourceConfidence": 0.95}
                if opening:
                    m.update(dict(zip(("OpenH", "OpenD", "OpenA"), opening)))
                if closing:
                    m.update(dict(zip(("CloseH", "CloseD", "CloseA"), closing)))
                if sharp:
                    m.update(dict(zip(("SharpH", "SharpD", "SharpA"), sharp)))
                market_rows.append(m)

        report.rows_written["schedule"] = _append_table(out_dir, "schedule", schedule_rows)
        report.rows_written["markets"] = _append_table(out_dir, "markets", market_rows)
        report.matched_fixtures = int(len(base))
        if not market_rows:
            report.notes.append("No usable 1X2 odds columns were present for market backfill.")
        return report


class StatsBombOpenSource:
    """Fetch the official Hudl/StatsBomb open-data repository."""

    def __init__(self, http: CachedHTTP):
        self.http = http

    def _json(self, rel: str) -> Any:
        return self.http.get_json(f"{STATSBOMB_BASE}/{rel}", ttl_hours=None)

    @staticmethod
    def _base_candidates(base: pd.DataFrame) -> dict[Any, list[pd.Series]]:
        b = base.copy()
        b["Date"] = pd.to_datetime(b["Date"], errors="coerce", dayfirst=True)
        out: dict[Any, list[pd.Series]] = {}
        for _, r in b.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).iterrows():
            out.setdefault(r["Date"].date(), []).append(r)
        return out

    def sync(self, base_matches: pd.DataFrame, out_dir: str | Path, max_matches: int | None = None) -> SourceReport:
        report = SourceReport("statsbomb_open")
        out_dir = Path(out_dir)
        if base_matches is None or base_matches.empty:
            report.notes.append("No base matches supplied.")
            return report
        base_map = self._base_candidates(base_matches)
        date_min = pd.to_datetime(base_matches["Date"], errors="coerce", dayfirst=True).min()
        date_max = pd.to_datetime(base_matches["Date"], errors="coerce", dayfirst=True).max()
        if pd.isna(date_min) or pd.isna(date_max):
            report.notes.append("Base match dates could not be parsed.")
            return report

        competitions = self._json("competitions.json")
        matched: list[tuple[dict[str, Any], pd.Series]] = []
        base_years = set(range(int(date_min.year) - 1, int(date_max.year) + 2))
        for comp in competitions:
            # Skip open-data seasons that cannot overlap the requested history.
            # season_name is commonly '2018/2019', '2022', etc.
            season_years = {int(y) for y in re.findall(r"(?:19|20)\d{2}", str(comp.get("season_name", "")))}
            if season_years and season_years.isdisjoint(base_years):
                continue
            try:
                ms = self._json(f"matches/{comp['competition_id']}/{comp['season_id']}.json")
            except Exception as exc:
                report.notes.append(f"Skipped competition {comp.get('competition_id')}/{comp.get('season_id')}: {exc}")
                continue
            for m in ms:
                md = pd.to_datetime(m.get("match_date"), errors="coerce")
                if pd.isna(md) or md < date_min.normalize() - pd.Timedelta(days=1) or md > date_max.normalize() + pd.Timedelta(days=1):
                    continue
                candidates = base_map.get(md.date(), [])
                sh = (m.get("home_team") or {}).get("home_team_name", "")
                sa = (m.get("away_team") or {}).get("away_team_name", "")
                best = None
                best_score = 0.0
                for r in candidates:
                    score = (team_similarity(sh, r["HomeTeam"]) + team_similarity(sa, r["AwayTeam"])) / 2
                    if score > best_score:
                        best, best_score = r, score
                if best is not None and best_score >= 0.78:
                    matched.append((m, best))
                    if max_matches and len(matched) >= max_matches:
                        break
            if max_matches and len(matched) >= max_matches:
                break

        event_rows: list[dict[str, Any]] = []
        lineup_rows: list[dict[str, Any]] = []
        player_rows: list[dict[str, Any]] = []
        tactic_rows: list[dict[str, Any]] = []
        fetched_at = utc_now_iso()

        for m, base in matched:
            match_id = m.get("match_id")
            if match_id is None:
                continue
            try:
                events = self._json(f"events/{match_id}.json")
            except Exception as exc:
                report.notes.append(f"StatsBomb events unavailable for {match_id}: {exc}")
                continue
            date = pd.Timestamp(base["Date"]).date().isoformat()
            home, away = str(base["HomeTeam"]), str(base["AwayTeam"])
            source_home = (m.get("home_team") or {}).get("home_team_name", "")
            source_away = (m.get("away_team") or {}).get("away_team_name", "")
            source_to_base = {norm_team(source_home): home, norm_team(source_away): away}

            shot_xg: dict[str, float] = {}
            # First pass lets xA link assisted passes to their shot's xG.
            for e in events:
                if (e.get("type") or {}).get("name") == "Shot":
                    xg = float((e.get("shot") or {}).get("statsbomb_xg") or 0.0)
                    if e.get("id"):
                        shot_xg[str(e["id"])] = xg

            player_acc: dict[tuple[str, str], dict[str, float]] = {}
            starters: dict[str, dict[str, str]] = {}
            sub_in: dict[tuple[str, str], float] = {}
            sub_out: dict[tuple[str, str], float] = {}
            max_min = 90.0

            for e in events:
                minute = float(e.get("minute") or 0.0)
                max_min = max(max_min, minute)
                source_team = (e.get("team") or {}).get("name", "")
                team = source_to_base.get(norm_team(source_team), source_team)
                etype = (e.get("type") or {}).get("name", "")
                player = (e.get("player") or {}).get("name", "")

                if etype == "Starting XI":
                    lineup = ((e.get("tactics") or {}).get("lineup") or [])
                    team_starters: dict[str, str] = {}
                    for p in lineup:
                        pname = (p.get("player") or {}).get("name", "")
                        role = (p.get("position") or {}).get("name", "")
                        if pname:
                            team_starters[pname] = role
                            lineup_rows.append({
                                "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": team,
                                "Player": pname, "IsStarter": True, "Confirmed": True,
                                "Role": "GK" if "goalkeeper" in role.lower() else role,
                                "Source": "StatsBomb Open Data", "SourceURL": f"{STATSBOMB_BASE}/events/{match_id}.json",
                                "FetchedAt": fetched_at, "SourceConfidence": 1.0,
                            })
                    starters[team] = team_starters

                if etype == "Substitution" and player:
                    sub_out[(team, player)] = minute
                    repl = ((e.get("substitution") or {}).get("replacement") or {}).get("name", "")
                    if repl:
                        sub_in[(team, repl)] = minute

                if not player:
                    continue
                key = (team, player)
                acc = player_acc.setdefault(key, {"xG": 0.0, "xA": 0.0})
                if etype == "Shot":
                    shot = e.get("shot") or {}
                    xg = float(shot.get("statsbomb_xg") or 0.0)
                    penalty = str((shot.get("type") or {}).get("name", "")).lower() == "penalty"
                    opp = away if team == home else home if team == away else ""
                    event_rows.append({
                        "MatchID": match_id, "EventID": f"statsbomb:{e.get('id', '')}", "Date": date, "Team": team, "Opponent": opp,
                        "Player": player, "xG": xg, "npxG": 0.0 if penalty else xg,
                        "BigChance": np.nan, "Source": "StatsBomb Open Data",
                        "SourceURL": f"{STATSBOMB_BASE}/events/{match_id}.json", "FetchedAt": fetched_at,
                        "SourceConfidence": 1.0,
                    })
                    acc["xG"] += xg
                elif etype == "Pass":
                    p = e.get("pass") or {}
                    assisted = p.get("assisted_shot_id")
                    if assisted is not None:
                        acc["xA"] += float(shot_xg.get(str(assisted), 0.0))

            # Event-derived tactical proxies. These are deliberately labelled
            # as proxies rather than proprietary provider metrics. Possession is
            # completed-pass share; PPDA and field tilt use StatsBomb locations.
            team_stats: dict[str, dict[str, float]] = {home: {"passes": 0, "opp60_passes": 0, "def_actions": 0, "final_third": 0}, away: {"passes": 0, "opp60_passes": 0, "def_actions": 0, "final_third": 0}}
            for e in events:
                source_team = (e.get("team") or {}).get("name", "")
                team = source_to_base.get(norm_team(source_team), source_team)
                if team not in team_stats:
                    continue
                etype = (e.get("type") or {}).get("name", "")
                loc = e.get("location") or []
                xloc = float(loc[0]) if loc and isinstance(loc[0], (int, float)) else np.nan
                if etype == "Pass":
                    outcome = str(((e.get("pass") or {}).get("outcome") or {}).get("name", ""))
                    if not outcome:  # StatsBomb omits outcome for completed passes.
                        team_stats[team]["passes"] += 1
                    if np.isfinite(xloc) and xloc >= 80:
                        team_stats[team]["final_third"] += 1
                elif etype in {"Pressure", "Duel", "Interception", "Foul Committed", "Ball Recovery", "Block"}:
                    if np.isfinite(xloc) and xloc >= 48:
                        team_stats[team]["def_actions"] += 1
            for team, opp in [(home, away), (away, home)]:
                # Opponent passes in the pressing zone approximate passes allowed
                # before a defensive action (PPDA denominator/numerator pairing).
                opp_passes = 0
                for e in events:
                    source_team = (e.get("team") or {}).get("name", "")
                    mapped = source_to_base.get(norm_team(source_team), source_team)
                    if mapped != opp or (e.get("type") or {}).get("name", "") != "Pass":
                        continue
                    loc = e.get("location") or []
                    xloc = float(loc[0]) if loc and isinstance(loc[0], (int, float)) else np.nan
                    if np.isfinite(xloc) and xloc <= 72:
                        opp_passes += 1
                team_stats[team]["opp60_passes"] = opp_passes
            total_passes = sum(v["passes"] for v in team_stats.values())
            total_ft = sum(v["final_third"] for v in team_stats.values())
            for team in [home, away]:
                st = team_stats[team]
                possession = 100.0 * st["passes"] / total_passes if total_passes else np.nan
                field_tilt = 100.0 * st["final_third"] / total_ft if total_ft else np.nan
                ppda = st["opp60_passes"] / max(st["def_actions"], 1.0) if st["opp60_passes"] else np.nan
                tactic_rows.append({
                    "Date": date, "Team": team, "Possession": possession, "PPDA": ppda,
                    "FieldTilt": field_tilt, "xT": np.nan, "xTAgainst": np.nan,
                    "Source": "StatsBomb Open Data tactical proxy", "FetchedAt": fetched_at,
                    "SourceConfidence": 0.85,
                })

            end_min = max(90.0, max_min)
            home_score = float(m.get("home_score") or 0.0)
            away_score = float(m.get("away_score") or 0.0)
            all_players = set(player_acc)
            for team, ps in starters.items():
                for pname in ps:
                    all_players.add((team, pname))
            for key in sub_in:
                all_players.add(key)

            for team, player in all_players:
                start = 0.0 if player in starters.get(team, {}) else sub_in.get((team, player), end_min)
                end = sub_out.get((team, player), end_min)
                mins = max(0.0, min(end_min, end) - min(end_min, start))
                role_name = starters.get(team, {}).get(player, "")
                role = "GK" if "goalkeeper" in role_name.lower() else role_name
                goals_allowed = np.nan
                if role == "GK":
                    goals_allowed = away_score if team == home else home_score if team == away else np.nan
                acc = player_acc.get((team, player), {"xG": 0.0, "xA": 0.0})
                player_rows.append({
                    "Date": date, "MatchID": match_id, "Team": team, "Player": player,
                    "Minutes": mins, "xG": acc["xG"], "xA": acc["xA"],
                    "PSxG": np.nan, "GoalsAllowed": goals_allowed, "Role": role,
                    "Source": "StatsBomb Open Data", "FetchedAt": fetched_at, "SourceConfidence": 1.0,
                })

        report.matched_fixtures = len(matched)
        for stem, rows in [("events", event_rows), ("lineups", lineup_rows), ("player_stats", player_rows), ("tactics", tactic_rows)]:
            report.rows_written[stem] = _append_table(out_dir, stem, rows)
        if not matched:
            report.notes.append("No base fixtures overlapped the competitions/seasons present in StatsBomb Open Data.")
        return report


class DerivedPlayerMovementSource:
    """Infer a small, auditable transfer table from observed player team changes.

    This is not a replacement for a licensed transfer feed. It only records a
    change when the same player is observed for two different teams in the
    auto-sourced player_stats table in chronological order.
    """

    def sync(self, out_dir: str | Path) -> SourceReport:
        report = SourceReport("derived_player_movements")
        out_dir = Path(out_dir)
        path = out_dir / "player_stats.csv"
        if not path.exists() or path.stat().st_size == 0:
            report.notes.append("No player_stats.csv available for transfer inference.")
            return report
        try:
            p = pd.read_csv(path, on_bad_lines="skip")
        except Exception as exc:
            report.notes.append(f"Could not read player_stats.csv: {exc}")
            return report
        if not {"Date", "Team", "Player"}.issubset(p.columns):
            return report
        p["Date"] = pd.to_datetime(p["Date"], errors="coerce", dayfirst=True)
        p = p.dropna(subset=["Date", "Team", "Player"]).sort_values(["Player", "Date"])
        rows: list[dict[str, Any]] = []
        fetched = utc_now_iso()
        for player, g in p.groupby("Player", sort=False):
            team_runs = g.loc[g["Team"].astype(str).ne(g["Team"].astype(str).shift())].copy()
            if len(team_runs) < 2:
                continue
            vals = list(team_runs.itertuples(index=False))
            for prev, cur in zip(vals, vals[1:]):
                prev_team, cur_team = str(prev.Team), str(cur.Team)
                if prev_team == cur_team:
                    continue
                date = pd.Timestamp(cur.Date).date().isoformat()
                rows.append({"Date": date, "Team": prev_team, "Player": str(player), "Direction": "OUT", "Source": "Derived from observed player team change", "FetchedAt": fetched, "SourceConfidence": 0.70})
                rows.append({"Date": date, "Team": cur_team, "Player": str(player), "Direction": "IN", "Source": "Derived from observed player team change", "FetchedAt": fetched, "SourceConfidence": 0.70})
        report.rows_written["transfers"] = _append_table(out_dir, "transfers", rows)
        return report


class OpenMeteoHistoricalWeatherSource:
    """Backfill team locations and historical match-day weather from open web APIs.

    Nominatim is queried conservatively and cached; Open-Meteo then retrieves a
    date range per home team, avoiding one HTTP request per match.
    """

    def __init__(self, http: CachedHTTP, geocode_delay_seconds: float = 1.05):
        self.http = http
        self.geocode_delay_seconds = float(geocode_delay_seconds)

    def _geocode(self, team: str) -> tuple[float, float] | None:
        params = {"q": f"{team} football stadium", "format": "jsonv2", "limit": 1}
        cache_path = self.http._path(NOMINATIM_SEARCH, params, ".bin")
        if not cache_path.exists():
            time.sleep(self.geocode_delay_seconds)
        try:
            data = self.http.get_json(NOMINATIM_SEARCH, params=params, ttl_hours=None)
        except Exception:
            return None
        if not data:
            return None
        try:
            return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            return None

    def sync(self, matches: pd.DataFrame, out_dir: str | Path, max_teams: int | None = None) -> SourceReport:
        report = SourceReport("open_meteo_historical_weather")
        out_dir = Path(out_dir)
        x = matches.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce", format="mixed", dayfirst=True)
        x = x.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        if x.empty:
            return report
        teams = list(dict.fromkeys(x["HomeTeam"].astype(str).tolist()))
        if max_teams:
            teams = teams[: int(max_teams)]
        fetched = utc_now_iso()
        location_rows: list[dict[str, Any]] = []
        weather_rows: list[dict[str, Any]] = []
        for team in teams:
            geo = self._geocode(team)
            if not geo:
                report.notes.append(f"Location not resolved for {team}")
                continue
            lat, lon = geo
            location_rows.append({"Team": team, "Lat": lat, "Lon": lon, "Source": "OpenStreetMap Nominatim", "FetchedAt": fetched, "SourceConfidence": 0.80})
            g = x[x["HomeTeam"].astype(str) == team].copy()
            start, end = g["Date"].min().date().isoformat(), g["Date"].max().date().isoformat()
            params = {
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
                "timezone": "auto", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
            }
            try:
                payload = self.http.get_json(OPEN_METEO_ARCHIVE, params=params, ttl_hours=None)
                daily = payload.get("daily") or {}
                lookup = {}
                for i, day in enumerate(daily.get("time") or []):
                    lookup[str(day)] = {
                        "TempC": (daily.get("temperature_2m_mean") or [np.nan] * (i + 1))[i],
                        "PrecipMm": (daily.get("precipitation_sum") or [np.nan] * (i + 1))[i],
                        "WindKph": (daily.get("wind_speed_10m_max") or [np.nan] * (i + 1))[i],
                    }
            except Exception as exc:
                report.notes.append(f"Weather failed for {team}: {exc}")
                continue
            for _, r in g.iterrows():
                day = pd.Timestamp(r["Date"]).date().isoformat()
                vals = lookup.get(day)
                if not vals:
                    continue
                weather_rows.append({
                    "Date": day, "HomeTeam": str(r["HomeTeam"]), "AwayTeam": str(r["AwayTeam"]),
                    **vals, "HumidityPct": np.nan, "Source": "Open-Meteo Historical Weather API",
                    "FetchedAt": fetched, "SourceConfidence": 0.90,
                })
        report.rows_written["team_locations"] = _append_table(out_dir, "team_locations", location_rows)
        report.rows_written["weather"] = _append_table(out_dir, "weather", weather_rows)
        report.matched_fixtures = int(len(weather_rows))
        return report


class HistoricalNewsContextSource:
    """Conservative historical news backfill for absences, managers and transfers.

    Queries are grouped by team-season, not match, and every promoted row must
    have a publication timestamp no later than the historical match/event date.
    Search archives can be incomplete, so zero rows are a valid outcome.
    """

    MANAGER_PATTERNS = [
        re.compile(r"(?:appoints?|appointed|names?|named|hires?|hired)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s+(?:as\s+)?(?:manager|head coach)", re.I),
        re.compile(r"([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s+(?:appointed|named|hired)\s+(?:as\s+)?(?:manager|head coach)", re.I),
    ]

    def __init__(self, http: CachedHTTP):
        self.http = http
        self.current = GoogleNewsAvailabilitySource(http, min_promote_confidence=0.72)

    def _rss(self, query: str) -> list[dict[str, str]]:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        raw = self.http.get_bytes(GOOGLE_NEWS_RSS, params=params, ttl_hours=None)
        root = ET.fromstring(raw)
        out = []
        for item in root.findall(".//item"):
            def txt(tag: str) -> str:
                node = item.find(tag)
                return "" if node is None or node.text is None else html.unescape(node.text)
            out.append({"title": txt("title"), "description": re.sub("<[^>]+>", " ", txt("description")), "link": txt("link"), "pubDate": txt("pubDate")})
        return out

    def sync(self, matches: pd.DataFrame, out_dir: str | Path, max_queries: int = 120) -> SourceReport:
        report = SourceReport("historical_news_context")
        out_dir = Path(out_dir)
        x = matches.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce", format="mixed", dayfirst=True)
        x = x.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        known = GoogleNewsAvailabilitySource._known_players(out_dir)
        if x.empty:
            return report
        team_dates: dict[str, list[pd.Timestamp]] = {}
        for team in pd.unique(pd.concat([x["HomeTeam"], x["AwayTeam"]], ignore_index=True).astype(str)):
            dates = x[(x["HomeTeam"].astype(str) == team) | (x["AwayTeam"].astype(str) == team)]["Date"].sort_values().tolist()
            team_dates[team] = dates
        seasons: list[tuple[str, int]] = []
        for team, dates in team_dates.items():
            for d in dates:
                sy = d.year if d.month >= 7 else d.year - 1
                key = (team, sy)
                if key not in seasons:
                    seasons.append(key)
        seasons = seasons[-int(max_queries):]
        availability: list[dict[str, Any]] = []
        managers: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        fetched = utc_now_iso()
        for team, sy in seasons:
            start = pd.Timestamp(year=sy, month=7, day=1)
            end = pd.Timestamp(year=sy + 1, month=7, day=1)
            q = f'"{team}" football (injury OR suspended OR manager OR "head coach" OR transfer OR signed OR joins OR leaves) after:{start.date()} before:{end.date()}'
            try:
                articles = self._rss(q)
            except Exception as exc:
                report.notes.append(f"Historical news failed for {team} {sy}: {exc}")
                continue
            players = sorted(known.get(norm_team(team), set()), key=len, reverse=True)
            for a in articles:
                try:
                    published = parsedate_to_datetime(a.get("pubDate", ""))
                    if published.tzinfo is not None:
                        published = published.astimezone(timezone.utc).replace(tzinfo=None)
                    pub = pd.Timestamp(published)
                except Exception:
                    continue
                if not (start <= pub < end):
                    continue
                text = f"{a.get('title','')} {a.get('description','')}"
                title = a.get("title", "")
                # Manager promotions require explicit appointment language.
                for pat in self.MANAGER_PATTERNS:
                    mm = pat.search(title)
                    if mm:
                        managers.append({"Date": pub.date().isoformat(), "Team": team, "Manager": mm.group(1).strip(), "Source": "Google News historical RSS", "SourceURL": a.get("link", ""), "PublishedAt": a.get("pubDate", ""), "FetchedAt": fetched, "SourceConfidence": 0.72})
                        break
                matched_player = next((p for p in players if len(p) >= 5 and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, flags=re.I)), None)
                if not matched_player:
                    continue
                status, reason, conf = self.current._classify(text)
                if status and conf >= 0.72:
                    future = [d for d in team_dates.get(team, []) if pub <= d <= pub + pd.Timedelta(days=21)]
                    for md in future:
                        mr = x[(x["Date"] == md) & ((x["HomeTeam"].astype(str) == team) | (x["AwayTeam"].astype(str) == team))].iloc[0]
                        availability.append({"Date": md.date().isoformat(), "HomeTeam": str(mr["HomeTeam"]), "AwayTeam": str(mr["AwayTeam"]), "Team": team, "Player": matched_player, "Status": status, "Reason": reason, "Source": "Google News historical RSS", "SourceURL": a.get("link", ""), "PublishedAt": a.get("pubDate", ""), "FetchedAt": fetched, "SourceConfidence": min(conf, 0.80)})
                low = text.lower()
                direction = None
                if re.search(r"\b(signs?|signed|joins?|joined|arrival)\b", low):
                    direction = "IN"
                if re.search(r"\b(leaves?|left|sold|departs?|departed|exit)\b", low):
                    direction = "OUT"
                if direction:
                    transfers.append({"Date": pub.date().isoformat(), "Team": team, "Player": matched_player, "Direction": direction, "Source": "Google News historical RSS", "SourceURL": a.get("link", ""), "PublishedAt": a.get("pubDate", ""), "FetchedAt": fetched, "SourceConfidence": 0.62})
        report.rows_written["availability"] = _append_table(out_dir, "availability", availability)
        report.rows_written["managers"] = _append_table(out_dir, "managers", managers)
        report.rows_written["transfers"] = _append_table(out_dir, "transfers", transfers)
        return report


class SportmonksSource:
    """Structured current/historical enrichment using a user's Sportmonks token."""

    def __init__(self, http: CachedHTTP, token: str | None = None):
        self.http = http
        self.token = token or os.getenv("SPORTMONKS_API_TOKEN")
        if not self.token:
            raise ValueError("SPORTMONKS_API_TOKEN is not set")

    def _get(self, path: str, *, include: str | None = None, ttl_hours: float = 6.0) -> Any:
        params: dict[str, Any] = {"api_token": self.token}
        if include:
            params["include"] = include
        return self.http.get_json(f"{SPORTMONKS_BASE}/{path.lstrip('/')}", params=params, ttl_hours=ttl_hours)

    @staticmethod
    def _participants(f: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        home = away = None
        parts = f.get("participants") or []
        for p in parts:
            loc = str((p.get("meta") or {}).get("location") or p.get("location") or (p.get("pivot") or {}).get("location") or "").lower()
            if loc == "home":
                home = p
            elif loc == "away":
                away = p
        if home is None and len(parts) >= 2:
            home, away = parts[0], parts[1]
        return home, away

    def _discover(self, fixtures: pd.DataFrame) -> list[tuple[pd.Series, dict[str, Any]]]:
        x = fixtures.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce", format="mixed", dayfirst=True)
        x = x.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        if x.empty:
            return []
        start, end = x["Date"].min().normalize(), x["Date"].max().normalize()
        provider_fixtures: list[dict[str, Any]] = []
        cur = start
        while cur <= end:
            stop = min(cur + pd.Timedelta(days=89), end)
            payload = self._get(
                f"fixtures/between/{cur.date().isoformat()}/{stop.date().isoformat()}",
                include="participants",
                ttl_hours=6.0 if stop >= pd.Timestamp.today().normalize() - pd.Timedelta(days=2) else 24 * 30,
            )
            provider_fixtures.extend(payload.get("data") or [])
            cur = stop + pd.Timedelta(days=1)

        out: list[tuple[pd.Series, dict[str, Any]]] = []
        for _, row in x.iterrows():
            day = row["Date"].date()
            best = None
            best_score = 0.0
            for f in provider_fixtures:
                fd = pd.to_datetime(f.get("starting_at"), errors="coerce")
                if pd.isna(fd) or abs((fd.date() - day).days) > 1:
                    continue
                hp, ap = self._participants(f)
                if not hp or not ap:
                    continue
                score = (team_similarity(hp.get("name"), row["HomeTeam"]) + team_similarity(ap.get("name"), row["AwayTeam"])) / 2
                if score > best_score:
                    best, best_score = f, score
            if best is not None and best_score >= 0.76:
                out.append((row, best))
        return out

    @staticmethod
    def _lineup_confirmed(f: dict[str, Any]) -> bool:
        meta = f.get("metadata") or []
        raw = json.dumps(meta).lower()
        if "lineup_confirmed" not in raw:
            # Completed fixtures are necessarily historical actual lineups.
            state = str((f.get("state") or {}).get("state") or f.get("result_info") or "").lower()
            return any(k in state for k in ["finished", "full", "won", "draw", "after full-time"])
        # Try common metadata structures.
        for item in meta if isinstance(meta, list) else [meta]:
            txt = json.dumps(item).lower()
            if "lineup_confirmed" in txt:
                val = _find_nested(item, {"value", "data", "lineupconfirmed"})
                if isinstance(val, dict):
                    val = _find_nested(val, {"value", "confirmed"})
                return _truthy(val)
        return False

    def sync(self, fixtures: pd.DataFrame, out_dir: str | Path, max_fixtures: int | None = None) -> SourceReport:
        report = SourceReport("sportmonks")
        out_dir = Path(out_dir)
        discovered = self._discover(fixtures)
        if max_fixtures:
            discovered = discovered[: int(max_fixtures)]
        fetched_at = utc_now_iso()
        lineup_rows: list[dict[str, Any]] = []
        availability_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        weather_rows: list[dict[str, Any]] = []
        news_rows: list[dict[str, Any]] = []

        includes = ";".join([
            "participants", "lineups.player", "lineups.position", "sidelined.player",
            "sidelined.type", "sidelined.sideline", "metadata", "weatherReport",
            "xGFixture", "prematchNews",
        ])

        for base, brief in discovered:
            fixture_id = brief.get("id")
            if fixture_id is None:
                continue
            try:
                payload = self._get(f"fixtures/{fixture_id}", include=includes, ttl_hours=1.0)
            except Exception as exc:
                report.notes.append(f"Fixture {fixture_id} enrichment failed: {exc}")
                continue
            f = payload.get("data") or {}
            hp, ap = self._participants(f)
            date = pd.Timestamp(base["Date"]).isoformat()
            home, away = str(base["HomeTeam"]), str(base["AwayTeam"])
            id_to_base: dict[Any, str] = {}
            if hp:
                id_to_base[hp.get("id")] = home
            if ap:
                id_to_base[ap.get("id")] = away
            confirmed = self._lineup_confirmed(f)

            for ln in f.get("lineups") or []:
                tid = ln.get("team_id")
                team = id_to_base.get(tid)
                if not team:
                    continue
                player = ln.get("player_name") or (ln.get("player") or {}).get("name") or ""
                if not player:
                    continue
                pos = (ln.get("position") or {}).get("name") or ""
                lineup_rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": team,
                    "Player": player, "IsStarter": int(ln.get("type_id") or 0) == 11,
                    "Confirmed": confirmed, "Role": "GK" if "goalkeeper" in pos.lower() else pos,
                    "Source": "Sportmonks", "SourceURL": f"{SPORTMONKS_BASE}/fixtures/{fixture_id}",
                    "FetchedAt": fetched_at, "SourceConfidence": 1.0 if confirmed else 0.55,
                })

            for sd in f.get("sidelined") or []:
                tid = sd.get("team_id") or sd.get("participant_id")
                team = id_to_base.get(tid)
                player = (sd.get("player") or {}).get("name") or sd.get("player_name") or ""
                if not team or not player:
                    continue
                type_name = str((sd.get("type") or {}).get("name") or (sd.get("sideline") or {}).get("name") or "")
                reason = "suspension" if re.search(r"susp|ban|card", type_name, re.I) else "injury"
                availability_rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": team,
                    "Player": player, "Status": "suspended" if reason == "suspension" else "out",
                    "Reason": reason, "ExpectedMinutes": np.nan, "PlayerXG90": np.nan, "PlayerXA90": np.nan,
                    "Source": "Sportmonks", "SourceURL": f"{SPORTMONKS_BASE}/fixtures/{fixture_id}",
                    "FetchedAt": fetched_at, "SourceConfidence": 0.95,
                })

            # Sportmonks fixture xG is genuine provider xG but is team-match level, not per-shot.
            xg_items = f.get("xGFixture") or f.get("xgfixture") or []
            for item in xg_items:
                tid = item.get("participant_id") or item.get("team_id")
                team = id_to_base.get(tid)
                loc = str(item.get("location") or "").lower()
                if not team:
                    team = home if loc == "home" else away if loc == "away" else None
                if not team:
                    continue
                val = _find_nested(item.get("data") or item, {"value"})
                try:
                    xg = float(val)
                except Exception:
                    continue
                opp = away if team == home else home
                event_rows.append({
                    "MatchID": fixture_id, "EventID": f"sportmonks:{fixture_id}:{team}:xg_total", "Date": date, "Team": team, "Opponent": opp,
                    "Player": "__TEAM_TOTAL__", "xG": xg, "npxG": xg, "BigChance": np.nan,
                    "Source": "Sportmonks xGFixture", "SourceURL": f"{SPORTMONKS_BASE}/fixtures/{fixture_id}",
                    "FetchedAt": fetched_at, "SourceConfidence": 1.0, "SourceLevel": "team_fixture_xg",
                })

            wr = f.get("weatherReport") or f.get("weather_report")
            if wr:
                def num(keys: set[str]) -> float:
                    v = _find_nested(wr, keys)
                    if isinstance(v, dict):
                        v = _find_nested(v, {"value", "temp", "temperature"})
                    try:
                        return float(v)
                    except Exception:
                        return np.nan
                weather_rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away,
                    "TempC": num({"temperature", "tempc", "temp"}),
                    "WindKph": num({"windkph", "windspeed", "wind"}),
                    "PrecipMm": num({"precipmm", "precipitation", "rain"}),
                    "HumidityPct": num({"humidity", "humiditypct"}),
                    "Source": "Sportmonks", "SourceURL": f"{SPORTMONKS_BASE}/fixtures/{fixture_id}",
                    "FetchedAt": fetched_at, "SourceConfidence": 0.95,
                })

            for item in f.get("prematchNews") or f.get("prematch_news") or []:
                title = str(item.get("title") or item.get("name") or "")
                body = str(item.get("description") or item.get("body") or item.get("content") or "")
                news_rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": "",
                    "Title": title, "Description": body, "SourceURL": item.get("url") or item.get("link") or "",
                    "PublishedAt": item.get("published_at") or item.get("created_at") or "",
                    "Source": "Sportmonks prematchNews", "FetchedAt": fetched_at,
                })

        report.matched_fixtures = len(discovered)
        for stem, rows in [
            ("lineups", lineup_rows), ("availability", availability_rows), ("events", event_rows),
            ("weather", weather_rows), ("news_evidence", news_rows),
        ]:
            report.rows_written[stem] = _append_table(out_dir, stem, rows)
        return report


class GoogleNewsAvailabilitySource:
    """Conservative current-team-news fallback using Google News RSS.

    It only promotes evidence to availability.csv when a known player name is
    present and the language is strong enough (e.g. 'ruled out', 'will miss',
    'suspended'). Weaker reports are retained in news_evidence.csv only.
    """

    OUT_PATTERNS = [r"ruled out", r"will miss", r"set to miss", r"sidelined", r"out for", r"unavailable"]
    SUSP_PATTERNS = [r"suspended", r"suspension", r"serving a ban", r"banned"]
    DOUBT_PATTERNS = [r"doubt", r"doubtful", r"fitness test", r"late fitness"]
    FIT_PATTERNS = [r"fit again", r"available again", r"returns? to training", r"back in training", r"cleared to play"]

    def __init__(self, http: CachedHTTP, min_promote_confidence: float = 0.78):
        self.http = http
        self.min_promote_confidence = float(min_promote_confidence)

    @staticmethod
    def _known_players(out_dir: Path) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for stem in ["player_stats", "lineups"]:
            path = out_dir / f"{stem}.csv"
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(path, on_bad_lines="skip", usecols=lambda c: c in {"Team", "Player"})
            except Exception:
                continue
            if not {"Team", "Player"}.issubset(df.columns):
                continue
            for team, g in df.dropna().groupby("Team"):
                result.setdefault(norm_team(team), set()).update(str(x) for x in g["Player"].dropna().unique())
        return result

    @staticmethod
    def _classify(text: str) -> tuple[str | None, str | None, float]:
        low = text.lower()
        if any(re.search(p, low) for p in GoogleNewsAvailabilitySource.SUSP_PATTERNS):
            return "suspended", "suspension", 0.92
        if any(re.search(p, low) for p in GoogleNewsAvailabilitySource.OUT_PATTERNS):
            return "out", "injury", 0.84
        if any(re.search(p, low) for p in GoogleNewsAvailabilitySource.FIT_PATTERNS):
            return "available", "fitness", 0.74
        if any(re.search(p, low) for p in GoogleNewsAvailabilitySource.DOUBT_PATTERNS):
            return "doubtful", "injury", 0.58
        if re.search(r"injur|hamstring|ankle|knee|muscle problem|knock", low):
            return "doubtful", "injury", 0.50
        return None, None, 0.0

    def _rss(self, query: str) -> list[dict[str, str]]:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        raw = self.http.get_bytes(GOOGLE_NEWS_RSS, params=params, ttl_hours=1.0)
        root = ET.fromstring(raw)
        out = []
        for item in root.findall(".//item"):
            def txt(tag: str) -> str:
                node = item.find(tag)
                return "" if node is None or node.text is None else html.unescape(node.text)
            out.append({"title": txt("title"), "description": re.sub("<[^>]+>", " ", txt("description")), "link": txt("link"), "pubDate": txt("pubDate")})
        return out

    def sync(self, fixtures: pd.DataFrame, out_dir: str | Path, max_articles_per_team: int = 12) -> SourceReport:
        report = SourceReport("google_news_rss")
        out_dir = Path(out_dir)
        known = self._known_players(out_dir)
        evidence: list[dict[str, Any]] = []
        availability: list[dict[str, Any]] = []
        fetched_at = utc_now_iso()
        seen_queries: set[str] = set()

        x = fixtures.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce", format="mixed", dayfirst=True)
        for _, r in x.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).iterrows():
            home, away = str(r["HomeTeam"]), str(r["AwayTeam"])
            date = pd.Timestamp(r["Date"]).isoformat()
            for team in [home, away]:
                q = f'"{team}" (injury OR injured OR suspended OR suspension OR "ruled out" OR doubtful OR lineup) football'
                if q in seen_queries:
                    continue
                seen_queries.add(q)
                try:
                    articles = self._rss(q)[:max_articles_per_team]
                except Exception as exc:
                    report.notes.append(f"News RSS failed for {team}: {exc}")
                    continue
                players = sorted(known.get(norm_team(team), set()), key=len, reverse=True)
                for a in articles:
                    # Current-news fallback is intentionally short-lived. Old
                    # articles can describe injuries that have already healed.
                    if a.get("pubDate"):
                        try:
                            published = parsedate_to_datetime(a["pubDate"])
                            if published.tzinfo is None:
                                published = published.replace(tzinfo=timezone.utc)
                            age_days = (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).total_seconds() / 86400.0
                            if age_days > 21:
                                continue
                        except Exception:
                            pass
                    text = f"{a['title']} {a['description']}"
                    status, reason, base_conf = self._classify(text)
                    matched_player = None
                    for p in players:
                        # Full-name matching avoids turning ordinary surnames/club names into players.
                        if len(p) >= 5 and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, flags=re.I):
                            matched_player = p
                            break
                    row = {
                        "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": team,
                        "Title": a["title"], "Description": a["description"], "SourceURL": a["link"],
                        "PublishedAt": a["pubDate"], "Source": "Google News RSS", "FetchedAt": fetched_at,
                        "MatchedPlayer": matched_player or "", "ClassifiedStatus": status or "", "Confidence": base_conf,
                    }
                    evidence.append(row)
                    if matched_player and status and base_conf >= self.min_promote_confidence:
                        availability.append({
                            "Date": date, "HomeTeam": home, "AwayTeam": away, "Team": team,
                            "Player": matched_player, "Status": status, "Reason": reason,
                            "ExpectedMinutes": np.nan, "PlayerXG90": np.nan, "PlayerXA90": np.nan,
                            "Source": "Google News RSS", "SourceURL": a["link"], "PublishedAt": a["pubDate"],
                            "FetchedAt": fetched_at, "SourceConfidence": base_conf,
                            "Evidence": a["title"],
                        })

        report.rows_written["news_evidence"] = _append_table(out_dir, "news_evidence", evidence)
        report.rows_written["availability"] = _append_table(out_dir, "availability", availability)
        report.matched_fixtures = len(x)
        if not known:
            report.notes.append("No known-player table was available, so news was stored as evidence but could not be promoted to player availability.")
        return report


def download_wyscout_open_archive(out_dir: str | Path, http: CachedHTTP | None = None) -> list[Path]:
    """Download the public research Wyscout dataset files without pretending they are live Wyscout API data."""
    out_dir = Path(out_dir)
    target = out_dir / "raw_wyscout_open"
    target.mkdir(parents=True, exist_ok=True)
    http = http or CachedHTTP(out_dir / ".cache")
    paths: list[Path] = []
    for name, url in WYSCOUT_OPEN_FILES.items():
        p = target / name
        if not p.exists():
            p.write_bytes(http.get_bytes(url, ttl_hours=None))
        paths.append(p)
    return paths


@dataclass
class AutoSourceManager:
    out_dir: Path
    cache_dir: Path | None = None
    sportmonks_token: str | None = None

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir) if self.cache_dir else self.out_dir / ".cache"
        self.http = CachedHTTP(self.cache_dir)

    def sync(
        self,
        matches_or_fixtures: pd.DataFrame,
        *,
        purpose: str = "prediction",
        use_statsbomb_open: bool | None = None,
        use_sportmonks: bool | None = None,
        use_news_fallback: bool | None = None,
        max_statsbomb_matches: int | None = None,
        max_sportmonks_fixtures: int | None = None,
        use_historical_weather: bool = False,
        use_historical_news: bool = False,
        max_historical_news_queries: int = 120,
        max_weather_teams: int | None = None,
    ) -> list[SourceReport]:
        purpose = purpose.lower().strip()
        if use_statsbomb_open is None:
            use_statsbomb_open = purpose == "training"
        if use_sportmonks is None:
            use_sportmonks = bool(self.sportmonks_token or os.getenv("SPORTMONKS_API_TOKEN"))
        if use_news_fallback is None:
            use_news_fallback = purpose == "prediction"

        reports: list[SourceReport] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Always materialize reproducible web-derived schedule/market context
        # from the base Football-Data rows before asking optional providers.
        reports.append(FootballDataBackfillSource().sync(matches_or_fixtures, self.out_dir))
        if use_statsbomb_open:
            try:
                reports.append(StatsBombOpenSource(self.http).sync(matches_or_fixtures, self.out_dir, max_matches=max_statsbomb_matches))
                reports.append(DerivedPlayerMovementSource().sync(self.out_dir))
            except Exception as exc:
                reports.append(SourceReport("statsbomb_open", notes=[f"Source unavailable; continuing with cached/other data: {exc}"]))
        if use_historical_weather and purpose == "training":
            try:
                reports.append(OpenMeteoHistoricalWeatherSource(self.http).sync(matches_or_fixtures, self.out_dir, max_teams=max_weather_teams))
            except Exception as exc:
                reports.append(SourceReport("open_meteo_historical_weather", notes=[f"Historical weather unavailable: {exc}"]))
        if use_historical_news and purpose == "training":
            try:
                reports.append(HistoricalNewsContextSource(self.http).sync(matches_or_fixtures, self.out_dir, max_queries=max_historical_news_queries))
            except Exception as exc:
                reports.append(SourceReport("historical_news_context", notes=[f"Historical news unavailable: {exc}"]))
        if use_sportmonks:
            try:
                reports.append(SportmonksSource(self.http, self.sportmonks_token).sync(matches_or_fixtures, self.out_dir, max_fixtures=max_sportmonks_fixtures))
            except Exception as exc:
                reports.append(SourceReport("sportmonks", notes=[f"Source unavailable; continuing with cached/other data: {exc}"]))
        if use_news_fallback:
            try:
                reports.append(GoogleNewsAvailabilitySource(self.http).sync(matches_or_fixtures, self.out_dir))
            except Exception as exc:
                reports.append(SourceReport("google_news", notes=[f"News fallback unavailable; continuing without it: {exc}"]))
        return reports


def format_source_reports(reports: Iterable[SourceReport]) -> str:
    lines: list[str] = []
    for r in reports:
        rows = ", ".join(f"{k}={v:,}" for k, v in r.rows_written.items() if v)
        lines.append(f"{r.provider}: matched={r.matched_fixtures:,}" + (f"; {rows}" if rows else ""))
        for note in r.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)
