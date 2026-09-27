from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable
import os
import re
import unicodedata
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import requests

from .advanced import AdvancedDataBundle
from .config import BASE_URL, FIXTURES_URL


def _get_csv(url: str, timeout: int = 25) -> pd.DataFrame:
    """Download a real CSV and reject HTML/error pages masquerading as CSV.

    Football websites occasionally return a 200 HTML page at a historical CSV
    URL.  Letting pandas parse that page creates a one-column dataframe and the
    failure only appears much later as a mysterious missing-column error.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SoccerPredictor/9.1)",
        "Accept": "text/csv,text/plain,application/octet-stream,*/*",
    }
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            probe = r.content[:4096].lstrip().lower()
            ctype = str(r.headers.get("content-type", "")).lower()
            if (probe.startswith(b"<!doctype html") or probe.startswith(b"<html")
                    or b"<html" in probe[:512]):
                raise ValueError(f"Expected CSV from {url}, but the server returned HTML ({ctype or 'unknown content-type'}).")
            frame = pd.read_csv(BytesIO(r.content), on_bad_lines="skip")
            if frame.empty and len(r.content) > 0:
                raise ValueError(f"CSV source {url} returned no usable rows")
            return frame
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise last_exc


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _normalize_fixture_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common fixture-provider column names to our canonical schema."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    aliases = {
        "HomeTeam": {"hometeam", "home", "homeclub", "hometeamname", "strhometeam", "team1"},
        "AwayTeam": {"awayteam", "away", "awayclub", "awayteamname", "strawayteam", "team2"},
        "Date": {"date", "matchdate", "fixturedate", "dateevent", "eventdate"},
        "Time": {"time", "matchtime", "fixturetime", "strtime", "streventtime", "timeevent"},
        "Div": {"div", "division", "leaguecode", "competitioncode"},
        "League": {"league", "competition", "strleague", "leaguename"},
    }
    existing = {_column_key(c): c for c in out.columns}
    rename = {}
    for canonical, keys in aliases.items():
        if canonical in out.columns:
            continue
        for key in keys:
            src = existing.get(key)
            if src is not None and src not in rename:
                rename[src] = canonical
                break
    if rename:
        out = out.rename(columns=rename)
    return out





_FIXTUREDOWNLOAD_SLUGS = {
    "E0": "epl",
    "SP1": "la-liga",
    "D1": "bundesliga",
    "I1": "serie-a",
    "F1": "ligue-1",
}

def _fixturedownload_season_start(now=None) -> int:
    stamp = pd.Timestamp(now if now is not None else pd.Timestamp.now())
    # European top-flight seasons normally begin in late summer and finish the
    # following spring. July is a safe boundary for selecting the season feed.
    return int(stamp.year if stamp.month >= 7 else stamp.year - 1)

def _download_fixturedownload_fixtures(
    leagues: Iterable[str],
    now=None,
) -> pd.DataFrame:
    """Download full-season schedules from FixtureDownload and normalize them.

    This provider is intentionally used before single-event fallbacks. It
    publishes complete season CSVs for the supported European leagues without
    requiring an API key. The caller still filters the returned season to the
    requested upcoming date window.
    """
    season_start = _fixturedownload_season_start(now)
    rows = []
    failures = []
    for code in list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip())):
        slug = _FIXTUREDOWNLOAD_SLUGS.get(code)
        if not slug:
            failures.append(f"{code}: no FixtureDownload mapping")
            continue
        url = f"https://fixturedownload.com/download/{slug}-{season_start}-GMTStandardTime.csv"
        try:
            raw = _get_csv(url)
            frame = _normalize_fixture_columns(raw)
            # FixtureDownload currently names these columns with spaces.
            extra = {}
            if "HomeTeam" not in frame.columns and "Home Team" in frame.columns:
                extra["Home Team"] = "HomeTeam"
            if "AwayTeam" not in frame.columns and "Away Team" in frame.columns:
                extra["Away Team"] = "AwayTeam"
            if extra:
                frame = frame.rename(columns=extra)
            required = {"Date", "HomeTeam", "AwayTeam"}
            if not required.issubset(frame.columns):
                raise ValueError(f"unrecognized columns: {list(raw.columns)[:12]}")
            frame = frame.copy()
            parsed = _parse_match_dates(frame["Date"])
            # Preserve kickoff time when it is embedded in FixtureDownload's Date.
            frame["Time"] = parsed.dt.strftime("%H:%M")
            frame["Date"] = parsed.dt.normalize()
            frame["Div"] = code
            frame["League"] = code
            frame["FixtureSource"] = "FixtureDownload"
            rows.append(frame)
        except Exception as exc:
            failures.append(f"{code}: {exc}")
    if not rows:
        detail = " | ".join(failures) if failures else "no schedules returned"
        raise RuntimeError(f"FixtureDownload returned no usable schedules. {detail}")
    return pd.concat(rows, ignore_index=True, sort=False)


_ESPN_LEAGUE_SLUGS = {
    "E0": "eng.1",
    "SP1": "esp.1",
    "D1": "ger.1",
    "I1": "ita.1",
    "F1": "fra.1",
}


def _download_espn_fixtures(
    leagues: Iterable[str],
    date_from,
    date_to,
    timeout: int = 25,
) -> pd.DataFrame:
    """Fetch a complete upcoming date window from ESPN's soccer scoreboard.

    ESPN's site scoreboard accepts a date range and normally returns every
    event in that league/date window.  It requires no API key and is used as
    the broad free fallback when Football-Data's weekly CSV is unavailable.
    """
    start = pd.Timestamp(date_from).normalize()
    end = pd.Timestamp(date_to).normalize()
    if end < start:
        raise ValueError("date_to must not be before date_from")
    date_arg = f"{start:%Y%m%d}-{end:%Y%m%d}"
    rows = []
    failures = []
    # ESPN has been observed to reject some spoofed browser user-agents.  Use
    # requests' normal client identity and request JSON explicitly.
    headers = {"Accept": "application/json"}
    for code in list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip())):
        slug = _ESPN_LEAGUE_SLUGS.get(code)
        if not slug:
            failures.append(f"{code}: no ESPN mapping")
            continue
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
        try:
            r = requests.get(url, params={"dates": date_arg, "limit": 1000}, headers=headers, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            events = payload.get("events") or []
            for event in events:
                competitions = event.get("competitions") or []
                if not competitions:
                    continue
                comp = competitions[0]
                competitors = comp.get("competitors") or []
                home = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "home"), None)
                away = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "away"), None)
                if not home or not away:
                    continue
                home_name = ((home.get("team") or {}).get("displayName")
                             or (home.get("team") or {}).get("shortDisplayName"))
                away_name = ((away.get("team") or {}).get("displayName")
                             or (away.get("team") or {}).get("shortDisplayName"))
                stamp = event.get("date") or comp.get("date")
                if not home_name or not away_name or not stamp:
                    continue
                status_type = (event.get("status") or {}).get("type") or {}
                state = str(status_type.get("state") or "").lower()
                status_name = str(status_type.get("name") or "").upper()
                # The scoreboard date range can include games already completed
                # earlier today. Prediction is pre-match only.
                if state in {"post", "in"} or any(token in status_name for token in ("FINAL", "IN_PROGRESS", "HALFTIME")):
                    continue
                dt = pd.to_datetime(stamp, errors="coerce", utc=True)
                if pd.isna(dt):
                    continue
                rows.append({
                    "Div": code,
                    "League": code,
                    "Date": dt.tz_convert(None).normalize(),
                    "Time": dt.strftime("%H:%M"),
                    "HomeTeam": home_name,
                    "AwayTeam": away_name,
                    "FixtureID": event.get("id"),
                    "FixtureStatus": ((event.get("status") or {}).get("type") or {}).get("name"),
                    "FixtureSource": "ESPN",
                })
        except Exception as exc:
            failures.append(f"{code}: {exc}")
    if not rows:
        detail = " | ".join(failures) if failures else "no events returned"
        raise RuntimeError(f"ESPN returned no usable upcoming fixtures. {detail}")
    return pd.DataFrame(rows)

_THESPORTSDB_LEAGUE_IDS = {
    "E0": "4328",  # English Premier League
    "D1": "4331",  # German Bundesliga
    "I1": "4332",  # Italian Serie A
    "F1": "4334",  # French Ligue 1
    "SP1": "4335", # Spanish La Liga
}


def _download_thesportsdb_fixtures(leagues: Iterable[str], timeout: int = 25) -> pd.DataFrame:
    """Fetch upcoming fixtures from TheSportsDB as a no-setup fallback.

    The public documentation uses API key ``123`` for examples/free access.
    Users with their own key can set THESPORTSDB_API_KEY.
    """
    api_key = os.getenv("THESPORTSDB_API_KEY", "123").strip() or "123"
    rows = []
    failures = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SoccerPredictor/9.1)"}
    for code in list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip())):
        league_id = _THESPORTSDB_LEAGUE_IDS.get(code)
        if not league_id:
            failures.append(f"{code}: no TheSportsDB mapping")
            continue
        url = f"https://www.thesportsdb.com/api/v1/json/{api_key}/eventsnextleague.php?id={league_id}"
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            events = payload.get("events") or []
            for event in events:
                home = event.get("strHomeTeam")
                away = event.get("strAwayTeam")
                date = event.get("dateEvent") or event.get("strTimestamp")
                if not home or not away or not date:
                    continue
                rows.append({
                    "Div": code,
                    "League": code,
                    "Date": date,
                    "Time": event.get("strTime") or event.get("strEventTime"),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FixtureID": event.get("idEvent"),
                    "FixtureSource": "TheSportsDB",
                })
        except Exception as exc:
            failures.append(f"{code}: {exc}")
    if not rows:
        detail = " | ".join(failures) if failures else "no events returned"
        raise RuntimeError(f"TheSportsDB returned no usable upcoming fixtures. {detail}")
    return pd.DataFrame(rows)


def _valid_fixture_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_fixture_columns(raw)
    required = {"HomeTeam", "AwayTeam"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "Fixture source did not contain recognizable home/away-team columns. "
            f"Received columns: {list(frame.columns)[:20]}"
        )
    return clean_fixtures(frame)


def _fetch_upcoming_fixture_feed(
    leagues: Iterable[str],
    cache_path: str | Path,
    now=None,
    days: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Try broad live fixture providers in order, then valid local cache.

    Provider order:
      1. Football-Data weekly CSV (when it is a real CSV),
      2. FixtureDownload full-season schedule CSVs (no API key),
      3. ESPN date-range soccer scoreboard (no API key, full date window),
      4. TheSportsDB next-league endpoint (last-resort free fallback),
      5. last known-valid normalized cache.

    TheSportsDB free v1 endpoint is intentionally last because it currently
    returns only one upcoming league event on the free key.
    """
    cache = Path(cache_path)
    errors = []
    today = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    horizon = today + pd.Timedelta(days=max(0, int(days)))

    try:
        raw = _get_csv(FIXTURES_URL)
        fixtures = _valid_fixture_frame(raw)
        if not ({"Div", "League"} & set(fixtures.columns)):
            raise ValueError("Football-Data fixture feed has no league identifier")
        fixtures["FixtureSource"] = fixtures.get("FixtureSource", "Football-Data")
        cache.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_csv(cache, index=False)
        return fixtures, "Football-Data"
    except Exception as exc:
        errors.append(f"Football-Data: {exc}")

    try:
        fixtures = _valid_fixture_frame(_download_fixturedownload_fixtures(leagues, now=today))
        cache.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_csv(cache, index=False)
        return fixtures, "FixtureDownload"
    except Exception as exc:
        errors.append(f"FixtureDownload: {exc}")

    try:
        fixtures = _valid_fixture_frame(_download_espn_fixtures(leagues, today, horizon))
        cache.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_csv(cache, index=False)
        return fixtures, "ESPN"
    except Exception as exc:
        errors.append(f"ESPN: {exc}")

    try:
        fixtures = _valid_fixture_frame(_download_thesportsdb_fixtures(leagues))
        cache.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_csv(cache, index=False)
        return fixtures, "TheSportsDB"
    except Exception as exc:
        errors.append(f"TheSportsDB: {exc}")

    if cache.exists():
        try:
            fixtures = _valid_fixture_frame(pd.read_csv(cache, on_bad_lines="skip"))
            return fixtures, "cache"
        except Exception as exc:
            errors.append(f"cache: {exc}")

    raise RuntimeError(
        "Could not retrieve upcoming fixtures from any provider. " + " | ".join(errors)
    )


_TEAM_NAME_ALIASES = {
    # England / common provider names
    "manchesterunited": "Man United", "manchesterunitedfc": "Man United",
    "manchestercity": "Man City", "manchestercityfc": "Man City",
    "tottenhamhotspur": "Tottenham", "tottenhamhotspurfc": "Tottenham",
    "newcastleunited": "Newcastle", "newcastleunitedfc": "Newcastle",
    "brightonhovealbion": "Brighton", "brightonandhovealbion": "Brighton",
    "nottinghamforest": "Nott'm Forest", "nottinghamforestfc": "Nott'm Forest",
    "wolverhamptonwanderers": "Wolves", "wolverhamptonwanderersfc": "Wolves",
    "westhamunited": "West Ham", "westhamunitedfc": "West Ham",
    "afcbournemouth": "Bournemouth", "bournemouthafc": "Bournemouth",
    "leedsunited": "Leeds", "leedsunitedfc": "Leeds",
    "sunderlandafc": "Sunderland", "sunderlandassociationfootballclub": "Sunderland",
    "ipswichtown": "Ipswich", "ipswichtownfc": "Ipswich",
    "hullcity": "Hull", "hullcityafc": "Hull",
    "coventrycity": "Coventry", "coventrycityfc": "Coventry",
    # Spain
    "atleticomadrid": "Ath Madrid", "clubatleticodemadrid": "Ath Madrid",
    "athleticclub": "Ath Bilbao", "athleticbilbao": "Ath Bilbao",
    "realbetis": "Betis", "realbetisbalompie": "Betis",
    "realsociedad": "Sociedad", "realsociedaddefutbol": "Sociedad",
    # Germany
    "borussiadortmund": "Dortmund",
    "borussiamonchengladbach": "M'gladbach", "monchengladbach": "M'gladbach",
    "1fckoln": "FC Koln", "fckoln": "FC Koln", "cologne": "FC Koln",
    # Italy
    "intermilan": "Inter", "internazionale": "Inter", "fcinternazionale": "Inter",
    "acmilan": "Milan", "associazionecalciomilan": "Milan",
    "asroma": "Roma", "associazionesportivaroma": "Roma",
    "sscnapoli": "Napoli", "hellasverona": "Verona",
    # France
    "parissaintgermain": "Paris SG", "parissg": "Paris SG", "psg": "Paris SG",
    "olympiquedemarseille": "Marseille", "olympiquemarseille": "Marseille",
    "olympiquelyonnais": "Lyon", "asmonaco": "Monaco", "losclille": "Lille",
    "staderennais": "Rennes", "staderennaisfc": "Rennes",
}


def _team_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", text)


def resolve_team_name(value: str, candidates: Iterable[str]) -> str:
    """Map provider team labels to the exact team spelling seen in training.

    Resolution is league-scoped by the caller.  Exact/alias matches are used
    first; conservative fuzzy matching is only accepted when the best match is
    clearly better than the runner-up.
    """
    raw = str(value).strip()
    pool = [str(x) for x in candidates if str(x).strip()]
    if raw in pool or not pool:
        return raw
    keys = {_team_key(x): x for x in pool}
    key = _team_key(raw)
    if key in keys:
        return keys[key]
    alias = _TEAM_NAME_ALIASES.get(key)
    if alias in pool:
        return alias

    # Strip common club suffix/prefix words and try once more.
    stop = ("footballclub", "associationfootballclub", "soccerclub", "clubdefutbol", "fc", "afc", "cf", "sc")
    def compact(k: str) -> str:
        out = k
        for token in stop:
            if out.endswith(token):
                out = out[:-len(token)]
        return out
    ckey = compact(key)
    compact_map = {compact(k): v for k, v in keys.items()}
    if ckey in compact_map:
        return compact_map[ckey]

    scored = sorted(
        ((SequenceMatcher(None, ckey, compact(k)).ratio(), v) for k, v in keys.items()),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.86 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08):
        return scored[0][1]
    return raw


def download_season(league: str, season: str) -> pd.DataFrame:
    df = _get_csv(BASE_URL.format(season=season, league=league))
    # Football-Data CSVs can be very wide. Adding columns one-by-one with
    # frame.insert/assignment can trigger pandas fragmentation warnings.
    # Concatenating both metadata columns in one operation also returns a
    # compact frame and is faster when several seasons are combined.
    meta = pd.DataFrame({
        "League": pd.Series(league, index=df.index, dtype="string"),
        "Season": pd.Series(season, index=df.index, dtype="string"),
    })
    return pd.concat([df, meta], axis=1).copy()


def download_history(league: str, seasons: Iterable[str]) -> pd.DataFrame:
    frames = []
    errors = []
    for season in seasons:
        try:
            frames.append(download_season(league, season))
        except Exception as exc:
            errors.append(f"{season}: {exc}")
    if not frames:
        raise RuntimeError("Could not download any seasons. " + " | ".join(errors))
    out = pd.concat(frames, ignore_index=True, sort=False)
    return clean_matches(out)


def _season_start_year(season: str) -> int:
    """Convert Football-Data season codes like 2223 or 9900 to a start year."""
    text = str(season).strip()
    if len(text) != 4 or not text.isdigit():
        raise ValueError(f"Invalid season code: {season!r}; expected e.g. 2223")
    yy = int(text[:2])
    return (1900 + yy) if yy >= 90 else (2000 + yy)


def season_code_from_start_year(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def download_history_target(
    league: str,
    seasons: Iterable[str],
    target_matches: int = 5000,
    min_start_year: int = 1993,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Download at least ``target_matches`` completed matches when possible.

    Requested seasons are tried first. If they are insufficient, older
    Football-Data seasons are added automatically. Downloads run in small
    parallel batches to keep 5,000-match training practical. The returned
    dataframe is always de-duplicated and sorted chronologically.
    """
    requested = [str(s).strip() for s in seasons if str(s).strip()]
    if not requested:
        raise ValueError("At least one season must be supplied")
    target_matches = max(int(target_matches), 180)

    ordered: list[str] = []
    seen: set[str] = set()
    for season in requested:
        _season_start_year(season)
        if season not in seen:
            ordered.append(season); seen.add(season)
    oldest = min(_season_start_year(s) for s in ordered)
    for year in range(oldest - 1, int(min_start_year) - 1, -1):
        code = season_code_from_start_year(year)
        if code not in seen:
            ordered.append(code); seen.add(code)

    frames: list[pd.DataFrame] = []
    used: list[str] = []
    errors: list[str] = []
    completed = 0
    batch_size = max(1, int(max_workers))

    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset: offset + batch_size]
        results: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as pool:
            futures = {pool.submit(download_season, league, season): season for season in batch}
            for future in as_completed(futures):
                season = futures[future]
                try:
                    raw = future.result()
                    clean = clean_matches(raw)
                    if clean.empty:
                        errors.append(f"{season}: no completed matches")
                    else:
                        results[season] = clean
                except Exception as exc:
                    errors.append(f"{season}: {exc}")
        # Preserve season order regardless of network completion order.
        for season in batch:
            clean = results.get(season)
            if clean is None:
                continue
            frames.append(clean)
            used.append(season)
            completed += len(clean)
        if completed >= target_matches:
            break

    if not frames:
        raise RuntimeError("Could not download any seasons. " + " | ".join(errors))
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = clean_matches(out).drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    out = out.sort_values("Date").reset_index(drop=True)
    return out, used, errors



def download_multi_league_history_target(
    leagues: Iterable[str],
    seasons: Iterable[str],
    target_matches: int = 5000,
    min_start_year: int = 1993,
    max_workers: int = 5,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    """Download a balanced multi-league training history.

    The target is interpreted as a minimum total match count.  To prevent the
    largest/longest competition from dominating, each requested league is first
    asked for an equal share of the target.  Requested recent seasons are always
    tried first, so the returned frame can exceed ``target_matches``.
    """
    league_list = [str(x).strip().upper() for x in leagues if str(x).strip()]
    league_list = list(dict.fromkeys(league_list))
    if not league_list:
        raise ValueError("At least one league must be supplied")
    season_list = [str(x).strip() for x in seasons if str(x).strip()]
    if not season_list:
        raise ValueError("At least one season must be supplied")

    per_league_target = max(180, int((max(int(target_matches), 180) + len(league_list) - 1) // len(league_list)))
    frames: dict[str, pd.DataFrame] = {}
    used_map: dict[str, list[str]] = {}
    error_map: dict[str, list[str]] = {}

    def worker(code: str):
        return code, download_history_target(
            code, season_list, target_matches=per_league_target,
            min_start_year=min_start_year, max_workers=3,
        )

    with ThreadPoolExecutor(max_workers=min(max(1, int(max_workers)), len(league_list))) as pool:
        futures = {pool.submit(worker, code): code for code in league_list}
        for future in as_completed(futures):
            code = futures[future]
            try:
                _, (frame, used, errors) = future.result()
                frames[code] = frame
                used_map[code] = used
                error_map[code] = errors
            except Exception as exc:
                error_map[code] = [str(exc)]

    missing = [code for code in league_list if code not in frames]
    if missing:
        details = " | ".join(f"{code}: {'; '.join(error_map.get(code, []))}" for code in missing)
        raise RuntimeError(f"Could not load every requested league ({', '.join(missing)}). {details}")

    out = pd.concat([frames[code] for code in league_list], ignore_index=True, sort=False)
    out = clean_matches(out).drop_duplicates(subset=["League", "Date", "HomeTeam", "AwayTeam"], keep="last")
    out = out.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return out, used_map, error_map


def current_season_code(now=None) -> str:
    stamp = pd.Timestamp(now if now is not None else pd.Timestamp.now())
    start = int(stamp.year if stamp.month >= 7 else stamp.year - 1)
    return season_code_from_start_year(start)


def _download_espn_completed_results(
    leagues: Iterable[str],
    date_from,
    date_to,
    timeout: int = 25,
) -> pd.DataFrame:
    """Best-effort completed-result refresh for matches after model training.

    Football-Data remains preferred because it carries shots/corners/odds. ESPN
    is used only to catch very recent final scores before the historical CSV has
    updated. Those rows intentionally leave optional match statistics missing;
    FeatureBuilder substitutes neutral defaults for those secondary fields.
    """
    start = pd.Timestamp(date_from).normalize()
    end = pd.Timestamp(date_to).normalize()
    if end < start:
        return pd.DataFrame()
    date_arg = f"{start:%Y%m%d}-{end:%Y%m%d}"
    rows = []
    headers = {"Accept": "application/json"}
    for code in list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip())):
        slug = _ESPN_LEAGUE_SLUGS.get(code)
        if not slug:
            continue
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
        try:
            r = requests.get(url, params={"dates": date_arg, "limit": 1000}, headers=headers, timeout=timeout)
            r.raise_for_status()
            for event in (r.json().get("events") or []):
                comps = event.get("competitions") or []
                if not comps:
                    continue
                comp = comps[0]
                status = ((event.get("status") or {}).get("type") or {})
                state = str(status.get("state") or "").lower()
                name = str(status.get("name") or "").upper()
                if state != "post" and "FINAL" not in name:
                    continue
                competitors = comp.get("competitors") or []
                home = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "home"), None)
                away = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "away"), None)
                if not home or not away:
                    continue
                def team_name(x):
                    t = x.get("team") or {}
                    return t.get("displayName") or t.get("shortDisplayName")
                try:
                    hg, ag = float(home.get("score")), float(away.get("score"))
                except Exception:
                    continue
                stamp = event.get("date") or comp.get("date")
                dt = pd.to_datetime(stamp, errors="coerce", utc=True)
                if pd.isna(dt):
                    continue
                rows.append({
                    "League": code, "Div": code, "Date": dt.tz_convert(None).normalize(),
                    "HomeTeam": team_name(home), "AwayTeam": team_name(away),
                    "FTHG": hg, "FTAG": ag, "ResultSource": "ESPN",
                })
        except Exception:
            continue
    return pd.DataFrame(rows)



# Cross-competition feeds used only to refresh recent form. They never modify
# league Elo/priors directly. Competition weights reflect how much each result
# should contribute to short-term form, not the importance of the tournament.
_ESPN_CROSS_COMPETITIONS = {
    "UCL": ("uefa.champions", 1.00),
    "UEL": ("uefa.europa", 0.90),
    "UECL": ("uefa.europa.conf", 0.85),
    "FA Cup": ("eng.fa", 0.82),
    "EFL Cup": ("eng.league_cup", 0.75),
    "Copa del Rey": ("esp.copa_del_rey", 0.82),
    "DFB-Pokal": ("ger.dfb_pokal", 0.82),
    "Coppa Italia": ("ita.coppa_italia", 0.82),
    "Coupe de France": ("fra.coupe_de_france", 0.82),
}

def download_recent_cross_competition_matches(
    now=None,
    lookback_days: int = 45,
    timeout: int = 25,
) -> pd.DataFrame:
    """Best-effort completed continental/domestic-cup results from ESPN.

    League matches are intentionally excluded because
    ``download_recent_completed_matches_multi`` already refreshes them with
    richer Football-Data rows. This feed supplies the missing UCL/UEL/cup
    context for last-3/5/10 all-competition form.
    """
    end = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    start = end - pd.Timedelta(days=max(7, int(lookback_days)))
    date_arg = f"{start:%Y%m%d}-{end:%Y%m%d}"
    rows = []
    headers = {"Accept": "application/json"}
    for competition, (slug, weight) in _ESPN_CROSS_COMPETITIONS.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
        try:
            r = requests.get(url, params={"dates": date_arg, "limit": 1000}, headers=headers, timeout=timeout)
            r.raise_for_status()
            events = r.json().get("events") or []
        except Exception:
            continue
        for event in events:
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            status = ((event.get("status") or {}).get("type") or {})
            state = str(status.get("state") or "").lower()
            name = str(status.get("name") or "").upper()
            if state != "post" and "FINAL" not in name:
                continue
            competitors = comp.get("competitors") or []
            home = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "home"), None)
            away = next((x for x in competitors if str(x.get("homeAway", "")).lower() == "away"), None)
            if not home or not away:
                continue
            def tname(x):
                t = x.get("team") or {}
                return t.get("displayName") or t.get("shortDisplayName")
            try:
                hg, ag = float(home.get("score")), float(away.get("score"))
            except Exception:
                continue
            stamp = event.get("date") or comp.get("date")
            dt = pd.to_datetime(stamp, errors="coerce", utc=True)
            if pd.isna(dt) or not tname(home) or not tname(away):
                continue
            rows.append({
                "Date": dt.tz_convert(None),
                "HomeTeam": tname(home), "AwayTeam": tname(away),
                "FTHG": hg, "FTAG": ag,
                "Competition": competition, "CompetitionWeight": float(weight),
                "ResultSource": "ESPN-all-competitions", "EventID": event.get("id"),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return (
        out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        .sort_values(["Date", "Competition", "HomeTeam", "AwayTeam"])
        .drop_duplicates(subset=["EventID"], keep="last")
        .reset_index(drop=True)
    )


def download_recent_completed_matches_multi(
    leagues: Iterable[str],
    since_by_league: dict[str, object] | None = None,
    now=None,
    max_workers: int = 5,
) -> pd.DataFrame:
    """Refresh only completed matches newer than each saved league state.

    The current Football-Data season CSV is downloaded first for richer match
    stats, then ESPN final scores are appended as a freshness fallback.  The
    caller is responsible for reconciling provider club names to model names.
    """
    league_list = list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip()))
    if not league_list:
        return pd.DataFrame()
    today = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    season = current_season_code(today)
    since_by_league = {str(k).upper(): pd.Timestamp(v).normalize() for k, v in (since_by_league or {}).items() if v is not None and not pd.isna(v)}
    frames = []

    def worker(code):
        try:
            return code, clean_matches(download_season(code, season))
        except Exception:
            return code, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(max(1, int(max_workers)), len(league_list))) as pool:
        futures = [pool.submit(worker, code) for code in league_list]
        for fut in as_completed(futures):
            code, frame = fut.result()
            if frame.empty:
                continue
            cutoff = since_by_league.get(code)
            if cutoff is not None:
                frame = frame[frame["Date"] > cutoff]
            frame = frame[frame["Date"] <= today].copy()
            if len(frame):
                frame["ResultSource"] = "Football-Data"
                frames.append(frame)

    earliest = min(since_by_league.values()) if since_by_league else today - pd.Timedelta(days=21)
    # Give the fallback a small overlap to catch late Football-Data updates.
    espn_start = max(earliest + pd.Timedelta(days=1), today - pd.Timedelta(days=30))
    espn = _download_espn_completed_results(league_list, espn_start, today)
    if len(espn):
        keep = []
        for _, row in espn.iterrows():
            cutoff = since_by_league.get(str(row.get("League", "")).upper())
            keep.append(cutoff is None or pd.Timestamp(row["Date"]) > cutoff)
        espn = espn.loc[keep].copy()
        if len(espn):
            frames.append(espn)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["Date"] = _parse_match_dates(out["Date"])
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return out.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def download_current_fixtures() -> pd.DataFrame:
    fixtures, _ = _fetch_upcoming_fixture_feed(_THESPORTSDB_LEAGUE_IDS.keys(), ".cache/upcoming_fixtures.csv", days=10)
    return fixtures


def download_upcoming_fixtures(
    league: str = "E0",
    days: int = 10,
    limit: int | None = None,
    now=None,
    cache_path: str | Path = ".cache/upcoming_fixtures.csv",
) -> pd.DataFrame:
    return download_upcoming_fixtures_multi(
        [league], days=days, limit=limit, now=now, cache_path=cache_path
    )


def download_upcoming_fixtures_multi(
    leagues: Iterable[str],
    days: int = 10,
    limit: int | None = None,
    now=None,
    cache_path: str | Path = ".cache/upcoming_fixtures.csv",
) -> pd.DataFrame:
    """Fetch upcoming fixtures only for the requested trained leagues.

    Provider order: validated Football-Data CSV -> ESPN full date window ->
    TheSportsDB last-resort fallback -> last valid normalized local cache.  HTML pages and schema changes are rejected before
    they can overwrite the cache.
    """
    league_list = [str(x).strip().upper() for x in leagues if str(x).strip()]
    league_list = list(dict.fromkeys(league_list))
    if not league_list:
        raise ValueError("At least one league must be supplied")

    fixtures, source = _fetch_upcoming_fixture_feed(league_list, cache_path, now=now, days=days)
    scope_col = "Div" if "Div" in fixtures.columns else ("League" if "League" in fixtures.columns else None)
    if scope_col is None:
        raise ValueError("Upcoming fixture feed does not contain Div/League, so league scope cannot be verified")
    fixture_codes = fixtures[scope_col].astype(str).str.upper()
    fixtures = fixtures[fixture_codes.isin(league_list)].copy()
    fixtures["League"] = fixtures[scope_col].astype(str).str.upper()
    fixtures["FixtureSource"] = fixtures.get("FixtureSource", source)

    today = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    horizon = today + pd.Timedelta(days=max(0, int(days)))
    dates = pd.to_datetime(fixtures["Date"], errors="coerce")
    fixtures = fixtures[(dates >= today) & (dates <= horizon)].copy()
    sort_cols = [c for c in ["Date", "Time", "League", "HomeTeam", "AwayTeam"] if c in fixtures.columns]
    fixtures = fixtures.sort_values(sort_cols) if sort_cols else fixtures
    fixtures = fixtures.drop_duplicates(subset=["League", "Date", "HomeTeam", "AwayTeam"], keep="last")
    if limit is not None and int(limit) > 0:
        fixtures = fixtures.head(int(limit))
    return fixtures.reset_index(drop=True)


def read_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer, on_bad_lines="skip")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, on_bad_lines="skip")


def _read_large_events_csv(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream and aggregate huge event CSVs to team-match rows.

    This keeps millions of shot/event rows from sitting in RAM.  The resulting
    table is compatible with AdvancedFeatureStore and preserves genuine event xG
    totals while discarding event-level detail the match model does not need.
    """
    chunks = []
    for chunk in pd.read_csv(path, on_bad_lines="skip", chunksize=chunksize):
        if not {"Date", "Team"}.issubset(chunk.columns):
            raise ValueError("events.csv requires at least Date and Team columns")
        for c in ["xG", "npxG", "BigChance"]:
            if c not in chunk.columns:
                chunk[c] = 0.0
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0.0)
        keys = ["Date", "Team"]
        if "MatchID" in chunk.columns:
            keys.insert(0, "MatchID")
        aggs = {"xG": "sum", "npxG": "sum", "BigChance": "sum"}
        if "Opponent" in chunk.columns:
            aggs["Opponent"] = "first"
        chunks.append(chunk.groupby(keys, as_index=False).agg(aggs))
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True, sort=False)
    keys = ["Date", "Team"]
    if "MatchID" in out.columns:
        keys.insert(0, "MatchID")
    aggs = {"xG": "sum", "npxG": "sum", "BigChance": "sum"}
    if "Opponent" in out.columns:
        aggs["Opponent"] = "first"
    return out.groupby(keys, as_index=False).agg(aggs)


def load_advanced_bundle(directory: str | Path | None) -> AdvancedDataBundle | None:
    """Load optional rich-data tables from a directory.

    Each table may be CSV or Parquet. Expected stems are: events, player_stats,
    lineups, availability, transfers, managers, team_locations, schedule,
    weather, markets and tactics.
    """
    if not directory:
        return None
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Advanced data directory does not exist: {root}")

    def locate(stem: str) -> Path | None:
        for ext in (".parquet", ".pq", ".csv"):
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    kwargs = {}
    for name in AdvancedDataBundle.__dataclass_fields__:
        p = locate(name)
        if p is None:
            kwargs[name] = None
        elif name == "events" and p.suffix.lower() == ".csv" and p.stat().st_size > 100_000_000:
            kwargs[name] = _read_large_events_csv(p)
        else:
            kwargs[name] = read_table(p)
    return AdvancedDataBundle(**kwargs)


def advanced_bundle_status(bundle: AdvancedDataBundle | None) -> dict[str, int]:
    """Return row counts for every optional advanced feed.

    A count of 0 means that feed cannot contribute any training signal.
    This deliberately reports empty template files as zero rather than
    suggesting that advanced data is active merely because a file exists.
    """
    status: dict[str, int] = {}
    for name in AdvancedDataBundle.__dataclass_fields__:
        table = getattr(bundle, name, None) if bundle is not None else None
        status[name] = 0 if table is None else int(len(table))
    return status


def _parse_match_dates(values) -> pd.Series:
    """Parse Football-Data style dates safely while preserving ISO dates.

    Football-Data historically uses day-first dates. ``format="mixed"`` lets
    pandas also accept ISO YYYY-MM-DD from user files without forcing one
    brittle format over the whole column.
    """
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)
    except TypeError:  # compatibility with older pandas
        return pd.to_datetime(values, errors="coerce", dayfirst=True)


def advanced_bundle_diagnostics(bundle: AdvancedDataBundle | None, matches: pd.DataFrame | None = None) -> dict[str, dict]:
    """Describe optional feeds, including date range and team-name overlap.

    This is diagnostic only; it does not mutate data.  Low overlap is a common
    reason a non-empty advanced feed still produces near-zero feature coverage.
    """
    base_teams: set[str] = set()
    if matches is not None and not matches.empty:
        if "HomeTeam" in matches.columns:
            base_teams.update(matches["HomeTeam"].dropna().astype(str))
        if "AwayTeam" in matches.columns:
            base_teams.update(matches["AwayTeam"].dropna().astype(str))

    report: dict[str, dict] = {}
    for name in AdvancedDataBundle.__dataclass_fields__:
        table = getattr(bundle, name, None) if bundle is not None else None
        item = {"rows": 0, "date_min": None, "date_max": None, "team_overlap": None}
        if table is None or table.empty:
            report[name] = item
            continue
        item["rows"] = int(len(table))
        if "Date" in table.columns:
            dates = pd.to_datetime(table["Date"], errors="coerce", format="mixed", dayfirst=True)
            if dates.notna().any():
                item["date_min"] = dates.min()
                item["date_max"] = dates.max()
        if base_teams:
            feed_teams: set[str] = set()
            for c in ["Team", "HomeTeam", "AwayTeam"]:
                if c in table.columns:
                    feed_teams.update(table[c].dropna().astype(str))
            if feed_teams:
                item["team_overlap"] = len(feed_teams & base_teams) / max(1, len(feed_teams))
        report[name] = item
    return report

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical data missing required columns: {sorted(missing)}")

    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("Historical data requires a Date column.")
    df["Date"] = _parse_match_dates(df["Date"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def clean_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_fixture_columns(df)
    required = {"HomeTeam", "AwayTeam"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fixture data missing required columns: {sorted(missing)}")
    df = df.copy()
    if "Date" not in df.columns:
        df["Date"] = pd.Timestamp.today().normalize()
    else:
        df["Date"] = _parse_match_dates(df["Date"])
    if "League" not in df.columns and "Div" in df.columns:
        df["League"] = df["Div"].astype(str).str.upper()
    elif "League" in df.columns:
        df["League"] = df["League"].astype(str).str.upper()
    return df.dropna(subset=["HomeTeam", "AwayTeam"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def download_independent_league_histories(
    leagues: Iterable[str],
    seasons: Iterable[str],
    target_matches_per_league: int = 5000,
    min_start_year: int = 1993,
    league_workers: int = 3,
    season_workers: int = 4,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, list[str]]]:
    """Download an independent history for every league.

    Unlike ``download_multi_league_history_target``, the target applies to EACH
    league.  Five leagues with a 5,000 target therefore request at least 25,000
    completed matches in total.  Histories remain separate for independent
    model fitting.
    """
    league_list = list(dict.fromkeys(str(x).strip().upper() for x in leagues if str(x).strip()))
    season_list = [str(x).strip() for x in seasons if str(x).strip()]
    if not league_list:
        raise ValueError("At least one league must be supplied")
    if not season_list:
        raise ValueError("At least one season must be supplied")

    target = max(180, int(target_matches_per_league))
    frames: dict[str, pd.DataFrame] = {}
    used_map: dict[str, list[str]] = {}
    error_map: dict[str, list[str]] = {}

    def worker(code: str):
        return code, download_history_target(
            code,
            season_list,
            target_matches=target,
            min_start_year=min_start_year,
            max_workers=season_workers,
        )

    with ThreadPoolExecutor(max_workers=min(max(1, int(league_workers)), len(league_list))) as pool:
        futures = {pool.submit(worker, code): code for code in league_list}
        for future in as_completed(futures):
            code = futures[future]
            try:
                _, (frame, used, errors) = future.result()
                frames[code] = frame
                used_map[code] = used
                error_map[code] = errors
            except Exception as exc:
                error_map[code] = [str(exc)]

    missing = [code for code in league_list if code not in frames]
    if missing:
        details = " | ".join(f"{code}: {'; '.join(error_map.get(code, []))}" for code in missing)
        raise RuntimeError(f"Could not load every requested league ({', '.join(missing)}). {details}")
    return frames, used_map, error_map
