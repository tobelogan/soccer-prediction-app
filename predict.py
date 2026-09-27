import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_predictor.config import LEAGUES
from soccer_predictor.data import download_upcoming_fixtures_multi, download_recent_completed_matches_multi, download_recent_cross_competition_matches, load_advanced_bundle, read_csv, save_csv
from soccer_predictor.bundle import LeagueModelBundle, load_model_artifact
from soccer_predictor.engine import SoccerPredictionEngine
from soccer_predictor.autosource import AutoSourceManager, format_source_reports


def main():
    p = argparse.ArgumentParser(
        description="Predict upcoming fixtures only in leagues contained in the saved model artifact."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--fixtures", default=None, help="Optional local fixture CSV. If omitted, upcoming fixtures are fetched automatically.")
    p.add_argument("--leagues", nargs="+", default=None, help="Optional subset of leagues already contained in the model bundle")
    p.add_argument("--league", default=None, help="Legacy single-league subset alias")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--next-matches", type=int, default=None)
    p.add_argument("--save-fixtures", default="auto_fixtures.csv")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--advanced-dir", default=None)
    ag = p.add_mutually_exclusive_group()
    ag.add_argument("--auto-source", dest="auto_source", action="store_true", help="Refresh current web/API data before prediction (default)")
    ag.add_argument("--no-auto-source", dest="auto_source", action="store_false")
    p.set_defaults(auto_source=True)
    p.add_argument("--auto-dir", default="advanced_data_auto")
    p.add_argument("--no-news-fallback", action="store_true")
    p.add_argument("--max-sportmonks-fixtures", type=int, default=None)
    args = p.parse_args()

    artifact = load_model_artifact(args.model)
    trained = artifact.trained_leagues
    print("Model league scope:")
    for code in trained:
        print(f"  {code}: {LEAGUES.get(code, code)}")

    requested = args.leagues or ([args.league] if args.league else None)
    if requested:
        selected = list(dict.fromkeys(str(x).upper() for x in requested))
        invalid = [x for x in selected if x not in trained]
        if invalid:
            raise ValueError(
                f"Requested league(s) {', '.join(invalid)} are not present in this model. "
                f"Allowed: {', '.join(trained)}"
            )
    else:
        selected = trained

    if args.fixtures:
        fixtures = read_csv(args.fixtures)
        fixtures = artifact.prepare_fixtures(fixtures)
        fixtures = fixtures[fixtures["League"].isin(selected)].reset_index(drop=True)
        if fixtures.empty:
            raise RuntimeError("No supplied fixtures belong to the selected model leagues.")
        print(f"Fixture source: {args.fixtures} ({len(fixtures):,} rows)")
    else:
        fixtures = download_upcoming_fixtures_multi(
            leagues=selected,
            days=args.days,
            limit=args.next_matches,
        )
        if fixtures.empty:
            raise RuntimeError(
                f"No upcoming fixtures were found for {', '.join(selected)} in the next {args.days} days. "
                "Increase --days or supply --fixtures manually."
            )
        source_counts = fixtures.groupby("FixtureSource").size().to_dict() if "FixtureSource" in fixtures.columns else {"web": len(fixtures)}
        fixtures = artifact.prepare_fixtures(fixtures)
        save_csv(fixtures, args.save_fixtures)
        source_text = ", ".join(f"{name}={count:,}" for name, count in source_counts.items())
        print(f"Fixture source(s): {source_text} ({len(fixtures):,} upcoming matches)")
        league_counts = fixtures.groupby("League").size().sort_index()
        for code, count in league_counts.items():
            print(f"  {code}: {count:,}")
        if len(selected) > 1 and len(league_counts) and int(league_counts.max()) <= 1:
            print("WARNING: Only one fixture per league was returned; all full-schedule providers likely failed and the limited fallback/cache was used.")
        print(f"Saved fetched fixtures: {args.save_fixtures}")

    if args.auto_source:
        # Update rolling team form/Elo with results played after the saved model
        # cutoff. This is separate from injury/lineup enrichment.
        try:
            if isinstance(artifact, LeagueModelBundle):
                cutoffs = artifact.state_asof_dates
            else:
                cutoffs = {artifact.trained_leagues[0]: artifact.state_asof_date}
            recent = download_recent_completed_matches_multi(selected, since_by_league=cutoffs)
            if isinstance(artifact, LeagueModelBundle):
                counts = artifact.refresh_completed_matches(recent)
            else:
                counts = {artifact.trained_leagues[0]: artifact.refresh_completed_matches(recent)}
            total = sum(counts.values())
            print(f"Recent completed-result refresh: {total} new matches " + ", ".join(f"{k}={v}" for k, v in counts.items()))
            cross = download_recent_cross_competition_matches(lookback_days=60)
            if isinstance(artifact, LeagueModelBundle):
                comp_counts = artifact.refresh_competitive_matches(cross)
            else:
                comp_counts = {artifact.trained_leagues[0]: artifact.refresh_competitive_matches(cross)}
            print("All-competition recent-form rebuild: " + ", ".join(f"{k}={v}" for k, v in comp_counts.items()))
        except Exception as exc:
            print(f"WARNING: recent completed-result refresh failed; using saved form state: {exc}")

        live_dir = Path(args.advanced_dir or args.auto_dir) / "live"
        reports = AutoSourceManager(live_dir).sync(
            fixtures,
            purpose="prediction",
            use_statsbomb_open=False,
            use_sportmonks=None,
            use_news_fallback=not args.no_news_fallback,
            max_sportmonks_fixtures=args.max_sportmonks_fixtures,
        )
        text = format_source_reports(reports)
        if text:
            print(text)
        args.advanced_dir = str(live_dir)

    rich = load_advanced_bundle(args.advanced_dir) if args.advanced_dir else None
    pred = artifact.predict(fixtures, advanced_data=rich)
    save_csv(pred, args.out)
    print(pred.to_string(index=False))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
