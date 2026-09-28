import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_predictor.autosource import AutoSourceManager, download_wyscout_open_archive, format_source_reports
from soccer_predictor.data import download_history_target, read_csv


def main():
    p = argparse.ArgumentParser(description="Automatically fill advanced_data from structured football sources and conservative web/news fallback")
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--fixtures", help="CSV containing Date,HomeTeam,AwayTeam")
    source.add_argument("--league", help="Football-Data league code, e.g. E0")
    p.add_argument("--seasons", nargs="+", help="Required with --league, e.g. 2223 2324 2425 2526 2627")
    p.add_argument("--out", default="advanced_data_auto")
    p.add_argument("--target-matches", type=int, default=5000, help="When --league is used, expand older seasons until this many completed matches are loaded")
    p.add_argument("--purpose", choices=["training", "prediction"], default=None)
    p.add_argument("--statsbomb-open", action="store_true", help="Fetch official StatsBomb Open Data where it overlaps your matches")
    p.add_argument("--sportmonks", action="store_true", help="Use SPORTMONKS_API_TOKEN for structured lineups/injuries/xG/weather/news")
    p.add_argument("--news", action="store_true", help="Use Google News RSS as a conservative current availability fallback")
    p.add_argument("--download-wyscout-open", action="store_true", help="Download the public Wyscout research archive (raw; not live commercial Wyscout)")
    p.add_argument("--max-statsbomb-matches", type=int, default=None)
    p.add_argument("--max-sportmonks-fixtures", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    manager = AutoSourceManager(out, sportmonks_token=os.getenv("SPORTMONKS_API_TOKEN"))

    if args.download_wyscout_open:
        paths = download_wyscout_open_archive(out, manager.http)
        print("Downloaded Wyscout public research files:")
        for path in paths:
            print(f"  {path}")

    if args.fixtures:
        frame = read_csv(args.fixtures)
        purpose = args.purpose or "prediction"
    elif args.league:
        if not args.seasons:
            p.error("--seasons is required with --league")
        frame, used, errs = download_history_target(args.league, args.seasons, target_matches=args.target_matches)
        print(f"Loaded {len(frame):,} matches for sync across {len(used)} seasons.")
        purpose = args.purpose or "training"
    else:
        if args.download_wyscout_open:
            return
        p.error("Provide --fixtures or --league/--seasons")

    explicit = any([args.statsbomb_open, args.sportmonks, args.news])
    reports = manager.sync(
        frame,
        purpose=purpose,
        use_statsbomb_open=args.statsbomb_open if explicit else None,
        use_sportmonks=args.sportmonks if explicit else None,
        use_news_fallback=args.news if explicit else None,
        max_statsbomb_matches=args.max_statsbomb_matches,
        max_sportmonks_fixtures=args.max_sportmonks_fixtures,
    )
    print(format_source_reports(reports) or "No sources were run.")
    print(f"Advanced data directory: {out.resolve()}")


if __name__ == "__main__":
    main()
