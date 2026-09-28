import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_predictor.config import DEFAULT_MODEL_LEAGUES, LEAGUES
from soccer_predictor.data import (
    download_independent_league_histories,
    load_advanced_bundle,
    advanced_bundle_status,
    advanced_bundle_diagnostics,
)
from soccer_predictor.engine import SoccerPredictionEngine
from soccer_predictor.bundle import LeagueModelBundle
from soccer_predictor.autosource import AutoSourceManager, format_source_reports
from soccer_predictor.compute import resolve_compute


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _resolve_shared_or_league_dir(base: str | None, league: str) -> str | None:
    if not base:
        return None
    root = Path(base)
    league_dir = root / league
    return str(league_dir if league_dir.exists() else root)


def main():
    p = argparse.ArgumentParser(
        description="Train independent 5,000-match models for each selected soccer league"
    )
    p.add_argument(
        "--leagues", nargs="+", default=None,
        help="League codes. Default: E0 SP1 D1 I1 F1 (EPL, La Liga, Bundesliga, Serie A, Ligue 1).",
    )
    p.add_argument("--league", default=None, help="Legacy single-league alias")
    p.add_argument("--seasons", nargs="+", required=True, help="Recent seasons to try first, e.g. 2223 2324 2425 2526 2627")
    p.add_argument("--out", default="models/top5_5000_each.joblib")
    p.add_argument(
        "--target-matches-per-league", "--target-matches", dest="target_per_league",
        type=int, default=5000,
        help="Minimum completed matches for EACH league (default: 5000).",
    )
    p.add_argument("--min-season-start", type=int, default=1993)
    p.add_argument("--download-workers", type=int, default=3, help="Number of leagues downloaded concurrently")
    p.add_argument("--season-workers", type=int, default=4, help="Concurrent season downloads inside each league")
    p.add_argument("--advanced-dir", default=None, help="Shared advanced data directory, or a root containing E0/SP1/etc subdirectories")
    ag = p.add_mutually_exclusive_group()
    ag.add_argument("--auto-source", dest="auto_source", action="store_true", help="Auto-source historical web/API enrichment (default)")
    ag.add_argument("--no-auto-source", dest="auto_source", action="store_false")
    p.set_defaults(auto_source=True)
    p.add_argument("--auto-dir", default="advanced_data_auto", help="Root directory for auto-sourced data")
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--sportmonks-history", dest="sportmonks_history", action="store_true", help="Use Sportmonks historical enrichment when a token is configured")
    sm.add_argument("--no-sportmonks-history", dest="sportmonks_history", action="store_false")
    p.set_defaults(sportmonks_history=None)
    wh = p.add_mutually_exclusive_group()
    wh.add_argument("--historical-weather", dest="historical_weather", action="store_true", help="Backfill historical match-day weather from Open-Meteo (default)")
    wh.add_argument("--no-historical-weather", dest="historical_weather", action="store_false")
    p.set_defaults(historical_weather=True)
    p.add_argument("--historical-news-backfill", action="store_true", help="Conservative timestamped Google News archive backfill for injuries/managers/transfers; slower and incomplete")
    p.add_argument("--max-historical-news-queries", type=int, default=120)
    p.add_argument("--max-weather-teams", type=int, default=None)
    p.add_argument("--max-statsbomb-matches", type=int, default=None)
    p.add_argument("--max-sportmonks-fixtures", type=int, default=None)
    p.add_argument("--backend", choices=["auto", "xgboost", "sklearn"], default="auto", help="ML backend. auto prefers XGBoost when installed.")
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto", help="Compute device. auto uses CUDA when an NVIDIA GPU is available.")
    args = p.parse_args()

    if args.leagues:
        leagues = [x.upper() for x in args.leagues]
    elif args.league:
        leagues = [args.league.upper()]
    else:
        leagues = DEFAULT_MODEL_LEAGUES.copy()
    leagues = list(dict.fromkeys(leagues))
    unknown = [x for x in leagues if x not in LEAGUES]
    if unknown:
        raise ValueError(f"Unknown league code(s): {', '.join(unknown)}. Supported: {', '.join(LEAGUES)}")

    target = max(180, int(args.target_per_league))
    expected_total = target * len(leagues)
    print("Independent league training plan:")
    for code in leagues:
        print(f"  {code}: {LEAGUES[code]} -> >= {target:,} matches")
    print(f"Minimum total history requested: {expected_total:,} matches")

    compute = resolve_compute(args.backend, args.accelerator)
    print("\nCompute configuration:")
    print(f"  backend: {compute.backend}")
    print(f"  device:  {compute.device}")
    print(f"  note:    {compute.note}")
    if compute.device == "cuda":
        print("  GPU acceleration will be used for XGBoost tree training.")
    else:
        print("  Training will use CPU. Use --accelerator gpu to require CUDA and fail fast if unavailable.")

    print("\nDownloading independent league histories...")
    histories, used_map, error_map = download_independent_league_histories(
        leagues,
        args.seasons,
        target_matches_per_league=target,
        min_start_year=args.min_season_start,
        league_workers=args.download_workers,
        season_workers=args.season_workers,
    )

    total_loaded = 0
    for code in leagues:
        data = histories[code]
        total_loaded += len(data)
        print(f"\n{code} — {LEAGUES[code]}")
        print(f"  loaded: {len(data):,} completed matches")
        print(f"  range:  {data['Date'].min().date()} .. {data['Date'].max().date()}")
        print(f"  seasons used: {', '.join(used_map.get(code, []))}")
        if len(data) < target:
            print(f"  WARNING: target {target:,} was not reached for this league")
        else:
            print(f"  target reached: {len(data):,} >= {target:,}")
        errs = error_map.get(code, [])
        if errs:
            print(f"  skipped/unavailable seasons: {len(errs)}")
    print(f"\nTotal completed matches loaded across all independent models: {total_loaded:,}")

    engines = {}
    for idx, code in enumerate(leagues, start=1):
        print("\n" + "=" * 76)
        print(f"TRAINING {idx}/{len(leagues)}: {code} — {LEAGUES[code]}")
        print("=" * 76)
        data = histories[code]
        rich = None

        if args.auto_source:
            auto_dir = Path(args.auto_dir) / "training" / code
            print(f"Auto-sourcing historical data into: {auto_dir}")
            reports = AutoSourceManager(auto_dir).sync(
                data,
                purpose="training",
                use_statsbomb_open=True,
                use_sportmonks=args.sportmonks_history,
                use_news_fallback=False,
                max_statsbomb_matches=args.max_statsbomb_matches,
                max_sportmonks_fixtures=args.max_sportmonks_fixtures,
                use_historical_weather=args.historical_weather,
                use_historical_news=args.historical_news_backfill,
                max_historical_news_queries=args.max_historical_news_queries,
                max_weather_teams=args.max_weather_teams,
            )
            text = format_source_reports(reports)
            if text:
                print(text)
            rich = load_advanced_bundle(str(auto_dir))

        explicit_dir = _resolve_shared_or_league_dir(args.advanced_dir, code)
        if explicit_dir:
            # Explicit data takes precedence when supplied.  Per-league subfolders
            # are supported to avoid accidental cross-league joins.
            rich = load_advanced_bundle(explicit_dir)

        if rich is not None:
            status = advanced_bundle_status(rich)
            print("Advanced feed rows:")
            for name, rows in status.items():
                print(f"  {name}: {rows:,}")
            if any(status.values()):
                diag = advanced_bundle_diagnostics(rich, data)
                for name, info in diag.items():
                    if not info["rows"]:
                        continue
                    overlap = info.get("team_overlap")
                    overlap_text = "n/a" if overlap is None else f"{overlap:.0%}"
                    print(f"  diagnostic {name}: team-name overlap {overlap_text}")
            else:
                print("  No advanced rows available; baseline/web-derived match features will be used.")

        engine = SoccerPredictionEngine(backend=args.backend, accelerator=args.accelerator).fit(data, advanced_data=rich)
        engines[code] = engine
        m = engine.metrics
        print("\nUntouched newest test block:")
        print(f"  train/calibration/test: {m['train_matches']:,} / {m['calibration_matches']:,} / {m['test_matches']:,}")
        for key in ["accuracy", "log_loss", "pure_model_log_loss", "market_baseline_log_loss", "market_ensemble_log_loss", "home_goal_mae", "away_goal_mae", "home_xg_std", "away_xg_std", "goal_margin_std"]:
            print(f"  {key}: {_fmt(m.get(key))}")
        print(f"  home_form_blend_weight: {_fmt(m.get('home_form_blend_weight'))}")
        print(f"  away_form_blend_weight: {_fmt(m.get('away_form_blend_weight'))}")
        print(f"  time_decay_half_life_days: {_fmt(m.get('time_decay_half_life_days'))}")
        print(f"  compute_backend: {m.get('compute_backend')}")
        print(f"  compute_device:  {m.get('compute_device')}")
        print(f"  compute_note:    {m.get('compute_note')}")
        wf = m.get("walk_forward") or {}
        print(f"  model_health_status: {wf.get('status', 'UNKNOWN')}")
        print(f"  walk_forward_recent_log_loss: {_fmt(wf.get('recent_log_loss'))}")
        print(f"  walk_forward_log_loss_drift: {_fmt(wf.get('log_loss_drift'))}")
        print(f"  retrain_recommended: {bool(wf.get('retrain_recommended', False))}")
        bt = m.get("betting_backtest") or {}
        print("  honest opening-odds betting backtest:")
        print(f"    edge_threshold: {m.get('bet_edge_threshold', 0):.3f}")
        print(f"    bets: {bt.get('bets', 0)}")
        for bk in ["profit_units", "roi", "hit_rate", "max_drawdown_units", "max_drawdown_pct", "mean_clv", "positive_clv_rate"]:
            print(f"    {bk}: {_fmt(bt.get(bk))}")

    bundle = LeagueModelBundle(engines=engines, target_matches_per_league=target)
    bundle.save(args.out)
    print("\n" + "=" * 76)
    print(f"Saved independent league model bundle: {args.out}")
    print("Contained leagues: " + ", ".join(bundle.trained_leagues))
    print(f"Each league target: {target:,} historical matches")
    print(f"Minimum requested history across bundle: {target * len(bundle.trained_leagues):,} matches")


if __name__ == "__main__":
    main()
