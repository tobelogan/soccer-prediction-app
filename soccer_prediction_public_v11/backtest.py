import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_predictor.bundle import LeagueModelBundle, load_model_artifact
from soccer_predictor.config import LEAGUES


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    p = argparse.ArgumentParser(description="Export the untouched-test betting backtest stored during training")
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="backtests")
    args = p.parse_args()

    artifact = load_model_artifact(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(artifact, LeagueModelBundle):
        engines = artifact.engines
    else:
        code = artifact.trained_leagues[0] if artifact.trained_leagues else "MODEL"
        engines = {code: artifact}

    summaries = []
    for code, engine in engines.items():
        metrics = engine.metrics
        bt = metrics.get("betting_backtest") or {}
        ledger = getattr(engine.model, "backtest_ledger_", pd.DataFrame())
        path = out_dir / f"{code}_betting_ledger.csv"
        ledger.to_csv(path, index=False)
        row = {"League": code, "LeagueName": LEAGUES.get(code, code), "EdgeThreshold": metrics.get("bet_edge_threshold")}
        row.update(bt)
        summaries.append(row)
        print(f"{code} — {LEAGUES.get(code, code)}")
        print(f"  bets: {bt.get('bets', 0)}")
        for k in ["profit_units", "roi", "hit_rate", "max_drawdown_units", "max_drawdown_pct", "mean_clv", "positive_clv_rate"]:
            print(f"  {k}: {fmt(bt.get(k))}")
        print(f"  ledger: {path}")

    summary = pd.DataFrame(summaries)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
