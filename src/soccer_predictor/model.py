from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline

from .compute import resolve_compute

from .advanced import ADVANCED_FEATURE_COLUMNS
from .config import LEAGUES, MAX_GOALS_MATRIX, RANDOM_STATE

BASE_FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_gf", "home_ga", "away_gf", "away_ga",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_ppg", "away_ppg",
    "home_gf_3", "home_gf_5", "home_gf_10",
    "home_ga_3", "home_ga_5", "home_ga_10",
    "away_gf_3", "away_gf_5", "away_gf_10",
    "away_ga_3", "away_ga_5", "away_ga_10",
    "home_ppg_5", "away_ppg_5", "home_ppg_10", "away_ppg_10",
    "home_sot_5", "away_sot_5", "home_goal_diff_5", "away_goal_diff_5",
    "home_attack_ewm", "home_defense_ewm", "away_attack_ewm", "away_defense_ewm",
    "home_allcomp_gf_5", "away_allcomp_gf_5", "home_allcomp_ga_5", "away_allcomp_ga_5",
    "home_allcomp_ppg_5", "away_allcomp_ppg_5",
    "home_allcomp_goal_diff_5", "away_allcomp_goal_diff_5",
    "home_opp_adj_gf_5", "away_opp_adj_gf_5", "home_opp_adj_ga_5", "away_opp_adj_ga_5",
    "home_allcomp_attack_ewm", "away_allcomp_attack_ewm",
    "home_allcomp_defense_ewm", "away_allcomp_defense_ewm",
    "form_home_xg", "form_away_xg",
    "home_rest_days", "away_rest_days", "home_matches_7d", "away_matches_7d",
    "home_matches_14d", "away_matches_14d",
    "market_home", "market_draw", "market_away", "month_sin", "month_cos",
    "league_home_goal_prior", "league_away_goal_prior",
    "league_total_goal_prior", "league_draw_rate_prior",
] + [f"league_is_{code}" for code in LEAGUES]
# Historical closing/sharp/exchange snapshots can arrive after the simulated
# betting entry and therefore are NOT allowed inside the fitted ML feature set.
# They remain in the feature frame for market benchmarking, final live blending
# and CLV measurement. This keeps the training/test score pre-close honest.
POST_ENTRY_MARKET_FEATURES = {
    "sharp_home", "sharp_draw", "sharp_away",
    "market_move_home", "market_move_draw", "market_move_away",
    "exchange_home", "exchange_draw", "exchange_away", "exchange_liquidity_log",
}
MODEL_ADVANCED_FEATURE_COLUMNS = [c for c in ADVANCED_FEATURE_COLUMNS if c not in POST_ENTRY_MARKET_FEATURES]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MODEL_ADVANCED_FEATURE_COLUMNS
# Expected-goal regressors should be driven by football performance/context, not
# by bookmaker probabilities. Auto-downloaded future fixtures often have no
# odds, and median-imputing market columns was compressing xG forecasts toward
# the same value across matches. The 1X2 classifier/ensemble can still use them.
GOAL_MARKET_FEATURES = {
    "market_home", "market_draw", "market_away",
    "open_home", "open_draw", "open_away",
}
GOAL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in GOAL_MARKET_FEATURES]


# The match models intentionally stay regularized.  v8 can use XGBoost on CUDA
# for faster training, while retaining sklearn as a CPU fallback.
def _regressor(backend: str = "sklearn", device: str = "cpu"):
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBRegressor(
                objective="count:poisson", eval_metric="poisson-nloglik",
                n_estimators=700, learning_rate=0.035, max_depth=6,
                min_child_weight=8, subsample=0.90, colsample_bytree=0.90,
                reg_lambda=2.0, reg_alpha=0.05, tree_method="hist",
                device=device, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.04, max_iter=420, max_leaf_nodes=28,
            min_samples_leaf=18, l2_regularization=1.4, random_state=RANDOM_STATE,
        )),
    ])


def _classifier(backend: str = "sklearn", device: str = "cpu"):
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                n_estimators=650, learning_rate=0.035, max_depth=6,
                min_child_weight=8, subsample=0.90, colsample_bytree=0.90,
                reg_lambda=2.5, reg_alpha=0.05, tree_method="hist",
                device=device, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=380, max_leaf_nodes=24,
            min_samples_leaf=20, l2_regularization=2.0, random_state=RANDOM_STATE,
        )),
    ])



def _diagnostic_regressor(backend: str = "sklearn", device: str = "cpu"):
    """Lightweight model used only for walk-forward stability diagnostics."""
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBRegressor(
                objective="count:poisson", eval_metric="poisson-nloglik",
                n_estimators=180, learning_rate=0.06, max_depth=4, min_child_weight=10,
                subsample=0.9, colsample_bytree=0.85, reg_lambda=2.5, tree_method="hist",
                device=device, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.06, max_iter=110, max_leaf_nodes=20,
            min_samples_leaf=24, l2_regularization=1.8, random_state=RANDOM_STATE,
        )),
    ])

def _diagnostic_classifier(backend: str = "sklearn", device: str = "cpu"):
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                n_estimators=180, learning_rate=0.06, max_depth=4, min_child_weight=10,
                subsample=0.9, colsample_bytree=0.85, reg_lambda=2.5, tree_method="hist",
                device=device, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.055, max_iter=100, max_leaf_nodes=18,
            min_samples_leaf=24, l2_regularization=2.2, random_state=RANDOM_STATE,
        )),
    ])

def _dc_factor(hg: int, ag: int, home_xg: float, away_xg: float, rho: float) -> float:
    """Dixon-Coles correction for the four low-scoring cells."""
    if hg == 0 and ag == 0:
        return 1.0 - home_xg * away_xg * rho
    if hg == 0 and ag == 1:
        return 1.0 + home_xg * rho
    if hg == 1 and ag == 0:
        return 1.0 + away_xg * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(home_xg: float, away_xg: float, max_goals: int = MAX_GOALS_MATRIX, rho: float = 0.0) -> np.ndarray:
    home_xg = max(float(home_xg), 0.05)
    away_xg = max(float(away_xg), 0.05)
    hp = poisson.pmf(np.arange(max_goals + 1), home_xg)
    ap = poisson.pmf(np.arange(max_goals + 1), away_xg)
    matrix = np.outer(hp, ap)
    for hg, ag in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        matrix[hg, ag] *= max(_dc_factor(hg, ag, home_xg, away_xg, rho), 0.01)
    total = matrix.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid score probability matrix")
    matrix /= total
    return matrix


def market_probs_from_row(row) -> tuple[float, float, float] | None:
    """De-vig a Football-Data style 1X2 odds row."""
    for cols in [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("MaxH", "MaxD", "MaxA")]:
        try:
            odds = np.array([float(row[c]) for c in cols])
            if np.all(np.isfinite(odds)) and np.all(odds > 1):
                p = 1 / odds
                return tuple((p / p.sum()).tolist())
        except Exception:
            continue
    return None


def _ordered_classifier_proba(model, X: pd.DataFrame) -> np.ndarray:
    raw = _pipeline_predict(model, X, proba=True)
    classes = model.named_steps["model"].classes_
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(classes):
        if int(cls) in (0, 1, 2):
            out[:, int(cls)] = raw[:, i]
    out = np.clip(out, 1e-9, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _poisson_outcome_probs(home_xg: np.ndarray, away_xg: np.ndarray, rho: float) -> np.ndarray:
    rows = []
    for h, a in zip(home_xg, away_xg):
        m = score_matrix(float(h), float(a), rho=rho)
        rows.append([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    out = np.asarray(rows, dtype=float)
    out = np.clip(out, 1e-9, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Multiclass temperature scaling operating on probabilities/logits."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    logits = np.log(p) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _market_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the best available market reference for each row.

    Priority is sharp-book -> exchange -> base de-vigged Football-Data market ->
    opening market.  The returned source labels make the blend auditable.
    """
    n = len(frame)
    market = np.full((n, 3), np.nan, dtype=float)
    sources = np.full(n, "NONE", dtype=object)
    priorities = [
        (("sharp_home", "sharp_draw", "sharp_away"), "SHARP"),
        (("exchange_home", "exchange_draw", "exchange_away"), "EXCHANGE"),
        (("market_home", "market_draw", "market_away"), "BASE"),
        (("open_home", "open_draw", "open_away"), "OPEN"),
    ]
    for cols, label in priorities:
        if not all(c in frame.columns for c in cols):
            continue
        vals = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1) & (sources == "NONE")
        if good.any():
            v = vals[good]
            v /= v.sum(axis=1, keepdims=True)
            market[good] = v
            sources[good] = label
    mask = np.isfinite(market).all(axis=1)
    return market, mask, sources


def _brier(y: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _pipeline_predict(pipe: Pipeline, X: pd.DataFrame, *, proba: bool = False) -> np.ndarray:
    """Predict without XGBoost's CPU-data/CUDA-booster mismatch warning.

    sklearn preprocessing lives on CPU.  When the fitted booster is on CUDA we
    prefer CuPy for inplace prediction if available.  If CuPy is not installed,
    prediction is performed on CPU explicitly and the booster device is restored
    afterwards.  Training remains CUDA accelerated either way.
    """
    model = pipe.named_steps["model"]
    if model.__class__.__module__.startswith("xgboost"):
        z = pipe.named_steps["imputer"].transform(X)
        booster = model.get_booster()
        device = str(model.get_params().get("device", "cpu"))
        if device.startswith("cuda"):
            try:
                import cupy as cp
                arr = cp.asarray(z)
                booster.set_param({"device": device})
                pred = booster.inplace_predict(arr)
                return cp.asnumpy(pred)
            except Exception:
                # Avoid the noisy automatic fallback warning by making the
                # device change explicit for host-memory prediction.
                booster.set_param({"device": "cpu"})
                try:
                    return np.asarray(booster.inplace_predict(np.asarray(z)))
                finally:
                    booster.set_param({"device": device})
        return np.asarray(booster.inplace_predict(np.asarray(z)))
    if proba:
        return np.asarray(pipe.predict_proba(X))
    return np.asarray(pipe.predict(X))


def _preclose_market_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Market probabilities available before the closing line.

    Closing/sharp-close/exchange snapshots are intentionally excluded here so
    the betting backtest cannot benefit from information that arrives after the
    simulated entry price.
    """
    n = len(frame)
    out = np.full((n, 3), np.nan, dtype=float)
    for cols in [("market_home", "market_draw", "market_away"), ("open_home", "open_draw", "open_away")]:
        if not all(c in frame.columns for c in cols):
            continue
        vals = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1) & ~np.isfinite(out).all(axis=1)
        if good.any():
            v = vals[good]
            v /= v.sum(axis=1, keepdims=True)
            out[good] = v
    return out, np.isfinite(out).all(axis=1)


def _odds_matrix(frame: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    cols = [f"raw_{prefix}_home_odds", f"raw_{prefix}_draw_odds", f"raw_{prefix}_away_odds"]
    if not all(c in frame.columns for c in cols):
        return np.full((len(frame), 3), np.nan), np.zeros(len(frame), dtype=bool)
    odds = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    return odds, mask


def _bet_ledger(frame: pd.DataFrame, y: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    open_odds, omask = _odds_matrix(frame, "open")
    close_odds, cmask = _odds_matrix(frame, "close")
    rows = []
    labels = np.array(["HOME", "DRAW", "AWAY"], dtype=object)
    for i in range(len(frame)):
        if not omask[i]:
            continue
        ev = probs[i] * open_odds[i] - 1.0
        pick = int(np.nanargmax(ev))
        if not np.isfinite(ev[pick]) or ev[pick] < float(threshold):
            continue
        won = int(y[i]) == pick
        profit = float(open_odds[i, pick] - 1.0) if won else -1.0
        clv = np.nan
        if cmask[i] and close_odds[i, pick] > 1.0:
            # Positive means the simulated entry captured a better decimal price
            # than the market's closing price for the same selection.
            clv = float(open_odds[i, pick] / close_odds[i, pick] - 1.0)
        row = frame.iloc[i]
        rows.append({
            "Date": row.get("Date"), "League": row.get("League"),
            "HomeTeam": row.get("HomeTeam"), "AwayTeam": row.get("AwayTeam"),
            "Bet": labels[pick], "ModelProbability": float(probs[i, pick]),
            "OpenOdds": float(open_odds[i, pick]), "ExpectedValue": float(ev[pick]),
            "Result": labels[int(y[i])], "Won": bool(won), "ProfitUnits": profit,
            "CloseOdds": float(close_odds[i, pick]) if cmask[i] else np.nan,
            "CLV": clv,
        })
    columns = [
        "Date", "League", "HomeTeam", "AwayTeam", "Bet", "ModelProbability",
        "OpenOdds", "ExpectedValue", "Result", "Won", "ProfitUnits",
        "CloseOdds", "CLV", "CumulativeProfitUnits",
    ]
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    ledger["CumulativeProfitUnits"] = ledger["ProfitUnits"].cumsum()
    return ledger.reindex(columns=columns)


def _ledger_metrics(ledger: pd.DataFrame) -> dict[str, float | int | None]:
    if ledger is None or ledger.empty:
        return {"bets": 0, "profit_units": 0.0, "roi": None, "yield": None, "hit_rate": None,
                "max_drawdown_units": None, "max_drawdown_pct": None, "mean_clv": None, "positive_clv_rate": None}
    pnl = ledger["ProfitUnits"].to_numpy(dtype=float)
    equity = 100.0 + np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[100.0, equity])[:-1]
    dd = peaks - equity
    dd_pct = np.where(peaks > 0, dd / peaks, 0.0)
    clv = pd.to_numeric(ledger["CLV"], errors="coerce")
    return {
        "bets": int(len(ledger)),
        "profit_units": float(pnl.sum()),
        "roi": float(pnl.sum() / len(ledger)),
        "yield": float(pnl.sum() / len(ledger)),
        "hit_rate": float(ledger["Won"].mean()),
        "max_drawdown_units": float(np.max(dd)) if len(dd) else 0.0,
        "max_drawdown_pct": float(np.max(dd_pct)) if len(dd_pct) else 0.0,
        "mean_clv": float(clv.mean()) if clv.notna().any() else None,
        "positive_clv_rate": float((clv.dropna() > 0).mean()) if clv.notna().any() else None,
    }


def _choose_bet_threshold(frame: pd.DataFrame, y: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    """Tune only the entry threshold on calibration data.

    A minimum bet count and drawdown penalty keep the search from selecting a
    spectacular ROI from a handful of lucky bets.
    """
    best_threshold, best_score, best_metrics = 0.08, -np.inf, {}
    min_bets = max(30, int(len(frame) * 0.04))
    for threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        ledger = _bet_ledger(frame, y, probs, threshold)
        metrics = _ledger_metrics(ledger)
        if metrics["bets"] < min_bets or metrics["roi"] is None:
            continue
        score = float(metrics["roi"]) - 0.20 * float(metrics["max_drawdown_pct"] or 0.0)
        if score > best_score:
            best_threshold, best_score, best_metrics = float(threshold), score, metrics
    return best_threshold, best_metrics


class MatchModel:
    def __init__(self, backend: str = "auto", accelerator: str = "auto"):
        compute = resolve_compute(backend=backend, accelerator=accelerator)
        self.requested_backend_ = backend
        self.requested_accelerator_ = accelerator
        self.compute_backend_ = compute.backend
        self.compute_device_ = compute.device
        self.compute_note_ = compute.note
        self.home_goal_model = _regressor(compute.backend, compute.device)
        self.away_goal_model = _regressor(compute.backend, compute.device)
        self.result_model = _classifier(compute.backend, compute.device)
        self.metrics_: dict = {}
        self.poisson_blend_weight_: float = 0.72
        self.dixon_coles_rho_: float = 0.0
        self.probability_temperature_: float = 1.0
        self.market_blend_weight_: float = 0.0
        self.backtest_market_blend_weight_: float = 0.0
        self.bet_edge_threshold_: float = 0.08
        self.home_form_blend_weight_: float = 0.0
        self.away_form_blend_weight_: float = 0.0
        self.time_decay_half_life_days_: float = 730.0
        self.backtest_ledger_: pd.DataFrame = pd.DataFrame()
        self.feature_columns_: list[str] = FEATURE_COLUMNS.copy()
        self.goal_feature_columns_: list[str] = GOAL_FEATURE_COLUMNS.copy()

    def _fit_estimators(self, X_goal: pd.DataFrame, X_result: pd.DataFrame, y_home, y_away, y_result, sample_weight=None) -> None:
        """Fit goal models without market columns and the 1X2 model with them."""
        fit_kwargs = {} if sample_weight is None else {"model__sample_weight": np.asarray(sample_weight, dtype=float)}
        try:
            self.home_goal_model.fit(X_goal, y_home, **fit_kwargs)
            self.away_goal_model.fit(X_goal, y_away, **fit_kwargs)
            self.result_model.fit(X_result, y_result, **fit_kwargs)
            return
        except Exception as exc:
            if not (self.compute_backend_ == "xgboost" and self.compute_device_ == "cuda"):
                raise
            # Driver/runtime mismatches can pass nvidia-smi detection but still fail
            # when XGBoost allocates CUDA memory.  Rebuild on CPU and continue.
            self.compute_device_ = "cpu"
            self.compute_note_ = f"CUDA runtime failed ({type(exc).__name__}); fell back to XGBoost CPU"
            self.home_goal_model = _regressor("xgboost", "cpu")
            self.away_goal_model = _regressor("xgboost", "cpu")
            self.result_model = _classifier("xgboost", "cpu")
            self.home_goal_model.fit(X_goal, y_home, **fit_kwargs)
            self.away_goal_model.fit(X_goal, y_away, **fit_kwargs)
            self.result_model.fit(X_result, y_result, **fit_kwargs)


    @staticmethod
    def _time_decay_weights(dates, half_life_days: float = 730.0) -> np.ndarray:
        d = pd.to_datetime(dates, errors="coerce")
        if len(d) == 0:
            return np.array([], dtype=float)
        latest = d.max()
        if pd.isna(latest):
            return np.ones(len(d), dtype=float)
        age = (latest - d).dt.total_seconds().fillna(0).to_numpy(dtype=float) / 86400.0
        w = np.power(0.5, age / max(float(half_life_days), 1.0))
        # Old history still contributes to robustness, but recent seasons matter
        # much more to the fitted trees.
        return np.clip(w, 0.08, 1.0)

    @staticmethod
    def _best_goal_form_blend(actual, model_xg, form_xg) -> tuple[float, float]:
        y = np.asarray(actual, dtype=float)
        m = np.asarray(model_xg, dtype=float)
        f = np.asarray(form_xg, dtype=float)
        valid = np.isfinite(y) & np.isfinite(m) & np.isfinite(f)
        if valid.sum() < 30:
            return 0.0, float("nan")
        best_w, best_loss = 0.0, float("inf")
        for w in np.arange(0.0, 0.751, 0.025):
            pred = np.clip((1.0 - w) * m[valid] + w * f[valid], 0.05, 7.5)
            loss = mean_poisson_deviance(y[valid], pred)
            if loss < best_loss:
                best_w, best_loss = float(w), float(loss)
        return best_w, best_loss

    def _blend_goal_form(self, frame: pd.DataFrame, hx, ax):
        hx = np.asarray(hx, dtype=float)
        ax = np.asarray(ax, dtype=float)
        fh = pd.to_numeric(frame.get("form_home_xg", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
        fa = pd.to_numeric(frame.get("form_away_xg", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
        wh = float(getattr(self, "home_form_blend_weight_", 0.0))
        wa = float(getattr(self, "away_form_blend_weight_", 0.0))
        good_h = np.isfinite(fh)
        good_a = np.isfinite(fa)
        out_h, out_a = hx.copy(), ax.copy()
        out_h[good_h] = (1.0 - wh) * hx[good_h] + wh * fh[good_h]
        out_a[good_a] = (1.0 - wa) * ax[good_a] + wa * fa[good_a]
        return np.clip(out_h, 0.05, 7.5), np.clip(out_a, 0.05, 7.5), fh, fa

    def _ensure_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Adding missing columns in a single concat avoids DataFrame fragmentation.
        missing = [c for c in self.feature_columns_ if c not in frame.columns]
        if not missing:
            return frame.copy()
        additions = pd.DataFrame(np.nan, index=frame.index, columns=missing)
        return pd.concat([frame.copy(), additions], axis=1)

    @staticmethod
    def _fit_rho(home_goals, away_goals, hxs, axs) -> float:
        """Penalized Dixon-Coles fit on calibration data only.

        The previous grid could land exactly on an arbitrary boundary.  A smooth
        bounded optimization with shrinkage toward zero is more stable on small
        football samples and prevents extreme corrections being selected from
        noise alone.
        """
        hg = np.asarray(home_goals, dtype=int)
        ag = np.asarray(away_goals, dtype=int)
        hx = np.asarray(hxs, dtype=float)
        ax = np.asarray(axs, dtype=float)
        valid = (hg >= 0) & (ag >= 0) & (hg <= MAX_GOALS_MATRIX) & (ag <= MAX_GOALS_MATRIX)
        hg, ag, hx, ax = hg[valid], ag[valid], hx[valid], ax[valid]
        if len(hg) < 20:
            return 0.0

        def objective(rho: float) -> float:
            nll = 0.0
            for gh, ga, xh, xa in zip(hg, ag, hx, ax):
                m = score_matrix(xh, xa, rho=float(rho))
                nll -= np.log(max(float(m[gh, ga]), 1e-12))
            # Regularize more when the calibration sample is small.
            shrink = max(0.6, 80.0 / len(hg))
            return nll / len(hg) + shrink * float(rho) ** 2

        result = minimize_scalar(objective, method="bounded", bounds=(-0.15, 0.15), options={"xatol": 1e-4})
        if not result.success or not np.isfinite(result.x):
            return 0.0
        return float(np.clip(result.x, -0.15, 0.15))

    @staticmethod
    def _best_blend(y: np.ndarray, a: np.ndarray, b: np.ndarray, step: float = 0.025) -> tuple[float, float]:
        """Return weight on `a` minimizing multiclass log loss."""
        best_w, best_loss = 0.5, float("inf")
        for w in np.arange(0.0, 1.0 + step / 2, step):
            candidate = w * a + (1.0 - w) * b
            candidate = np.clip(candidate, 1e-9, None)
            candidate /= candidate.sum(axis=1, keepdims=True)
            loss = log_loss(y, candidate, labels=[0, 1, 2])
            if loss < best_loss:
                best_w, best_loss = float(w), float(loss)
        return best_w, best_loss

    @staticmethod
    def _best_temperature(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
        best_t, best_loss = 1.0, float("inf")
        for t in np.linspace(0.70, 1.80, 45):
            p = _temperature_scale(probs, float(t))
            loss = log_loss(y, p, labels=[0, 1, 2])
            if loss < best_loss:
                best_t, best_loss = float(t), float(loss)
        return best_t, best_loss

    def _model_probs(self, frame: pd.DataFrame, *, fit_rho: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        goal_cols = getattr(self, "goal_feature_columns_", self.feature_columns_)
        X_goal = frame[goal_cols]
        X_result = frame[self.feature_columns_]
        base_hx = np.clip(_pipeline_predict(self.home_goal_model, X_goal), 0.05, 7.5)
        base_ax = np.clip(_pipeline_predict(self.away_goal_model, X_goal), 0.05, 7.5)
        hx, ax, _, _ = self._blend_goal_form(frame, base_hx, base_ax)
        if fit_rho:
            self.dixon_coles_rho_ = self._fit_rho(frame["home_goals"], frame["away_goals"], hx, ax)
        pois = _poisson_outcome_probs(hx, ax, self.dixon_coles_rho_)
        clf = _ordered_classifier_proba(self.result_model, X_result)
        raw = self.poisson_blend_weight_ * pois + (1.0 - self.poisson_blend_weight_) * clf
        raw = np.clip(raw, 1e-9, None)
        raw /= raw.sum(axis=1, keepdims=True)
        calibrated = _temperature_scale(raw, self.probability_temperature_)
        return calibrated, hx, ax

    @staticmethod
    def _split_points(n: int, calibration_fraction: float, test_fraction: float) -> tuple[int, int]:
        if calibration_fraction <= 0 or test_fraction <= 0 or calibration_fraction + test_fraction >= 0.5:
            raise ValueError("calibration_fraction and test_fraction must be positive and sum to less than 0.5")
        train_end = int(n * (1.0 - calibration_fraction - test_fraction))
        calib_end = int(n * (1.0 - test_fraction))
        # Enforce practical minimum chronological blocks while keeping at least 100 training matches.
        min_holdout = 30
        train_end = max(100, train_end)
        calib_end = max(train_end + min_holdout, calib_end)
        calib_end = min(calib_end, n - min_holdout)
        if train_end >= calib_end or calib_end >= n:
            raise ValueError("Not enough matches for train/calibration/test split")
        return train_end, calib_end

    def _walk_forward_diagnostics(self, frame: pd.DataFrame, folds: int = 3) -> dict:
        """Expanding-window pure-model diagnostics with no future calibration.

        This intentionally excludes market blending and tuned meta-parameters.
        Its purpose is stability/drift monitoring, not headline backtest ROI.
        """
        n = len(frame)
        if n < 1500 or folds < 2:
            return {"folds": [], "status": "UNKNOWN", "reason": "insufficient history"}
        folds = int(np.clip(folds, 2, 4))
        block = max(100, min(300, n // 12))
        start = n - folds * block
        if start < 600:
            start = 600
            block = max(80, (n - start) // folds)
        if block < 60:
            return {"folds": [], "status": "UNKNOWN", "reason": "insufficient walk-forward block"}
        rows = []
        goal_cols = getattr(self, "goal_feature_columns_", self.feature_columns_)
        for fold in range(folds):
            tr_end = start + fold * block
            te_end = n if fold == folds - 1 else min(n, tr_end + block)
            train = frame.iloc[:tr_end]
            test = frame.iloc[tr_end:te_end]
            if len(train) < 500 or len(test) < 50:
                continue
            hm = _diagnostic_regressor(self.compute_backend_, self.compute_device_)
            am = _diagnostic_regressor(self.compute_backend_, self.compute_device_)
            cm = _diagnostic_classifier(self.compute_backend_, self.compute_device_)
            kwargs = {"model__sample_weight": self._time_decay_weights(train["Date"], self.time_decay_half_life_days_)}
            try:
                hm.fit(train[goal_cols], train["home_goals"], **kwargs)
                am.fit(train[goal_cols], train["away_goals"], **kwargs)
                cm.fit(train[self.feature_columns_], train["result"], **kwargs)
            except Exception:
                # Monitoring must not prevent the production model from training.
                hm = _diagnostic_regressor("sklearn", "cpu"); am = _diagnostic_regressor("sklearn", "cpu"); cm = _diagnostic_classifier("sklearn", "cpu")
                hm.fit(train[goal_cols], train["home_goals"], **kwargs)
                am.fit(train[goal_cols], train["away_goals"], **kwargs)
                cm.fit(train[self.feature_columns_], train["result"], **kwargs)
            hx = np.clip(_pipeline_predict(hm, test[goal_cols]), 0.05, 7.5)
            ax = np.clip(_pipeline_predict(am, test[goal_cols]), 0.05, 7.5)
            pois = _poisson_outcome_probs(hx, ax, 0.0)
            clf = _ordered_classifier_proba(cm, test[self.feature_columns_])
            probs = 0.72 * pois + 0.28 * clf
            probs = np.clip(probs, 1e-9, None); probs /= probs.sum(axis=1, keepdims=True)
            y = test["result"].to_numpy(dtype=int)
            pred = probs.argmax(axis=1)
            rows.append({
                "fold": fold + 1, "train_matches": int(len(train)), "test_matches": int(len(test)),
                "start_date": str(pd.to_datetime(test["Date"]).min().date()),
                "end_date": str(pd.to_datetime(test["Date"]).max().date()),
                "accuracy": float(accuracy_score(y, pred)),
                "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
                "home_goal_mae": float(mean_absolute_error(test["home_goals"], hx)),
                "away_goal_mae": float(mean_absolute_error(test["away_goals"], ax)),
            })
        if len(rows) < 2:
            return {"folds": rows, "status": "UNKNOWN", "reason": "too few completed folds"}
        prior_ll = float(np.median([r["log_loss"] for r in rows[:-1]]))
        recent_ll = float(rows[-1]["log_loss"])
        delta = recent_ll - prior_ll
        prior_acc = float(np.median([r["accuracy"] for r in rows[:-1]]))
        recent_acc = float(rows[-1]["accuracy"])
        acc_delta = recent_acc - prior_acc
        if delta <= 0.03 and acc_delta >= -0.03:
            status = "GREEN"
        elif delta <= 0.08 and acc_delta >= -0.06:
            status = "AMBER"
        else:
            status = "RED"
        return {
            "folds": rows, "status": status,
            "recent_log_loss": recent_ll, "prior_log_loss": prior_ll, "log_loss_drift": float(delta),
            "recent_accuracy": recent_acc, "prior_accuracy": prior_acc, "accuracy_drift": float(acc_delta),
            "retrain_recommended": bool(status == "RED"),
        }

    def fit(self, frame: pd.DataFrame, calibration_fraction: float = 0.20, test_fraction: float = 0.20, refit_full: bool = True):
        if len(frame) < 180:
            raise ValueError("At least 180 historical matches are required; 500+ is strongly recommended.")
        frame = self._ensure_features(frame).sort_values("Date").reset_index(drop=True)
        train_end, calib_end = self._split_points(len(frame), calibration_fraction, test_fraction)
        train = frame.iloc[:train_end]
        calib = frame.iloc[train_end:calib_end]
        test = frame.iloc[calib_end:]

        goal_cols = getattr(self, "goal_feature_columns_", self.feature_columns_)
        Xtr_goal = train[goal_cols]
        Xtr_result = train[self.feature_columns_]
        train_weights = self._time_decay_weights(train["Date"], self.time_decay_half_life_days_)
        self._fit_estimators(Xtr_goal, Xtr_result, train["home_goals"], train["away_goals"], train["result"], sample_weight=train_weights)

        # ----- calibration block: choose all meta-parameters here, never on test -----
        Xc_goal = calib[goal_cols]
        Xc_result = calib[self.feature_columns_]
        base_chx = np.clip(_pipeline_predict(self.home_goal_model, Xc_goal), 0.05, 7.5)
        base_cax = np.clip(_pipeline_predict(self.away_goal_model, Xc_goal), 0.05, 7.5)
        self.home_form_blend_weight_, _ = self._best_goal_form_blend(calib["home_goals"], base_chx, calib.get("form_home_xg"))
        self.away_form_blend_weight_, _ = self._best_goal_form_blend(calib["away_goals"], base_cax, calib.get("form_away_xg"))
        chx, cax, _, _ = self._blend_goal_form(calib, base_chx, base_cax)
        self.dixon_coles_rho_ = self._fit_rho(calib["home_goals"], calib["away_goals"], chx, cax)
        cpois = _poisson_outcome_probs(chx, cax, self.dixon_coles_rho_)
        cclf = _ordered_classifier_proba(self.result_model, Xc_result)
        yc = calib["result"].to_numpy(dtype=int)
        self.poisson_blend_weight_, _ = self._best_blend(yc, cpois, cclf)
        cmodel_raw = self.poisson_blend_weight_ * cpois + (1.0 - self.poisson_blend_weight_) * cclf
        self.probability_temperature_, _ = self._best_temperature(yc, cmodel_raw)
        cmodel = _temperature_scale(cmodel_raw, self.probability_temperature_)

        cmarket, cmask, _ = _market_matrix(calib)
        self.market_blend_weight_ = 0.0
        if cmask.sum() >= 25:
            # market_blend_weight_ is explicitly the weight placed on market probabilities.
            model_w, _ = self._best_blend(yc[cmask], cmodel[cmask], cmarket[cmask])
            self.market_blend_weight_ = float(1.0 - model_w)

        # Separate pre-close blend for betting simulation. Closing/sharp-close
        # prices are excluded so the calibration cannot see the future line.
        cpre, cpre_mask = _preclose_market_matrix(calib)
        self.backtest_market_blend_weight_ = 0.0
        cbet_probs = cmodel.copy()
        if cpre_mask.sum() >= 25:
            model_w, _ = self._best_blend(yc[cpre_mask], cmodel[cpre_mask], cpre[cpre_mask])
            self.backtest_market_blend_weight_ = float(1.0 - model_w)
            w = self.backtest_market_blend_weight_
            cbet_probs[cpre_mask] = (1.0 - w) * cmodel[cpre_mask] + w * cpre[cpre_mask]
            cbet_probs[cpre_mask] /= cbet_probs[cpre_mask].sum(axis=1, keepdims=True)
        self.bet_edge_threshold_, calib_betting_metrics = _choose_bet_threshold(calib, yc, cbet_probs)

        # ----- untouched newest block: honest reported metrics -----
        model_test, thx, tax = self._model_probs(test)
        yt = test["result"].to_numpy(dtype=int)
        tmarket, tmask, tsources = _market_matrix(test)
        final_probs = model_test.copy()
        if self.market_blend_weight_ > 0 and tmask.any():
            w = self.market_blend_weight_
            final_probs[tmask] = (1.0 - w) * model_test[tmask] + w * tmarket[tmask]
            final_probs[tmask] /= final_probs[tmask].sum(axis=1, keepdims=True)
        pred = final_probs.argmax(axis=1)

        market_ll = None
        ensemble_market_subset_ll = None
        pure_model_market_subset_ll = None
        if tmask.any():
            market_ll = float(log_loss(yt[tmask], tmarket[tmask], labels=[0, 1, 2]))
            ensemble_market_subset_ll = float(log_loss(yt[tmask], final_probs[tmask], labels=[0, 1, 2]))
            pure_model_market_subset_ll = float(log_loss(yt[tmask], model_test[tmask], labels=[0, 1, 2]))

        # Honest betting backtest on the untouched newest block. Entry uses
        # only opening odds and a pre-close market blend; closing odds are used
        # only after the bet for CLV measurement.
        tpre, tpre_mask = _preclose_market_matrix(test)
        bet_probs = model_test.copy()
        if self.backtest_market_blend_weight_ > 0 and tpre_mask.any():
            w = self.backtest_market_blend_weight_
            bet_probs[tpre_mask] = (1.0 - w) * model_test[tpre_mask] + w * tpre[tpre_mask]
            bet_probs[tpre_mask] /= bet_probs[tpre_mask].sum(axis=1, keepdims=True)
        self.backtest_ledger_ = _bet_ledger(test, yt, bet_probs, self.bet_edge_threshold_)
        betting_metrics = _ledger_metrics(self.backtest_ledger_)

        coverage = {}
        for c in [x for x in self.feature_columns_ if x.endswith("_available")]:
            coverage[c] = float(pd.to_numeric(frame[c], errors="coerce").fillna(0).mean())

        source_counts = {s: int((tsources == s).sum()) for s in ["SHARP", "EXCHANGE", "BASE", "OPEN"]}
        league_metrics = {}
        if "League" in test.columns:
            league_values = test["League"].astype(str).str.upper().to_numpy()
            for league in sorted(set(league_values)):
                lm = league_values == league
                if not lm.any():
                    continue
                league_metrics[league] = {
                    "matches": int(lm.sum()),
                    "accuracy": float(accuracy_score(yt[lm], pred[lm])),
                    "log_loss": float(log_loss(yt[lm], final_probs[lm], labels=[0, 1, 2])),
                    "pure_model_log_loss": float(log_loss(yt[lm], model_test[lm], labels=[0, 1, 2])),
                }

        walk_forward = self._walk_forward_diagnostics(frame, folds=3)

        self.metrics_ = {
            "train_matches": int(len(train)),
            "calibration_matches": int(len(calib)),
            "test_matches": int(len(test)),
            # backwards-compatible alias for existing UI/code
            "validation_matches": int(len(test)),
            "accuracy": float(accuracy_score(yt, pred)),
            "log_loss": float(log_loss(yt, final_probs, labels=[0, 1, 2])),
            "pure_model_log_loss": float(log_loss(yt, model_test, labels=[0, 1, 2])),
            "brier_score": _brier(yt, final_probs),
            "pure_model_brier_score": _brier(yt, model_test),
            "poisson_blend_weight": self.poisson_blend_weight_,
            "classifier_blend_weight": 1.0 - self.poisson_blend_weight_,
            "probability_temperature": self.probability_temperature_,
            "dixon_coles_rho": self.dixon_coles_rho_,
            "home_form_blend_weight": self.home_form_blend_weight_,
            "away_form_blend_weight": self.away_form_blend_weight_,
            "time_decay_half_life_days": self.time_decay_half_life_days_,
            "market_blend_weight": self.market_blend_weight_,
            "model_blend_weight": 1.0 - self.market_blend_weight_,
            "market_baseline_log_loss": market_ll,
            "market_baseline_matches": int(tmask.sum()),
            "market_ensemble_log_loss": ensemble_market_subset_ll,
            "pure_model_market_subset_log_loss": pure_model_market_subset_ll,
            "market_source_counts": source_counts,
            "league_metrics": league_metrics,
            "backtest_market_blend_weight": self.backtest_market_blend_weight_,
            "bet_edge_threshold": self.bet_edge_threshold_,
            "calibration_betting_metrics": calib_betting_metrics,
            "betting_backtest": betting_metrics,
            "home_goal_mae": float(mean_absolute_error(test["home_goals"], thx)),
            "away_goal_mae": float(mean_absolute_error(test["away_goals"], tax)),
            "home_xg_std": float(np.std(thx)),
            "away_xg_std": float(np.std(tax)),
            "goal_margin_std": float(np.std(thx - tax)),
            "home_poisson_deviance": float(mean_poisson_deviance(test["home_goals"], thx)),
            "away_poisson_deviance": float(mean_poisson_deviance(test["away_goals"], tax)),
            "advanced_coverage": coverage,
            "walk_forward": walk_forward,
            "model_health_status": walk_forward.get("status", "UNKNOWN"),
            "retrain_recommended": bool(walk_forward.get("retrain_recommended", False)),
            "compute_backend": self.compute_backend_,
            "compute_device": self.compute_device_,
            "compute_note": self.compute_note_,
        }

        # Production refit happens only after the untouched test metrics are frozen.
        if refit_full:
            X_goal = frame[goal_cols]
            X_result = frame[self.feature_columns_]
            full_weights = self._time_decay_weights(frame["Date"], self.time_decay_half_life_days_)
            self._fit_estimators(X_goal, X_result, frame["home_goals"], frame["away_goals"], frame["result"], sample_weight=full_weights)
        return self

    def predict_frame(self, fixture_features: pd.DataFrame) -> pd.DataFrame:
        fixture_features = self._ensure_features(fixture_features)
        goal_cols = getattr(self, "goal_feature_columns_", self.feature_columns_)
        X_goal = fixture_features[goal_cols]
        X_result = fixture_features[self.feature_columns_]
        base_hxs = np.clip(_pipeline_predict(self.home_goal_model, X_goal), 0.05, 7.5)
        base_axs = np.clip(_pipeline_predict(self.away_goal_model, X_goal), 0.05, 7.5)
        hxs, axs, form_hxs, form_axs = self._blend_goal_form(fixture_features, base_hxs, base_axs)
        clf_probs = _ordered_classifier_proba(self.result_model, X_result)
        market, market_mask, market_sources = _market_matrix(fixture_features)
        out = []
        labels = ["HOME", "DRAW", "AWAY"]

        rich_cols = [c for c in self.feature_columns_ if c.endswith("_available")]
        for i, (_, row) in enumerate(fixture_features.iterrows()):
            h, a = float(hxs[i]), float(axs[i])
            matrix = score_matrix(h, a, rho=self.dixon_coles_rho_)
            pois = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
            pw = self.poisson_blend_weight_
            raw_model_probs = pw * pois + (1.0 - pw) * clf_probs[i]
            raw_model_probs /= raw_model_probs.sum()
            model_probs = _temperature_scale(raw_model_probs.reshape(1, -1), self.probability_temperature_)[0]

            probs = model_probs.copy()
            applied_market_weight = 0.0
            if market_mask[i] and self.market_blend_weight_ > 0:
                applied_market_weight = self.market_blend_weight_
                probs = (1.0 - applied_market_weight) * model_probs + applied_market_weight * market[i]
                probs /= probs.sum()

            total = h + a
            under25 = sum(
                matrix[x, y]
                for x in range(matrix.shape[0])
                for y in range(matrix.shape[1])
                if x + y <= 2
            )
            btts = 1.0 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]
            flat_idx = np.argsort(matrix.ravel())[::-1][:3]
            scorelines = []
            for idx in flat_idx:
                hg, ag = np.unravel_index(idx, matrix.shape)
                scorelines.append(f"{hg}-{ag} ({matrix[hg, ag]:.1%})")
            best = int(np.argmax(probs))
            home_margin3 = float(sum(matrix[x, y] for x in range(matrix.shape[0]) for y in range(matrix.shape[1]) if x - y >= 3))
            home_margin4 = float(sum(matrix[x, y] for x in range(matrix.shape[0]) for y in range(matrix.shape[1]) if x - y >= 4))
            away_margin3 = float(sum(matrix[x, y] for x in range(matrix.shape[0]) for y in range(matrix.shape[1]) if y - x >= 3))
            away_margin4 = float(sum(matrix[x, y] for x in range(matrix.shape[0]) for y in range(matrix.shape[1]) if y - x >= 4))

            market_vals = market[i] if market_mask[i] else [np.nan, np.nan, np.nan]
            rich_score = float(np.mean([float(row.get(c, 0.0) or 0.0) for c in rich_cols])) if rich_cols else 0.0
            out.append({
                "Date": row.get("Date"), "League": row.get("League", "UNKNOWN"), "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
                "Home_xG": round(h, 3), "Away_xG": round(a, 3), "Total_xG": round(total, 3),
                "Base_Home_xG": round(float(base_hxs[i]), 3), "Base_Away_xG": round(float(base_axs[i]), 3),
                "Form_Home_xG": round(float(form_hxs[i]), 3) if np.isfinite(form_hxs[i]) else np.nan,
                "Form_Away_xG": round(float(form_axs[i]), 3) if np.isfinite(form_axs[i]) else np.nan,
                "Expected_Goal_Margin": round(h - a, 3),
                "P_Home": float(probs[0]), "P_Draw": float(probs[1]), "P_Away": float(probs[2]),
                "Prediction": labels[best], "Confidence": float(probs[best]),
                "Model_P_Home": float(model_probs[0]), "Model_P_Draw": float(model_probs[1]), "Model_P_Away": float(model_probs[2]),
                "Market_P_Home": float(market_vals[0]), "Market_P_Draw": float(market_vals[1]), "Market_P_Away": float(market_vals[2]),
                "MarketSource": str(market_sources[i]), "MarketBlendWeightApplied": float(applied_market_weight),
                "Fair_Home": round(1 / probs[0], 3), "Fair_Draw": round(1 / probs[1], 3), "Fair_Away": round(1 / probs[2], 3),
                "P_Over2.5": float(1 - under25), "P_Under2.5": float(under25), "P_BTTS": float(btts),
                "P_HomeWin3Plus": home_margin3, "P_HomeWin4Plus": home_margin4,
                "P_AwayWin3Plus": away_margin3, "P_AwayWin4Plus": away_margin4,
                "Top_Scorelines": " | ".join(scorelines),
                "RichDataScore": rich_score,
            })
        return pd.DataFrame(out)
