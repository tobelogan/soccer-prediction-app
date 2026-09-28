from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from soccer_predictor.autosource import AutoSourceManager
from soccer_predictor.bundle import LeagueModelBundle, load_model_artifact
from soccer_predictor.compute import resolve_compute
from soccer_predictor.config import DEFAULT_MODEL_LEAGUES, LEAGUES
from soccer_predictor.deployment import ensure_remote_model, setting, truthy
from soccer_predictor.data import (
    download_independent_league_histories,
    download_recent_completed_matches_multi,
    download_recent_cross_competition_matches,
    download_upcoming_fixtures_multi,
    load_advanced_bundle,
    read_csv,
)
from soccer_predictor.engine import SoccerPredictionEngine


# -----------------------------------------------------------------------------
# Page setup + visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Soccer Prediction Lab v11",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Deployment mode
# -----------------------------------------------------------------------------
PUBLIC_APP = truthy(setting("PUBLIC_APP", st.secrets, "false"))
MODEL_URL = setting("MODEL_URL", st.secrets, "") or ""
MODEL_SHA256 = setting("MODEL_SHA256", st.secrets, "") or ""
SPORTMONKS_TOKEN = setting("SPORTMONKS_API_TOKEN", st.secrets, "") or ""
if SPORTMONKS_TOKEN:
    # Existing provider adapters read this token from the environment.  Streamlit
    # secrets stay server-side; this only exposes it to this Python process.
    os.environ["SPORTMONKS_API_TOKEN"] = SPORTMONKS_TOKEN


st.markdown(
    """
    <style>
      .block-container {max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .app-title {font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .1rem;}
      .app-subtitle {opacity: .72; margin-bottom: 1.25rem;}
      .hero {
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(34,197,94,.08), rgba(59,130,246,.06));
        margin-bottom: 1rem;
      }
      .section-label {font-weight: 750; font-size: 1.05rem; margin: .25rem 0 .7rem 0;}
      .muted {opacity: .68;}
      .match-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 14px;
        padding: .95rem 1rem;
        margin-bottom: .55rem;
      }
      .pill {display:inline-block; padding:.22rem .55rem; border-radius:999px; font-size:.78rem; font-weight:700; background:rgba(128,128,128,.12);}
      .good {background:rgba(34,197,94,.13);}
      .warn {background:rgba(245,158,11,.15);}
      .soft {background:rgba(59,130,246,.12);}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .7rem .85rem;
        border-radius: 12px;
      }
      div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
DEFAULT_MODEL_CANDIDATES = [
    "models/top5_5000_each_v11.joblib",
    "models/top5_5000_each_v10.joblib",
    "models/top_leagues_5000_each.joblib",
    "models/top5_5000_each_v9.joblib",
]

for key, value in {
    "bundle": None,
    "model_path": None,
    "fixtures": None,
    "pred": None,
    "pred_audit": None,
    "fixture_message": None,
    "refresh_message": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_pct(v: Any, digits: int = 1) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v) * 100:.{digits}f}%"
    except Exception:
        return "—"


def _fmt_num(v: Any, digits: int = 3) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "—"


def _artifact_engines(artifact):
    if isinstance(artifact, LeagueModelBundle):
        return artifact.engines
    code = artifact.trained_leagues[0] if artifact.trained_leagues else "MODEL"
    return {code: artifact}


def _artifact_metrics(artifact) -> dict[str, dict]:
    if isinstance(artifact, LeagueModelBundle):
        return artifact.metrics
    code = artifact.trained_leagues[0] if artifact.trained_leagues else "MODEL"
    return {code: artifact.metrics}


def _artifact_cutoffs(artifact) -> dict[str, Any]:
    if isinstance(artifact, LeagueModelBundle):
        return artifact.state_asof_dates
    code = artifact.trained_leagues[0]
    return {code: artifact.state_asof_date}


def _default_model_path() -> str:
    for p in DEFAULT_MODEL_CANDIDATES:
        if Path(p).exists():
            return p
    return DEFAULT_MODEL_CANDIDATES[0]


def _existing_local_model() -> str | None:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def _cached_load_model(path_text: str, mtime_ns: int):
    # mtime_ns is part of the cache key so a deliberately replaced model is
    # reloaded, while normal Streamlit reruns do not deserialize it repeatedly.
    return load_model_artifact(Path(path_text))


def _load_model_path(path_text: str) -> None:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    artifact = _cached_load_model(str(path), int(path.stat().st_mtime_ns))
    st.session_state.bundle = artifact
    st.session_state.model_path = str(path)
    st.session_state.pred = None
    st.session_state.pred_audit = None


def _load_deployment_model() -> None:
    local = _existing_local_model()
    if local:
        _load_model_path(local)
        return
    if MODEL_URL:
        path = ensure_remote_model(
            MODEL_URL,
            destination=Path(".cache") / "top5_5000_each_v11.joblib",
            expected_sha256=MODEL_SHA256 or None,
        )
        _load_model_path(str(path))
        return
    raise FileNotFoundError(
        "No trusted model is available. Add models/top5_5000_each_v11.joblib "
        "to the deployment or configure MODEL_URL in Streamlit Secrets."
    )


def _performance_frame(artifact) -> pd.DataFrame:
    rows = []
    for code, m in _artifact_metrics(artifact).items():
        rows.append(
            {
                "LeagueCode": code,
                "League": LEAGUES.get(code, code),
                "Train": m.get("train_matches"),
                "Calibration": m.get("calibration_matches"),
                "Test": m.get("test_matches"),
                "Accuracy": m.get("accuracy"),
                "Final Log Loss": m.get("log_loss"),
                "Pure Model Log Loss": m.get("pure_model_log_loss"),
                "Market Log Loss": m.get("market_baseline_log_loss"),
                "Home Goal MAE": m.get("home_goal_mae"),
                "Away Goal MAE": m.get("away_goal_mae"),
                "Home xG Std": m.get("home_xg_std"),
                "Away xG Std": m.get("away_xg_std"),
                "Goal Margin Std": m.get("goal_margin_std"),
                "Backend": m.get("compute_backend"),
                "Device": m.get("compute_device"),
            }
        )
    return pd.DataFrame(rows)


def _betting_frame(artifact) -> pd.DataFrame:
    rows = []
    for code, m in _artifact_metrics(artifact).items():
        bt = m.get("betting_backtest") or {}
        rows.append(
            {
                "LeagueCode": code,
                "League": LEAGUES.get(code, code),
                "Edge Threshold": m.get("bet_edge_threshold"),
                "Bets": bt.get("bets", 0),
                "Profit Units": bt.get("profit_units"),
                "ROI": bt.get("roi"),
                "Hit Rate": bt.get("hit_rate"),
                "Max Drawdown": bt.get("max_drawdown_units"),
                "Max Drawdown %": bt.get("max_drawdown_pct"),
                "Mean CLV": bt.get("mean_clv"),
                "Positive CLV": bt.get("positive_clv_rate"),
            }
        )
    return pd.DataFrame(rows)


def _coverage_frame(artifact) -> pd.DataFrame:
    rows = []
    for code, m in _artifact_metrics(artifact).items():
        cov = m.get("advanced_coverage") or {}
        for feature, value in cov.items():
            rows.append(
                {
                    "LeagueCode": code,
                    "League": LEAGUES.get(code, code),
                    "Feature": feature.replace("_data_available", "").replace("_", " ").title(),
                    "Coverage": value,
                }
            )
    return pd.DataFrame(rows)


def _confidence_label(row: pd.Series) -> tuple[str, str]:
    probs = sorted([float(row.get("P_Home", 0)), float(row.get("P_Draw", 0)), float(row.get("P_Away", 0))], reverse=True)
    top = probs[0]
    gap = top - probs[1]
    if top >= 0.60 and gap >= 0.12:
        return "HIGH", "good"
    if top >= 0.50 and gap >= 0.08:
        return "MEDIUM", "soft"
    return "LOW / NO STRONG EDGE", "warn"


def _display_prediction_detail(row: pd.Series, audit_row: pd.Series | None) -> None:
    home = row.get("HomeTeam", "Home")
    away = row.get("AwayTeam", "Away")
    league = row.get("League", "")
    confidence_text, confidence_css = _confidence_label(row)

    st.markdown(
        f"""
        <div class="hero">
          <div class="muted">{LEAGUES.get(str(league), str(league))} · {row.get('Date', '')}</div>
          <div style="font-size:1.55rem;font-weight:800;margin:.15rem 0 .4rem 0;">{home} vs {away}</div>
          <span class="pill {confidence_css}">{confidence_text}</span>
          <span class="pill">Model lean: {row.get('Prediction', '—')}</span>
          <span class="pill">Market source: {row.get('MarketSource', 'none')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{home} xG", _fmt_num(row.get("Home_xG"), 2), delta=f"Base {_fmt_num(row.get('Base_Home_xG'), 2)}")
    c2.metric(f"{away} xG", _fmt_num(row.get("Away_xG"), 2), delta=f"Base {_fmt_num(row.get('Base_Away_xG'), 2)}")
    c3.metric("Expected margin", _fmt_num(row.get("Expected_Goal_Margin"), 2))
    c4.metric("Total xG", _fmt_num(row.get("Total_xG"), 2))

    probs = pd.DataFrame(
        {
            "Outcome": [home, "Draw", away],
            "Probability": [row.get("P_Home"), row.get("P_Draw"), row.get("P_Away")],
        }
    )
    fig = px.bar(probs, x="Outcome", y="Probability", text_auto=".1%", range_y=[0, 1])
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%", showlegend=False)

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-label">Goal markets</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Over 2.5", _fmt_pct(row.get("P_Over2.5")))
        b.metric("BTTS", _fmt_pct(row.get("P_BTTS")))
        a.metric(f"{home} win 3+", _fmt_pct(row.get("P_HomeWin3Plus")))
        b.metric(f"{away} win 3+", _fmt_pct(row.get("P_AwayWin3Plus")))
        a.metric(f"{home} win 4+", _fmt_pct(row.get("P_HomeWin4Plus")))
        b.metric(f"{away} win 4+", _fmt_pct(row.get("P_AwayWin4Plus")))
        st.caption(f"Most likely scorelines: {row.get('Top_Scorelines', '—')}")

    st.markdown('<div class="section-label">xG decomposition</div>', unsafe_allow_html=True)
    xg_table = pd.DataFrame(
        {
            "Team": [home, away],
            "Base xG": [row.get("Base_Home_xG"), row.get("Base_Away_xG")],
            "Recent-form xG": [row.get("Form_Home_xG"), row.get("Form_Away_xG")],
            "Final xG": [row.get("Home_xG"), row.get("Away_xG")],
        }
    )
    st.dataframe(xg_table, use_container_width=True, hide_index=True)

    model_market = pd.DataFrame(
        {
            "Outcome": ["Home", "Draw", "Away"],
            "Pure model": [row.get("Model_P_Home"), row.get("Model_P_Draw"), row.get("Model_P_Away")],
            "Market": [row.get("Market_P_Home"), row.get("Market_P_Draw"), row.get("Market_P_Away")],
            "Final": [row.get("P_Home"), row.get("P_Draw"), row.get("P_Away")],
        }
    )
    with st.expander("Model vs market blend"):
        st.dataframe(model_market, use_container_width=True, hide_index=True)
        st.caption(f"Market blend weight actually applied: {_fmt_pct(row.get('MarketBlendWeightApplied'))}")

    if audit_row is not None:
        st.markdown('<div class="section-label">Recent-form audit</div>', unsafe_allow_html=True)
        form_rows = [
            ("Elo", "home_elo", "away_elo"),
            ("League goals / match — last 3", "home_gf_3", "away_gf_3"),
            ("League goals / match — last 5", "home_gf_5", "away_gf_5"),
            ("League conceded / match — last 5", "home_ga_5", "away_ga_5"),
            ("League points / game — last 5", "home_ppg_5", "away_ppg_5"),
            ("All-competition goals / match — last 5", "home_allcomp_gf_5", "away_allcomp_gf_5"),
            ("All-competition conceded / match — last 5", "home_allcomp_ga_5", "away_allcomp_ga_5"),
            ("All-competition points / game — last 5", "home_allcomp_ppg_5", "away_allcomp_ppg_5"),
            ("Opponent-adjusted goals — last 5", "home_opp_adj_gf_5", "away_opp_adj_gf_5"),
            ("Opponent-adjusted conceded — last 5", "home_opp_adj_ga_5", "away_opp_adj_ga_5"),
            ("All-competition attack EWMA", "home_allcomp_attack_ewm", "away_allcomp_attack_ewm"),
            ("All-competition defence EWMA", "home_allcomp_defense_ewm", "away_allcomp_defense_ewm"),
            ("Confirmed XI completeness", "home_xi_completeness", "away_xi_completeness"),
            ("XI attack vs expected XI", "home_xi_attack_delta_pct", "away_xi_attack_delta_pct"),
            ("Unavailable attack share", "home_unavailable_attack_share", "away_unavailable_attack_share"),
            ("Goalkeeper delta", "home_gk_delta", "away_gk_delta"),
            ("Form xG input", "form_home_xg", "form_away_xg"),
        ]
        recent = []
        for label, hc, ac in form_rows:
            if hc in audit_row.index or ac in audit_row.index:
                recent.append({"Metric": label, home: audit_row.get(hc), away: audit_row.get(ac)})
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)


def _refresh_live_state(artifact, selected: list[str]) -> None:
    cutoffs = _artifact_cutoffs(artifact)
    recent = download_recent_completed_matches_multi(selected, since_by_league=cutoffs)
    if isinstance(artifact, LeagueModelBundle):
        refresh_counts = artifact.refresh_completed_matches(recent)
    else:
        refresh_counts = {artifact.trained_leagues[0]: artifact.refresh_completed_matches(recent)}
    refreshed = int(sum(refresh_counts.values()))

    # Rebuild last-3/5/10 all-competition state with continental/domestic cups.
    cross = download_recent_cross_competition_matches(lookback_days=60)
    cross_count = 0
    if isinstance(artifact, LeagueModelBundle):
        comp_counts = artifact.refresh_competitive_matches(cross)
        cross_count = int(sum(comp_counts.values()))
    else:
        comp_counts = {artifact.trained_leagues[0]: artifact.refresh_competitive_matches(cross)}
        cross_count = int(sum(comp_counts.values()))

    if refreshed or cross_count:
        st.session_state.refresh_message = (
            f"Live form refreshed: {refreshed} new domestic-league results; "
            f"all-competition state rebuilt with {cross_count} team-event updates. "
            + ", ".join(f"{k}={v}" for k, v in refresh_counts.items())
        )
    else:
        st.session_state.refresh_message = "Recent-form state is current with the available league and cup/continental feeds."


# -----------------------------------------------------------------------------
# Sidebar — model + navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚽ Prediction Lab")
    st.caption("v11 · all-competition form + player strength + walk-forward health")

    pages = ["Dashboard", "Predictions", "Performance", "Backtest", "Data Health"]
    if not PUBLIC_APP:
        pages.insert(-1, "Training")
    page = st.radio("Navigate", pages, label_visibility="collapsed")

    st.divider()
    st.markdown("### Model")
    if PUBLIC_APP:
        st.caption("Public prediction mode · trusted model only")
        if st.session_state.bundle is None:
            st.info("The deployment model will load automatically.")
    else:
        default_path = st.session_state.model_path or _default_model_path()
        model_path_input = st.text_input("Local model path", value=default_path)
        if st.button("Load local model", use_container_width=True):
            try:
                _load_model_path(model_path_input)
                st.success("Model loaded")
            except Exception as exc:
                st.error(str(exc))

        st.caption("Only load model files you created or trust. joblib/pickle is executable data.")
        uploaded_model = st.file_uploader("Or upload trusted .joblib", type=["joblib"])
        if uploaded_model is not None and st.button("Load uploaded model", use_container_width=True):
            try:
                tmp = Path(".cache/uploaded_model.joblib")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(uploaded_model.read())
                _load_model_path(str(tmp))
                st.success("Uploaded trusted model loaded")
            except Exception as exc:
                st.error(str(exc))

    artifact = st.session_state.bundle
    if artifact is not None:
        st.divider()
        st.markdown("### Active model")
        st.write(f"**Leagues:** {', '.join(artifact.trained_leagues)}")
        if st.session_state.model_path and not PUBLIC_APP:
            st.caption(st.session_state.model_path)


# Best-effort auto-load the trusted deployment/local model on first run.
if st.session_state.bundle is None and not st.session_state.get("autoload_attempted", False):
    st.session_state.autoload_attempted = True
    try:
        with st.spinner("Loading prediction model..."):
            _load_deployment_model()
    except Exception as exc:
        st.session_state.model_load_error = str(exc)

artifact = st.session_state.bundle

st.markdown('<div class="app-title">Soccer Prediction Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">All-competition opponent-adjusted form, player availability, league-specific xG, walk-forward monitoring and honest backtesting.</div>',
    unsafe_allow_html=True,
)
if PUBLIC_APP:
    st.caption("Public demo · prediction and diagnostics only. Model training remains on the owner's GPU machine.")

if artifact is None and page != "Training":
    err = st.session_state.get("model_load_error")
    if PUBLIC_APP:
        st.error("The public prediction model could not be loaded.")
        if err:
            st.code(err)
        st.caption("Deployment owner: configure MODEL_URL (and optionally MODEL_SHA256) in Streamlit Secrets, or include the trusted model in models/.")
    else:
        st.info("Load a saved model from the sidebar, or open **Training** to create a new model bundle.")
        if err:
            st.caption(err)
    st.stop()


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
if page == "Dashboard":
    perf = _performance_frame(artifact)
    metrics = _artifact_metrics(artifact)

    mean_acc = pd.to_numeric(perf["Accuracy"], errors="coerce").mean()
    mean_ll = pd.to_numeric(perf["Final Log Loss"], errors="coerce").mean()
    total_test = pd.to_numeric(perf["Test"], errors="coerce").sum()

    st.markdown(
        """
        <div class="hero">
          <div style="font-weight:800;font-size:1.2rem;">Model control centre</div>
          <div class="muted">Use Predictions for upcoming matches, Performance to inspect accuracy/calibration, Backtest for betting diagnostics, and Data Health to verify the feeds the model actually saw.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Independent leagues", len(artifact.trained_leagues))
    c2.metric("Untouched test matches", f"{int(total_test):,}" if np.isfinite(total_test) else "—")
    c3.metric("Mean 1X2 accuracy", _fmt_pct(mean_acc))
    c4.metric("Mean final log loss", _fmt_num(mean_ll, 3))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-label">League performance</div>', unsafe_allow_html=True)
        show = perf[["LeagueCode", "League", "Accuracy", "Final Log Loss", "Market Log Loss", "Home Goal MAE", "Away Goal MAE"]].copy()
        show["Accuracy %"] = pd.to_numeric(show.pop("Accuracy"), errors="coerce") * 100.0
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accuracy %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            },
        )
    with right:
        st.markdown('<div class="section-label">Model status</div>', unsafe_allow_html=True)
        for code in artifact.trained_leagues:
            engine = _artifact_engines(artifact)[code]
            cutoff = getattr(engine, "state_asof_date", None)
            m = metrics.get(code, {})
            device = m.get("compute_device", "—")
            health = m.get("model_health_status", "UNKNOWN")
            st.markdown(
                f"**{LEAGUES.get(code, code)}**  \n"
                f"State as of: `{cutoff}` · Compute: `{device}` · Health: `{health}` · xG margin std: `{_fmt_num(m.get('goal_margin_std'), 2)}`"
            )

    bt = _betting_frame(artifact)
    if len(bt):
        st.markdown('<div class="section-label">Backtest snapshot</div>', unsafe_allow_html=True)
        st.dataframe(bt[["League", "Bets", "ROI", "Mean CLV", "Max Drawdown %"]], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------
elif page == "Predictions":
    trained = artifact.trained_leagues

    st.markdown('<div class="section-label">Upcoming fixtures</div>', unsafe_allow_html=True)
    with st.container(border=True):
        r1, r2, r3 = st.columns([1.5, 1, 1])
        selected = r1.multiselect(
            "Leagues",
            trained,
            default=trained,
            format_func=lambda x: f"{x} — {LEAGUES.get(x, x)}",
        )
        days = r2.number_input("Look ahead", min_value=1, max_value=30, value=10, step=1, help="Days into the future")
        fixture_mode = r3.selectbox("Fixture source", ["Auto-download", "Upload CSV"])
        auto_live = st.checkbox(
            "Refresh completed results + current injury/lineup/news data immediately before prediction",
            value=True,
        )

        if fixture_mode == "Auto-download":
            if st.button("Fetch upcoming fixtures", type="primary"):
                if not selected:
                    st.warning("Select at least one trained league.")
                else:
                    try:
                        with st.spinner("Fetching complete upcoming fixture window..."):
                            fetched = download_upcoming_fixtures_multi(selected, days=int(days))
                        st.session_state.fixtures = fetched
                        if fetched.empty:
                            st.session_state.fixture_message = f"No fixtures found in the next {int(days)} days."
                        else:
                            sources = (
                                fetched.groupby("FixtureSource").size().to_dict()
                                if "FixtureSource" in fetched.columns
                                else {"web": len(fetched)}
                            )
                            st.session_state.fixture_message = (
                                f"Loaded {len(fetched)} fixtures · " + ", ".join(f"{k}: {v}" for k, v in sources.items())
                            )
                    except Exception as exc:
                        st.session_state.fixtures = None
                        st.error(f"Fixture download failed: {exc}")
        else:
            up = st.file_uploader("Fixture CSV", type="csv", key="fixture_upload")
            if up is not None:
                st.session_state.fixtures = read_csv(up)
                st.session_state.fixture_message = f"Loaded {len(st.session_state.fixtures)} uploaded fixtures."

    if st.session_state.fixture_message:
        st.info(st.session_state.fixture_message)

    fixtures = st.session_state.fixtures
    if fixtures is not None and len(fixtures):
        try:
            fixtures = artifact.prepare_fixtures(fixtures)
            if selected:
                fixtures = fixtures[fixtures["League"].isin(selected)].reset_index(drop=True)
        except Exception as exc:
            st.error(f"Could not prepare fixtures: {exc}")
            fixtures = None

    if fixtures is not None and len(fixtures):
        counts = fixtures.groupby("League").size().rename("Fixtures").reset_index()
        counts["Competition"] = counts["League"].map(lambda x: LEAGUES.get(x, x))
        st.dataframe(counts[["League", "Competition", "Fixtures"]], use_container_width=True, hide_index=True)

        with st.expander("View fetched fixtures", expanded=False):
            st.dataframe(fixtures, use_container_width=True, hide_index=True)

        if st.button("Run predictions", type="primary", use_container_width=True):
            rich = None
            if auto_live:
                try:
                    with st.spinner("Refreshing completed results since the model cutoff..."):
                        _refresh_live_state(artifact, selected)
                    st.success(st.session_state.refresh_message)
                except Exception as exc:
                    st.warning(f"Recent-result refresh failed; using saved state. {exc}")

                try:
                    live_dir = Path("advanced_data_auto") / "live"
                    with st.spinner("Refreshing current structured/news data..."):
                        AutoSourceManager(live_dir).sync(
                            fixtures,
                            purpose="prediction",
                            use_statsbomb_open=False,
                            use_sportmonks=None,
                            use_news_fallback=True,
                        )
                    rich = load_advanced_bundle(str(live_dir))
                except Exception as exc:
                    st.warning(f"Advanced live-data refresh failed; continuing without it. {exc}")

            with st.spinner("Running league-specific prediction engines..."):
                pred = artifact.predict(fixtures, advanced_data=rich)
                try:
                    audit = artifact.fixture_feature_audit(fixtures)
                except Exception:
                    audit = pd.DataFrame()
            st.session_state.pred = pred
            st.session_state.pred_audit = audit

    pred = st.session_state.pred
    if pred is not None and len(pred):
        st.divider()
        st.markdown('<div class="section-label">Prediction board</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns([1.3, 1, 1])
        league_filter = p1.multiselect(
            "Show leagues",
            sorted(pred["League"].dropna().unique().tolist()),
            default=sorted(pred["League"].dropna().unique().tolist()),
            format_func=lambda x: LEAGUES.get(x, x),
            key="prediction_league_filter",
        )
        sort_by = p2.selectbox("Sort", ["Date", "Confidence", "Total xG", "Expected goal margin"])
        only_stronger = p3.checkbox("Hide low-confidence leans", value=False)

        board = pred[pred["League"].isin(league_filter)].copy()
        if only_stronger:
            keep = []
            for _, r in board.iterrows():
                label, _ = _confidence_label(r)
                keep.append(label != "LOW / NO STRONG EDGE")
            board = board[pd.Series(keep, index=board.index)]

        sort_map = {
            "Date": ("Date", True),
            "Confidence": ("Confidence", False),
            "Total xG": ("Total_xG", False),
            "Expected goal margin": ("Expected_Goal_Margin", False),
        }
        col, asc = sort_map[sort_by]
        if col in board.columns:
            board = board.sort_values(col, ascending=asc)

        summary_cols = [
            c
            for c in [
                "Date",
                "League",
                "HomeTeam",
                "AwayTeam",
                "Home_xG",
                "Away_xG",
                "P_Home",
                "P_Draw",
                "P_Away",
                "Prediction",
                "Confidence",
                "P_Over2.5",
                "P_BTTS",
            ]
            if c in board.columns
        ]
        board_display = board[summary_cols].copy()
        for c in ["P_Home", "P_Draw", "P_Away", "Confidence", "P_Over2.5", "P_BTTS"]:
            if c in board_display.columns:
                board_display[c] = pd.to_numeric(board_display[c], errors="coerce") * 100.0
        st.dataframe(
            board_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P_Home": st.column_config.NumberColumn(format="%.1f%%"),
                "P_Draw": st.column_config.NumberColumn(format="%.1f%%"),
                "P_Away": st.column_config.NumberColumn(format="%.1f%%"),
                "Confidence": st.column_config.NumberColumn(format="%.1f%%"),
                "P_Over2.5": st.column_config.NumberColumn(format="%.1f%%"),
                "P_BTTS": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        st.download_button(
            "Download all predictions",
            pred.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv",
        )

        if len(board):
            options = {
                f"{r['League']} · {r['HomeTeam']} vs {r['AwayTeam']} · {str(r.get('Date', ''))[:10]}": idx
                for idx, r in board.iterrows()
            }
            chosen = st.selectbox("Inspect a match", list(options.keys()))
            idx = options[chosen]
            row = pred.loc[idx]

            audit_row = None
            audit = st.session_state.pred_audit
            if audit is not None and len(audit):
                match = audit[
                    (audit.get("League") == row.get("League"))
                    & (audit.get("HomeTeam") == row.get("HomeTeam"))
                    & (audit.get("AwayTeam") == row.get("AwayTeam"))
                ]
                if len(match):
                    audit_row = match.iloc[0]
            _display_prediction_detail(row, audit_row)


# -----------------------------------------------------------------------------
# Performance
# -----------------------------------------------------------------------------
elif page == "Performance":
    perf = _performance_frame(artifact)
    st.markdown('<div class="section-label">Untouched test performance</div>', unsafe_allow_html=True)
    st.caption("These metrics come from the newest holdout block that was not used to fit or calibrate the model.")

    league_choice = st.selectbox(
        "Focus league",
        ["All"] + perf["LeagueCode"].tolist(),
        format_func=lambda x: "All leagues" if x == "All" else f"{x} — {LEAGUES.get(x, x)}",
    )
    subset = perf if league_choice == "All" else perf[perf["LeagueCode"] == league_choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", _fmt_pct(pd.to_numeric(subset["Accuracy"], errors="coerce").mean()))
    c2.metric("Final log loss", _fmt_num(pd.to_numeric(subset["Final Log Loss"], errors="coerce").mean(), 3))
    c3.metric("Home goal MAE", _fmt_num(pd.to_numeric(subset["Home Goal MAE"], errors="coerce").mean(), 2))
    c4.metric("Away goal MAE", _fmt_num(pd.to_numeric(subset["Away Goal MAE"], errors="coerce").mean(), 2))

    log_cols = ["Pure Model Log Loss", "Market Log Loss", "Final Log Loss"]
    chart_data = subset[["League"] + log_cols].melt(id_vars="League", var_name="System", value_name="Log loss")
    fig = px.bar(chart_data, x="League", y="Log loss", color="System", barmode="group")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    xg_cols = ["League", "Home xG Std", "Away xG Std", "Goal Margin Std"]
    xg_chart = subset[xg_cols].melt(id_vars="League", var_name="Spread", value_name="Std")
    fig2 = px.bar(xg_chart, x="League", y="Std", color="Spread", barmode="group")
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("xG standard deviations help reveal whether goal forecasts are collapsing toward the same value across fixtures.")

    st.markdown('<div class="section-label">Walk-forward model health</div>', unsafe_allow_html=True)
    health_rows = []
    fold_rows = []
    for code, m in _artifact_metrics(artifact).items():
        wf = m.get("walk_forward") or {}
        health_rows.append({
            "League": LEAGUES.get(code, code), "Status": wf.get("status", "UNKNOWN"),
            "Recent log loss": wf.get("recent_log_loss"), "Prior log loss": wf.get("prior_log_loss"),
            "Log-loss drift": wf.get("log_loss_drift"), "Accuracy drift": wf.get("accuracy_drift"),
            "Retrain recommended": wf.get("retrain_recommended", False),
        })
        for fr in wf.get("folds", []):
            fold_rows.append({"League": LEAGUES.get(code, code), **fr})
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)
    if fold_rows:
        with st.expander("Walk-forward folds", expanded=False):
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)

    st.dataframe(perf, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
elif page == "Backtest":
    bt = _betting_frame(artifact)
    st.markdown('<div class="section-label">Untouched-test betting backtest</div>', unsafe_allow_html=True)
    st.caption("Opening odds are used for simulated bets; closing prices are reserved for CLV measurement.")

    bt_display = bt.copy()
    for c in ["ROI", "Hit Rate", "Max Drawdown %", "Mean CLV", "Positive CLV", "Edge Threshold"]:
        if c in bt_display.columns:
            bt_display[c] = pd.to_numeric(bt_display[c], errors="coerce") * 100.0
    st.dataframe(
        bt_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ROI": st.column_config.NumberColumn(format="%.1f%%"),
            "Hit Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Max Drawdown %": st.column_config.NumberColumn(format="%.1f%%"),
            "Mean CLV": st.column_config.NumberColumn(format="%.1f%%"),
            "Positive CLV": st.column_config.NumberColumn(format="%.1f%%"),
            "Edge Threshold": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    engines = _artifact_engines(artifact)
    league = st.selectbox(
        "Inspect league ledger",
        list(engines),
        format_func=lambda x: f"{x} — {LEAGUES.get(x, x)}",
    )
    ledger = getattr(engines[league].model, "backtest_ledger_", pd.DataFrame())
    if ledger is None or ledger.empty:
        st.info("No qualifying bets were stored for this league's untouched test block.")
    else:
        if "CumulativeProfitUnits" in ledger.columns:
            plot = ledger.copy()
            plot["Bet #"] = np.arange(1, len(plot) + 1)
            fig = px.line(plot, x="Bet #", y="CumulativeProfitUnits", title="Cumulative profit")
            fig.update_layout(height=370, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        st.download_button(
            f"Download {league} ledger",
            ledger.to_csv(index=False).encode(),
            f"{league}_betting_ledger.csv",
            "text/csv",
        )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
elif page == "Training":
    if PUBLIC_APP:
        st.error("Training is disabled in the public deployment. Train on your GPU machine and deploy the saved model artifact.")
        st.stop()
    st.markdown('<div class="section-label">Train independent league models</div>', unsafe_allow_html=True)
    st.warning("Training can be computationally heavy. You do not need to retrain every day; daily prediction refreshes recent results separately.")

    with st.container(border=True):
        leagues = st.multiselect(
            "Leagues",
            list(LEAGUES),
            default=DEFAULT_MODEL_LEAGUES,
            format_func=lambda x: f"{x} — {LEAGUES[x]}",
        )
        c1, c2 = st.columns(2)
        seasons_text = c1.text_input("Recent seasons to try first", "2223,2324,2425,2526,2627")
        target = c2.number_input("Minimum matches per league", min_value=500, max_value=10000, value=5000, step=500)
        c3, c4 = st.columns(2)
        backend = c3.selectbox("ML backend", ["auto", "xgboost", "sklearn"], index=0)
        accelerator = c4.selectbox("Accelerator", ["auto", "gpu", "cpu"], index=0)
        auto_source = st.checkbox("Auto-source available historical advanced data", value=True)
        out_path = st.text_input("Save model as", "models/top5_5000_each_v11.joblib")

        try:
            compute = resolve_compute(backend, accelerator)
            st.info(f"Resolved compute: {compute.backend} / {compute.device} — {compute.note}")
        except Exception as exc:
            st.error(str(exc))

        if st.button("Download history + train", type="primary", use_container_width=True):
            if not leagues:
                st.error("Select at least one league.")
            else:
                seasons = [s.strip() for s in seasons_text.split(",") if s.strip()]
                with st.status("Training model bundle", expanded=True) as status:
                    st.write("Downloading independent league histories...")
                    histories, used, errors = download_independent_league_histories(
                        leagues,
                        seasons,
                        target_matches_per_league=int(target),
                    )
                    if errors:
                        st.warning(str(errors))

                    engines = {}
                    progress = st.progress(0.0)
                    for i, code in enumerate(leagues, 1):
                        data = histories[code]
                        st.write(f"**{code} — {LEAGUES[code]}:** {len(data):,} completed matches")
                        rich = None
                        if auto_source:
                            adv_dir = Path("advanced_data_auto") / "training" / code
                            st.write(f"Auto-sourcing advanced data for {code}...")
                            AutoSourceManager(adv_dir).sync(
                                data,
                                purpose="training",
                                use_statsbomb_open=True,
                                use_sportmonks=False,
                                use_news_fallback=False,
                            )
                            rich = load_advanced_bundle(str(adv_dir))
                        st.write(f"Training {code} model...")
                        engines[code] = SoccerPredictionEngine(
                            backend=backend,
                            accelerator=accelerator,
                        ).fit(data, advanced_data=rich)
                        progress.progress(i / len(leagues))

                    bundle = LeagueModelBundle(
                        engines=engines,
                        target_matches_per_league=int(target),
                    )
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    bundle.save(out_path)
                    st.session_state.bundle = bundle
                    st.session_state.model_path = out_path
                    st.session_state.pred = None
                    st.session_state.pred_audit = None
                    status.update(label=f"Training complete — saved to {out_path}", state="complete")
                    st.success("Model bundle is active in this app session.")


# -----------------------------------------------------------------------------
# Data Health
# -----------------------------------------------------------------------------
elif page == "Data Health":
    st.markdown('<div class="section-label">Data health and model state</div>', unsafe_allow_html=True)
    st.caption("Use this page to verify what information the model actually had instead of assuming every advanced feed was populated.")

    engines = _artifact_engines(artifact)
    rows = []
    for code, engine in engines.items():
        rows.append(
            {
                "LeagueCode": code,
                "League": LEAGUES.get(code, code),
                "State as of": getattr(engine, "state_asof_date", None),
                "Known teams": len((getattr(engine, "team_leagues_", {}) or {}).keys()),
                "Compute": (engine.metrics or {}).get("compute_device"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    cov = _coverage_frame(artifact)
    if cov.empty:
        st.info("No advanced-coverage diagnostics were stored in this artifact.")
    else:
        league_choice = st.selectbox(
            "Coverage league",
            cov["LeagueCode"].unique().tolist(),
            format_func=lambda x: f"{x} — {LEAGUES.get(x, x)}",
        )
        one = cov[cov["LeagueCode"] == league_choice].sort_values("Coverage", ascending=False)
        fig = px.bar(one, x="Coverage", y="Feature", orientation="h", range_x=[0, 1], text_auto=".0%")
        fig.update_layout(height=max(380, 32 * len(one)), margin=dict(l=10, r=10, t=20, b=10), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(one, use_container_width=True, hide_index=True)

    if not PUBLIC_APP:
        st.markdown('<div class="section-label">Local data directories</div>', unsafe_allow_html=True)
        checks = []
        for path in [Path("advanced_data_auto/training"), Path("advanced_data_auto/live"), Path("models")]:
            checks.append({"Path": str(path), "Exists": path.exists(), "Files": sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0})
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    else:
        st.caption("Local filesystem details are hidden in public mode.")

