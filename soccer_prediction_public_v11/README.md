# Soccer Betting Predictor v10 — Live Form + Goal-Spread Calibration

v10 addresses two observed production failures: **stale recent form** and **expected-goal forecasts collapsing toward similar league-average values**. It keeps the independent 5,000+ match models for EPL, La Liga, Bundesliga, Serie A and Ligue 1, but changes how current strength and goals are modeled.

## What changed in v10

- Before every live prediction, the app/CLI downloads completed league matches newer than each saved model's state date and advances Elo/form without refitting the trees. Football-Data is preferred; ESPN final scores are a freshness fallback.
- Adds last-3, last-5, last-10 and exponentially weighted goals-for/goals-against features, last-5/10 PPG, last-5 goal difference and shots-on-target form.
- Replaces lifetime league scoring priors with a rolling recent-league window so a 2012 scoring environment does not define 2026.
- Training uses exponential time decay (default two-year half-life with a small floor), keeping 5,000+ matches but giving recent seasons much more influence.
- Expected-goal regressors no longer use bookmaker probability columns. This prevents missing future odds from being median-imputed and compressing many fixture xG forecasts toward the same value.
- A separate recent-form goal estimate is calibrated against the ML xG model on the calibration block. The blend weight is learned independently for home and away goals rather than hard-coded.
- Prediction output now includes `Base_Home_xG`, `Base_Away_xG`, `Form_Home_xG`, `Form_Away_xG`, `Expected_Goal_Margin`, and probabilities of 3+ / 4+ goal wins.
- The Streamlit app includes a **Prediction audit** expander showing the exact recent-form inputs used for each fixture.

Expected goals are a mean scoring forecast, not a promise of the exact score. A team that has recently won 5-0 can still have a 2.5-3.5 xG forecast next time depending on opponent strength. For potential blowouts, use the new `P_HomeWin3Plus`, `P_HomeWin4Plus`, `P_AwayWin3Plus`, and `P_AwayWin4Plus` columns alongside xG.

## Retrain for the full v10 model

The live-state refresh is backward-compatible with a v9 bundle, but the new 3/5/10-match features, time decay, market-free xG models and calibrated form/xG blend require retraining:

```bash
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator auto --out models/top5_5000_each_v10.joblib
```

Then run the app:

```bash
streamlit run app.py
```

or automatic CLI prediction:

```bash
python predict.py --model models/top5_5000_each_v10.joblib --out predictions.csv
```

During training, inspect `home_xg_std`, `away_xg_std`, and `goal_margin_std`. Extremely tiny values indicate forecast compression. Also inspect `home_form_blend_weight` / `away_form_blend_weight`; they are selected only on calibration data.

---

# Historical v9 notes

v9 keeps the v8 architecture: **five fully independent league models**, each trained on **at least 5,000 matches from its own league**. It adds three major changes:

1. an honest opening-odds betting backtest on the untouched newest test block;
2. richer historical web/API enrichment (weather/location, StatsBomb tactical proxies, observed player-movement transfers, optional timestamped historical news, and automatic Sportmonks history when a token is configured);
3. a fix for the XGBoost CUDA prediction device-mismatch warning while retaining GPU tree training.

Default leagues:

- `E0` — English Premier League
- `SP1` — Spain La Liga
- `D1` — Germany Bundesliga
- `I1` — Italy Serie A
- `F1` — France Ligue 1

Each league is fitted independently. EPL rows never fit La Liga trees and vice versa.

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU check:

```bat
nvidia-smi
python check_gpu.py
```

## Train all five independent 5,000+ match models

```bat
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator auto --out models/top5_5000_each.joblib
```

To require CUDA instead of permitting CPU fallback:

```bat
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator gpu --out models/top5_5000_each.joblib
```

Historical match-day weather backfill is enabled by default. Disable it with:

```bat
--no-historical-weather
```

If `SPORTMONKS_API_TOKEN` is configured, Sportmonks historical enrichment is now automatically attempted. You can disable it explicitly:

```bat
--no-sportmonks-history
```

The conservative historical Google News archive backfill is optional because archive search coverage is incomplete and it can be slow/rate-limited. Enable it with:

```bat
--historical-news-backfill --max-historical-news-queries 120
```

The historical-news source only promotes evidence published **before** the historical fixture/event date. It never uses a current article as evidence for an old match.

## Historical enrichment in v9

### Football-Data

Still provides the long match history, base statistics, opening/market odds, closing odds when present, and schedule context.

### StatsBomb Open Data

Where competitions/seasons overlap, v9 extracts:

- shot-level xG and npxG;
- player xG and xA;
- actual starting XIs;
- player minutes;
- goalkeeper roles;
- tactical proxies derived from event locations:
  - completed-pass possession share;
  - PPDA-style pressing proxy;
  - field-tilt proxy.

The tactical rows are labelled as **proxies**, not proprietary StatsBomb tactical products.

### Derived player movements

When the same player is observed changing clubs in the sourced `player_stats.csv`, v9 records a low-confidence, auditable transfer IN/OUT row. This supplements a licensed transfer feed; it does not pretend to replace one.

### Historical weather + travel location

v9 can geocode home-team stadium/location queries through OpenStreetMap Nominatim and then retrieve historical match-day weather from Open-Meteo. The results populate:

```text
team_locations.csv
weather.csv
```

The public geocoder is queried conservatively and responses are cached.

### Sportmonks

If your token/package permits it, historical/current fixtures can add lineups, sidelined players, xG, weather and other structured information. Provider access depends on your subscription.

### Historical news context

Optional Google News RSS archive queries can add conservative evidence for:

- injuries/suspensions;
- manager appointments;
- transfers.

Promoted rows include publication timestamp, URL, source and confidence. Search archives can be incomplete, so zero coverage is not treated as an error.

## Leakage protection

v9 makes a stricter distinction between information available at simulated bet entry and information known only later.

Historical closing/sharp/exchange prices and close-vs-open line movement are **excluded from the fitted ML feature set**. They remain available for:

- market benchmark comparison;
- live/current market blending when actually available;
- closing-line-value measurement after a simulated opening bet.

This prevents the ML model's untouched-test performance from benefiting from historical closing information that would not have been known at opening time.

## GPU change

Training still uses XGBoost CUDA (`tree_method=hist`, `device=cuda`) when available.

The old warning about a CUDA booster receiving CPU NumPy prediction data is handled explicitly in v9. If CuPy is installed, prediction can use GPU arrays. If it is not, v9 explicitly switches prediction to CPU host-memory prediction and restores the booster device afterwards, while training remains GPU accelerated.

CuPy is optional because the correct package depends on your CUDA runtime. It is not forced in `requirements.txt`.

## Honest betting backtest

During training, each league is still split chronologically:

```text
oldest 60%  -> fit goal + 1X2 models
next 20%    -> calibrate probabilities, market weight and betting edge threshold
newest 20%  -> untouched final evaluation and betting backtest
```

The betting backtest:

- enters at historical **opening/early 1X2 odds** only;
- selects at most one 1X2 bet per match;
- tunes only the minimum expected-value threshold on the calibration block;
- measures performance only on the untouched newest block;
- uses closing odds only **after** the bet to compute CLV.

Reported metrics include:

- bet count;
- profit in flat-stake units;
- ROI / yield per unit staked;
- hit rate;
- maximum drawdown in units;
- maximum drawdown percentage from a 100-unit reference bankroll;
- mean closing-line value;
- percentage of bets with positive CLV.

Positive CLV means the simulated opening price was better than the closing price for the same selection.

A positive backtest is **not** a guarantee of future profit. The untouched block is a much stronger check than training accuracy, but betting markets and data availability change over time.

## Export the betting ledgers

After training v9:

```bat
python backtest.py --model models/top5_5000_each.joblib --out-dir backtests
```

This creates:

```text
backtests/
  E0_betting_ledger.csv
  SP1_betting_ledger.csv
  D1_betting_ledger.csv
  I1_betting_ledger.csv
  F1_betting_ledger.csv
  summary.csv
```

Each ledger contains the historical date, fixture, selected side, model probability, opening odds, expected value, actual result, profit, closing odds and CLV.

## Predict upcoming fixtures automatically

```bat
python predict.py --model models/top5_5000_each.joblib --out predictions.csv
```

No fixture CSV is required. The program reads the league scope from the model bundle, downloads upcoming fixtures only for those trained leagues, refreshes available current pre-match data, and routes every fixture to its matching independent league model.

To look farther ahead:

```bat
python predict.py --model models/top5_5000_each.joblib --days 21 --out predictions.csv
```

Manual fixtures remain supported:

```bat
python predict.py --model models/top5_5000_each.joblib --fixtures fixtures.csv --out predictions.csv
```

## Recommended next run

Because v9 changes both the allowed training features and the honest backtest, **retrain the bundle**. Do not judge v9 using a v8 `.joblib` file.

```bat
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator auto --out models/top5_5000_each_v9.joblib
```

Then:

```bat
python backtest.py --model models/top5_5000_each_v9.joblib --out-dir backtests_v9
python predict.py --model models/top5_5000_each_v9.joblib --out predictions.csv
```

The most important numbers to compare with v8 are `pure_model_log_loss`, `market_baseline_log_loss`, `market_ensemble_log_loss`, plus the new untouched-test `roi`, `max_drawdown_pct` and `mean_clv`.

## Important limitations

- A 5,000-match league history is substantial but does not guarantee an edge over efficient markets.
- Public StatsBomb coverage is only a fraction of the full 5,000-match history in many leagues.
- Public news archives are incomplete and noisy; structured licensed feeds remain preferable for injuries, transfers and managers.
- Proprietary Opta, commercial StatsBomb and commercial Wyscout data require legitimate licensed access. This project does not bypass access controls.
- Historical Open-Meteo weather is reanalysis data; it is useful context but is not a record from the stadium's own weather station.
- Confirmed starting XIs are generally available close to kickoff, so rerunning predictions near kickoff can materially change inputs.


## v9.1 fixture-source hotfix

Football-Data's fixtures URL can temporarily return an HTML download page instead of a raw CSV. v9.1 validates the response before parsing/caching it and uses the following automatic fallback chain:

1. Football-Data raw fixture CSV when its schema is valid.
2. TheSportsDB upcoming-league schedule API for E0, SP1, D1, I1 and F1.
3. The last locally cached *valid normalized* fixture file.

The fixture parser also accepts common provider column names such as `strHomeTeam`, `strAwayTeam`, `dateEvent`, `Home`, `Away`, `Home Team`, and `Away Team`, and the model reconciles common provider team-name variants (for example `Manchester United` -> `Man United`) against the exact names learned during training.

If every live source fails, Streamlit now shows a readable error and leaves the Upload CSV route available instead of displaying a Python traceback.


## v9.2 full-fixture hotfix

The free TheSportsDB v1 `eventsnextleague` fallback is limited to one upcoming event per league, which made the app appear to predict only one game from each competition whenever Football-Data's fixture CSV was unavailable. v9.2 inserts ESPN's soccer scoreboard date-range feed ahead of TheSportsDB. The new automatic order is:

1. Football-Data weekly fixture CSV when it is a valid CSV.
2. ESPN league scoreboard for the entire selected date window (no API key).
3. TheSportsDB v1 next-league endpoint as a last-resort fallback.
4. The last valid local fixture cache.

The app also displays fixture counts per league after fetching, and completed/in-progress ESPN events are filtered out before prediction. No model retraining is required for this fixture-only hotfix.


## v9.3 fixture-provider hardening

v9.3 adds FixtureDownload full-season schedules ahead of ESPN and TheSportsDB. This specifically addresses environments where Football-Data's fixtures URL serves HTML and ESPN is blocked/returns no usable range. Provider order is now:

1. Football-Data validated fixture CSV.
2. FixtureDownload full-season CSV for each trained league.
3. ESPN date-range scoreboard.
4. TheSportsDB free next-league fallback.
5. Last valid local cache.

The app and CLI now print the actual provider and fixture count per league. If only one match per league is returned, the UI explicitly warns that the limited fallback/cache path was reached. No model retraining is required for this fixture-only update.

## v10.1 Streamlit dashboard redesign

The redesigned `app.py` keeps the v10 prediction engine unchanged and adds six pages:
Dashboard, Predictions, Performance, Backtest, Training, and Data Health.
It auto-loads `models/top5_5000_each_v10.joblib` when present and includes a detailed
match audit showing base/form/final xG, recent-form inputs, model-vs-market probabilities,
and 3+/4+ goal-margin probabilities.

Run with:

```bash
python -m streamlit run app.py
```

On Windows you can also double-click `start_app.bat`.

---

# v11 — All-competition form, player-strength deltas, walk-forward health

v11 keeps the five independent 5,000+ match league models and adds three major upgrades:

1. **All-competition recent form.** Domestic-league results still drive league Elo/priors, while recent Champions League, Europa League, Conference League and domestic-cup results refresh separate last-3/5/10 all-competition form before prediction. Cup/continental results do not contaminate league Elo.
2. **Opponent-adjusted form.** Recent goals scored/conceded are adjusted modestly by opponent Elo, so the same scoreline against a stronger opponent carries more information than against a much weaker one.
3. **Player/lineup strength.** When player feeds exist, the model now includes confirmed-XI completeness, XI attack strength versus the team expected-XI baseline, goalkeeper delta and the share of normal attacking output lost through unavailable players.
4. **Walk-forward health monitoring.** Training stores expanding-window pure-model diagnostics and a GREEN/AMBER/RED health flag. RED marks material deterioration and sets `retrain_recommended=True`.

## Train v11

```bat
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator auto --out models/top5_5000_each_v11.joblib
```

## Predict

```bat
python predict.py --model models/top5_5000_each_v11.joblib --out predictions.csv
```

Before each auto-sourced prediction run, v11 refreshes completed domestic-league results and then rebuilds recent all-competition form from available continental/domestic-cup result feeds. The Streamlit Prediction Audit shows league form, all-competition form, opponent-adjusted form and player availability strength side-by-side.

## Run the dashboard

```bat
python -m streamlit run app.py
```

The Performance page includes walk-forward fold results and a model-health status per league. A RED status is a recommendation to retrain/review data; it is not an automatic claim that a model is unusable.

## Public Streamlit deployment

A deployment-ready public mode is included. See `DEPLOY_STREAMLIT.md`.

Run this before deployment to inspect your model artifact:

```bash
python cloud_preflight.py --model models/top5_5000_each_v11.joblib
```

In Streamlit Secrets, set `PUBLIC_APP="true"`. For large models, also set a direct `MODEL_URL` and optionally `MODEL_SHA256`. Public mode hides training, local filesystem paths, and arbitrary joblib uploads.
