# Deploy Soccer Prediction Lab v11 on Streamlit Community Cloud

This folder is ready to push to GitHub and deploy with `app.py` as the entrypoint.

## 1. Train locally

Keep training on your own GPU machine. The public app is prediction-only.

```bat
python train.py --seasons 2223 2324 2425 2526 2627 --target-matches-per-league 5000 --backend xgboost --accelerator auto --out models/top5_5000_each_v11.joblib
```

## 2. Choose how the public app gets the model

### Option A — small enough to keep in the repository

The default `.gitignore` excludes `*.joblib`. If you intentionally want to commit the model, remove that rule and add the model file under `models/`.

### Option B — recommended for a large model bundle

Upload the trained `.joblib` to a trusted direct-download host, such as a GitHub Release asset or your own object storage. Then configure:

```toml
PUBLIC_APP = "true"
MODEL_URL = "https://.../top5_5000_each_v11.joblib"
MODEL_SHA256 = "your-sha256"
```

Generate the hash locally:

```bat
python -c "import hashlib; p='models/top5_5000_each_v11.joblib'; h=hashlib.sha256(); h.update(open(p,'rb').read()); print(h.hexdigest())"
```

`MODEL_URL` must return the model bytes directly, not an HTML landing page.

## 3. Push this folder to GitHub

Do not commit `.streamlit/secrets.toml` or API tokens.

## 4. Deploy on Streamlit Community Cloud

Create an app from the GitHub repository and set the entrypoint to:

```text
app.py
```

In **Advanced settings**, use the same Python version you use locally where possible and paste the values from `.streamlit/secrets.toml.example` into the Secrets box (with your real values).

## Public-mode behavior

When `PUBLIC_APP=true`:

- training controls are hidden;
- arbitrary `.joblib` upload is disabled;
- local filesystem/model paths are hidden;
- the app uses only the trusted bundled model or `MODEL_URL`;
- predictions, performance, backtest summaries, and data-health information remain available.

This is intentional. `joblib`/pickle artifacts must never be accepted from untrusted visitors.
