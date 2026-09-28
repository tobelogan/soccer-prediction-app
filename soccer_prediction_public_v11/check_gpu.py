from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from soccer_predictor.compute import resolve_compute

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    print(f"XGBoost CUDA build: {bool(xgb.build_info().get('USE_CUDA', False))}")
except Exception as exc:
    print(f"XGBoost: unavailable ({exc})")

for accelerator in ["auto", "gpu", "cpu"]:
    try:
        cfg = resolve_compute("auto", accelerator)
        print(f"accelerator={accelerator}: backend={cfg.backend}, device={cfg.device}, note={cfg.note}")
    except Exception as exc:
        print(f"accelerator={accelerator}: ERROR: {exc}")
