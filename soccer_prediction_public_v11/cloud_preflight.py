from __future__ import annotations

import argparse
from pathlib import Path

from src.soccer_predictor.deployment import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a trained model before Streamlit Cloud deployment.")
    parser.add_argument("--model", default="models/top5_5000_each_v11.joblib")
    args = parser.parse_args()
    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    digest = sha256_file(path)
    print(f"Model:  {path}")
    print(f"Size:   {size_mb:.1f} MB")
    print(f"SHA256: {digest}")
    print()
    if size_mb >= 90:
        print("Recommendation: host the model as a direct-download asset and set MODEL_URL + MODEL_SHA256 in Streamlit Secrets.")
    else:
        print("The model may be practical to keep with the deployment, but check your Git host's file-size policy before committing it.")


if __name__ == "__main__":
    main()
