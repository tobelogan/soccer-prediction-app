from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping, Any

import requests


def truthy(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "public"}


def setting(name: str, secrets: Mapping[str, Any] | None = None, default: str | None = None) -> str | None:
    """Read a deployment setting from Streamlit secrets first, then environment."""
    if secrets is not None:
        try:
            if name in secrets:
                value = secrets[name]
                if value is not None and str(value).strip():
                    return str(value).strip()
        except Exception:
            pass
    value = os.getenv(name)
    if value is not None and str(value).strip():
        return str(value).strip()
    return default


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_remote_model(
    url: str,
    destination: str | Path = ".cache/public_model.joblib",
    expected_sha256: str | None = None,
    timeout: int = 180,
) -> Path:
    """Download a trusted model artifact atomically and optionally verify SHA-256.

    The URL must point directly to the model bytes (for example a GitHub Release
    asset or an object-storage/Hugging Face download URL), not to an HTML page.
    """
    if not str(url).lower().startswith(("https://", "http://")):
        raise ValueError("MODEL_URL must be an http(s) direct-download URL")

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = (expected_sha256 or "").strip().lower()

    if dest.exists() and dest.stat().st_size > 1024:
        if not expected or sha256_file(dest) == expected:
            return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    headers = {"User-Agent": "SoccerPredictionLab-v11-public/1.0"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
        response.raise_for_status()
        ctype = str(response.headers.get("content-type", "")).lower()
        if "text/html" in ctype:
            raise RuntimeError("MODEL_URL returned HTML instead of a model file. Use a direct-download URL.")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    if not tmp.exists() or tmp.stat().st_size <= 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model file is unexpectedly small")
    if expected:
        actual = sha256_file(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"MODEL_SHA256 mismatch: expected {expected}, got {actual}")
    tmp.replace(dest)
    return dest
