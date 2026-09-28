from __future__ import annotations

import hashlib
from pathlib import Path

from soccer_predictor.deployment import setting, sha256_file, truthy


def test_truthy_and_setting_precedence(monkeypatch):
    monkeypatch.setenv("DEMO_FLAG", "env-value")
    assert truthy("true") is True
    assert truthy("0") is False
    assert setting("DEMO_FLAG", {"DEMO_FLAG": "secret-value"}) == "secret-value"
    assert setting("DEMO_FLAG", {}) == "env-value"


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "model.joblib"
    p.write_bytes(b"trusted-model-bytes")
    assert sha256_file(p) == hashlib.sha256(b"trusted-model-bytes").hexdigest()
