from soccer_predictor import compute


def test_auto_uses_cuda_when_visible(monkeypatch):
    monkeypatch.setattr(compute, "_nvidia_gpu_visible", lambda: True)
    monkeypatch.setattr(compute, "_xgboost_cuda_build", lambda: (True, True))
    cfg = compute.resolve_compute("auto", "auto")
    assert cfg.backend == "xgboost"
    assert cfg.device == "cuda"


def test_auto_falls_back_to_cpu_without_visible_gpu(monkeypatch):
    monkeypatch.setattr(compute, "_nvidia_gpu_visible", lambda: False)
    monkeypatch.setattr(compute, "_xgboost_cuda_build", lambda: (True, True))
    cfg = compute.resolve_compute("auto", "auto")
    assert cfg.backend == "xgboost"
    assert cfg.device == "cpu"
