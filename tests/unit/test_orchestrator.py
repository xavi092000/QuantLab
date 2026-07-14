from types import SimpleNamespace

import pytest

import ml.quantlab_orchestrator as orchestrator


def test_run_step_invokes_module_from_project_root(monkeypatch) -> None:
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            stdout="ok\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    orchestrator.run_step("ml.quant_metrics_engine")

    assert captured["args"] == [
        orchestrator.sys.executable,
        "-m",
        "ml.quant_metrics_engine",
    ]
    assert captured["kwargs"]["cwd"] == orchestrator.PROJECT_ROOT
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True


def test_run_step_raises_when_stage_fails(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="",
            stderr="boom",
            returncode=1,
        )

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        orchestrator.run_step("ml.quant_metrics_engine")
