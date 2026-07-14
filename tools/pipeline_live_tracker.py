from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = PROJECT_ROOT / "artifacts" / "live_engine"
STATUS_PATH = STATUS_DIR / "pipeline_status.json"

PIPELINE_STEPS = [
    ("Quant Metrics", "ml.quant_metrics_engine"),
    ("Hybrid Return Signal", "ml.live_return_signal_engine"),
    ("Strategy Router", "ml.strategy_router"),
    ("Adaptive Strategy", "ml.adaptive_strategy_engine"),
    ("Momentum Strategy", "ml.momentum_strategy_engine"),
    ("Final Decision", "ml.final_strategy_decision_engine"),
    ("Portfolio Construction", "ml.portfolio_construction_engine"),
    ("Risk Management", "ml.risk_management_v2"),
    ("Trade Execution", "ml.trade_execution_engine"),
    ("Position Monitor", "ml.position_monitor_engine"),
    ("Equity Curve", "ml.equity_curve_engine"),
    ("Trade Performance", "ml.trade_performance_engine"),
    ("Strategy Attribution", "ml.strategy_attribution_engine"),
    ("Strategy Analytics", "ml.strategy_analytics_engine"),
    ("Regime Analytics", "ml.regime_analytics_engine"),
    ("Meta Strategy", "ml.meta_strategy_engine"),
    ("Strategy Selection", "ml.strategy_selection_engine"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: dict[str, Any]) -> None:
    """Write pipeline status without crashing on transient Windows locks."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = STATUS_DIR / (
        f"pipeline_status.{os.getpid()}.{time.time_ns()}.tmp"
    )

    temp_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    max_attempts = 40
    retry_delay_seconds = 0.05

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                os.replace(temp_path, STATUS_PATH)
                return
            except PermissionError:
                if attempt == max_attempts:
                    print(
                        "WARNING: pipeline status file remained locked; "
                        "this dashboard refresh will be skipped.",
                        flush=True,
                    )
                    return
                time.sleep(retry_delay_seconds)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def initial_state(cycle_id: str) -> dict[str, Any]:
    return {
        "cycle_id": cycle_id,
        "cycle_status": "RUNNING",
        "cycle_started_at_utc": utc_now(),
        "cycle_finished_at_utc": None,
        "cycle_duration_seconds": None,
        "current_step": None,
        "completed_steps": 0,
        "total_steps": len(PIPELINE_STEPS),
        "steps": [
            {
                "order": index,
                "name": name,
                "module": module,
                "status": "PENDING",
                "started_at_utc": None,
                "finished_at_utc": None,
                "duration_seconds": None,
                "return_code": None,
                "error": None,
            }
            for index, (name, module) in enumerate(PIPELINE_STEPS, start=1)
        ],
    }


def run_pipeline_once() -> int:
    cycle_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    state = initial_state(cycle_id)
    cycle_started = time.perf_counter()
    atomic_write_json(state)

    print()
    print("=" * 72)
    print("QUANTLAB LIVE PIPELINE TRACKER")
    print("=" * 72)
    print(f"Cycle ID : {cycle_id}")
    print(f"Status   : {STATUS_PATH}")
    print("=" * 72)

    for index, (name, module) in enumerate(PIPELINE_STEPS):
        step = state["steps"][index]
        step["status"] = "RUNNING"
        step["started_at_utc"] = utc_now()
        state["current_step"] = name
        atomic_write_json(state)

        print()
        print("-" * 72)
        print(f"[{index + 1}/{len(PIPELINE_STEPS)}] {name}")
        print("-" * 72)

        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-u", "-m", module],
            cwd=PROJECT_ROOT,
            text=True,
            check=False,
        )
        elapsed = round(time.perf_counter() - started, 3)

        step["duration_seconds"] = elapsed
        step["finished_at_utc"] = utc_now()
        step["return_code"] = result.returncode

        if result.returncode == 0:
            step["status"] = "PASS"
            state["completed_steps"] += 1
            atomic_write_json(state)
            print(f"PASS: {name} ({elapsed:.3f}s)")
            continue

        step["status"] = "FAIL"
        step["error"] = f"Exit code {result.returncode}"
        state["cycle_status"] = "FAIL"
        state["current_step"] = name
        state["cycle_finished_at_utc"] = utc_now()
        state["cycle_duration_seconds"] = round(
            time.perf_counter() - cycle_started,
            3,
        )
        atomic_write_json(state)

        print(f"FAIL: {name} ({elapsed:.3f}s)")
        return result.returncode

    state["cycle_status"] = "PASS"
    state["current_step"] = None
    state["cycle_finished_at_utc"] = utc_now()
    state["cycle_duration_seconds"] = round(
        time.perf_counter() - cycle_started,
        3,
    )
    atomic_write_json(state)

    print()
    print("=" * 72)
    print(
        f"PIPELINE PASS — {state['completed_steps']}/{state['total_steps']} "
        f"steps in {state['cycle_duration_seconds']:.3f}s"
    )
    print("=" * 72)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one tracked QuantLab pipeline cycle."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Accepted for compatibility with quantlab_live_engine.",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit(run_pipeline_once())


if __name__ == "__main__":
    main()
