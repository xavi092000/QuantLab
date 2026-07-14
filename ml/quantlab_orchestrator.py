import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    "ml.quant_metrics_engine",
    "ml.live_return_signal_engine",
    "ml.strategy_router",
    "ml.adaptive_strategy_engine",
    "ml.momentum_strategy_engine",
    "ml.final_strategy_decision_engine",
    "ml.portfolio_construction_engine",
    "ml.risk_management_v2",
    "ml.trade_execution_engine",
    "ml.position_monitor_engine",
    "ml.equity_curve_engine",
    "ml.trade_performance_engine",
    "ml.strategy_attribution_engine",
    "ml.strategy_analytics_engine",
    "ml.regime_analytics_engine",
    "ml.meta_strategy_engine",
    "ml.strategy_selection_engine",
]


def run_step(module_name: str) -> None:
    """Run one pipeline stage as a Python module from the project root."""
    print()
    print("=" * 50)
    print(f"[{datetime.now()}]")
    print(f"RUNNING: {module_name}")
    print("=" * 50)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    if result.returncode != 0:
        if result.stderr:
            print("ERRORS:")
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

        raise RuntimeError(
            f"Pipeline stopped because {module_name} failed "
            f"with exit code {result.returncode}."
        )

    if result.stderr:
        print("WARNINGS:")
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    print(f"SUCCESS: {module_name}")


def run_cycle() -> None:
    """Execute one complete QuantLab pipeline cycle."""
    cycle_started_at = datetime.now()

    print()
    print("=" * 50)
    print(f"CYCLE START: {cycle_started_at}")
    print("=" * 50)

    for step in PIPELINE_STEPS:
        run_step(step)

    elapsed = (datetime.now() - cycle_started_at).total_seconds()

    print()
    print("=" * 50)
    print(f"CYCLE COMPLETE in {elapsed:.2f} seconds")
    print("=" * 50)


def main() -> None:
    """Run the QuantLab pipeline continuously every 60 seconds."""
    print()
    print("====================================")
    print("QUANTLAB AUTONOMOUS PIPELINE")
    print("====================================")

    try:
        while True:
            cycle_started_at = time.monotonic()
            run_cycle()

            elapsed = time.monotonic() - cycle_started_at
            sleep_seconds = max(0.0, 60.0 - elapsed)

            print()
            print("------------------------------------")
            print(
                f"Cycle duration: {elapsed:.2f} seconds | "
                f"Sleeping: {sleep_seconds:.2f} seconds"
            )
            print("------------------------------------")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print()
        print("QuantLab pipeline stopped.")


if __name__ == "__main__":
    main()
