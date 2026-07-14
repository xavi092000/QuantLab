import subprocess
import sys
import time
from datetime import datetime


PIPELINE_STEPS = [
    "ml/quant_metrics_engine.py",
    "ml/live_return_signal_engine.py",
    "ml/strategy_router.py",
    "ml/adaptive_strategy_engine.py",
    "ml/momentum_strategy_engine.py",
    "ml/final_strategy_decision_engine.py",
    "ml/portfolio_construction_engine.py",
    "ml/risk_management_v2.py",
    "ml/trade_execution_engine.py",
    "ml/position_monitor_engine.py",
    "ml/equity_curve_engine.py",
    "ml/trade_performance_engine.py",
    "ml/strategy_attribution_engine.py",
    "ml/strategy_analytics_engine.py",
    "ml/regime_analytics_engine.py",
    "ml/meta_strategy_engine.py",
    "ml/strategy_selection_engine.py",
]


def run_step(script_path):
    print("")
    print("=" * 50)
    print(f"[{datetime.now()}]")
    print(f"RUNNING: {script_path}")
    print("=" * 50)

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)

        raise RuntimeError(
            f"Pipeline stopped because {script_path} failed "
            f"with exit code {result.returncode}."
        )

    print(f"SUCCESS: {script_path}")


def main():
    print("")
    print("====================================")
    print("QUANTLAB AUTONOMOUS PIPELINE")
    print("====================================")

    try:
        while True:
            for step in PIPELINE_STEPS:
                run_step(step)

            print("")
            print("------------------------------------")
            print("Sleeping 60 seconds...")
            print("------------------------------------")
            time.sleep(60)

    except KeyboardInterrupt:
        print("")
        print("QuantLab pipeline stopped.")


if __name__ == "__main__":
    main()





