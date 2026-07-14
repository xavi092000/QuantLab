from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "artifacts" / "live_engine"
INGESTION_LOG_PATH = LOG_DIR / "ingestion.log"

INGESTION_MODULE = "ingestion.quantlab_live_ingestion"
ORCHESTRATOR_MODULE = "tools.pipeline_live_tracker"
SYSTEM_METRICS_MODULE = "tools.system_metrics"
TRADING_METRICS_MODULE = "tools.trading_metrics"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"[{timestamp()}] {title}")
    print("=" * 72, flush=True)


def run_module(
    module_name: str,
    extra_args: Sequence[str] | None = None,
    timeout_seconds: int | None = None,
) -> float:
    command = [sys.executable, "-u", "-m", module_name]
    if extra_args:
        command.extend(extra_args)

    started = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )

    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        raise RuntimeError(
            f"{module_name} failed with exit code {result.returncode}."
        )

    return elapsed


def start_ingestion() -> tuple[subprocess.Popen[str], TextIO]:
    print_header("STARTING LIVE MARKET INGESTION")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = INGESTION_LOG_PATH.open(
        "a",
        encoding="utf-8",
        buffering=1,
    )

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        [sys.executable, "-u", "-m", INGESTION_MODULE],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    print(f"Ingestion PID        : {process.pid}")
    print(f"Ingestion log        : {INGESTION_LOG_PATH}")
    print("Console mode         : quiet", flush=True)

    return process, log_handle


def stop_process(
    process: subprocess.Popen[str] | None,
    log_handle: TextIO | None,
) -> None:
    if process is not None and process.poll() is None:
        print_header("STOPPING LIVE MARKET INGESTION")

        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()

            process.wait(timeout=10)

        except Exception:
            process.kill()
            process.wait(timeout=5)

    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def wait_for_ingestion(
    process: subprocess.Popen[str],
    startup_seconds: int,
) -> None:
    print(f"Waiting {startup_seconds}s for fresh market data...", flush=True)

    deadline = time.time() + startup_seconds

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Live ingestion stopped during startup "
                f"with exit code {process.returncode}. "
                f"Check {INGESTION_LOG_PATH}."
            )
        time.sleep(1)

    print("Ingestion startup check: PASS", flush=True)


def refresh_metrics(iterations: int) -> None:
    print_header("REFRESHING SYSTEM METRICS")
    run_module(
        SYSTEM_METRICS_MODULE,
        [
            "--iterations",
            str(iterations),
            "--store-db",
        ],
        timeout_seconds=300,
    )

    print_header("REFRESHING TRADING METRICS")
    run_module(
        TRADING_METRICS_MODULE,
        ["--store-db"],
        timeout_seconds=180,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run QuantLab as a live supervised service: market ingestion, "
            "one-cycle orchestration, and metric refresh."
        )
    )

    parser.add_argument("--cycle-seconds", type=float, default=60.0)
    parser.add_argument("--startup-seconds", type=int, default=15)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--metric-iterations", type=int, default=20)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--once", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cycle_seconds <= 0:
        raise ValueError("--cycle-seconds must be greater than zero.")
    if args.startup_seconds < 0:
        raise ValueError("--startup-seconds cannot be negative.")
    if args.metrics_every <= 0:
        raise ValueError("--metrics-every must be greater than zero.")
    if args.metric_iterations <= 0:
        raise ValueError("--metric-iterations must be greater than zero.")

    ingestion_process: subprocess.Popen[str] | None = None
    ingestion_log_handle: TextIO | None = None
    cycle_number = 0

    print()
    print("=" * 72)
    print("QUANTLAB LIVE ENGINE")
    print("=" * 72)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Cycle target        : {args.cycle_seconds:.2f}s")
    print(f"Metrics every       : {args.metrics_every} cycle(s)")
    print(f"Start ingestion     : {not args.skip_ingestion}")
    print("=" * 72, flush=True)

    try:
        if not args.skip_ingestion:
            ingestion_process, ingestion_log_handle = start_ingestion()
            wait_for_ingestion(
                ingestion_process,
                startup_seconds=args.startup_seconds,
            )

        while True:
            if (
                ingestion_process is not None
                and ingestion_process.poll() is not None
            ):
                raise RuntimeError(
                    "Live ingestion stopped unexpectedly "
                    f"with exit code {ingestion_process.returncode}. "
                    f"Check {INGESTION_LOG_PATH}."
                )

            cycle_number += 1
            cycle_started = time.perf_counter()

            print_header(f"LIVE CYCLE {cycle_number}")
            print("Running QuantLab orchestrator...", flush=True)

            pipeline_elapsed = run_module(
                ORCHESTRATOR_MODULE,
                ["--once"],
                timeout_seconds=300,
            )

            print(
                f"Pipeline cycle {cycle_number} completed "
                f"in {pipeline_elapsed:.2f}s.",
                flush=True,
            )

            if (
                not args.skip_metrics
                and cycle_number % args.metrics_every == 0
            ):
                refresh_metrics(args.metric_iterations)

            total_elapsed = time.perf_counter() - cycle_started

            if args.once:
                print_header("ONE-CYCLE LIVE RUN COMPLETE")
                break

            sleep_seconds = max(
                0.0,
                args.cycle_seconds - total_elapsed,
            )

            print(
                f"Total live-cycle time: {total_elapsed:.2f}s | "
                f"Sleeping: {sleep_seconds:.2f}s",
                flush=True,
            )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print()
        print("QuantLab live engine stopped by user.", flush=True)

    finally:
        stop_process(
            ingestion_process,
            ingestion_log_handle,
        )


if __name__ == "__main__":
    main()
