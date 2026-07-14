from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg2

from configs.database import DB_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURN_MODEL_PATH = PROJECT_ROOT / "ml" / "return_prediction_model.pkl"
DIRECTION_MODEL_PATH = PROJECT_ROOT / "ml" / "return_direction_model.pkl"
DEFAULT_JSON_PATH = PROJECT_ROOT / "artifacts" / "system_metrics_latest.json"


FEATURE_QUERY = """
WITH latest_quant AS (
    SELECT DISTINCT ON (symbol)
        symbol,
        metric_time,
        rsi,
        z_score,
        rolling_volatility,
        liquidity_pressure,
        market_regime
    FROM quant_metrics
    WHERE rsi IS NOT NULL
      AND z_score IS NOT NULL
      AND rolling_volatility IS NOT NULL
      AND liquidity_pressure IS NOT NULL
      AND market_regime IS NOT NULL
    ORDER BY symbol, metric_time DESC
),
latest_momentum AS (
    SELECT DISTINCT ON (symbol)
        symbol,
        momentum_5m,
        momentum_15m,
        momentum_30m
    FROM bar_momentum_features
    ORDER BY symbol, bar_time DESC
)
SELECT
    q.symbol,
    q.metric_time,
    q.rsi,
    q.z_score,
    q.rolling_volatility,
    q.liquidity_pressure,
    q.market_regime,
    COALESCE(m.momentum_5m, 0),
    COALESCE(m.momentum_15m, 0),
    COALESCE(m.momentum_30m, 0)
FROM latest_quant q
LEFT JOIN latest_momentum m ON q.symbol = m.symbol
ORDER BY q.symbol;
"""


@dataclass
class Metrics:
    generated_at_utc: str
    pipeline_stages: int
    python_modules: int
    automated_tests_collected: int
    github_workflows: int
    live_assets: int
    asset_universe: list[str]
    validated_observations: int
    engineered_features: int
    feature_names: list[str]
    market_trade_rows: int
    quant_metric_rows: int
    live_signal_rows: int
    final_decision_rows: int
    open_positions: int
    closed_trades: int
    latest_market_data_age_seconds: float | None
    latest_signal_age_seconds: float | None
    inference_batch_size: int
    return_model_batch_latency_ms_median: float | None
    direction_model_batch_latency_ms_median: float | None
    hybrid_batch_latency_ms_median: float | None
    hybrid_batch_latency_ms_p95: float | None
    hybrid_per_row_latency_ms_median: float | None
    hybrid_throughput_rows_per_second: float | None
    pipeline_cycle_seconds: float | None
    pipeline_status: str
    notes: list[str]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def scalar(cursor, query: str) -> Any:
    cursor.execute(query)
    row = cursor.fetchone()
    return row[0] if row else None


def count_python_modules() -> int:
    excluded = {".git", ".venv", "venv", "__pycache__", "tests"}
    return sum(
        1
        for path in PROJECT_ROOT.rglob("*.py")
        if not any(part in excluded for part in path.parts)
    )


def count_workflows() -> int:
    path = PROJECT_ROOT / ".github" / "workflows"
    if not path.exists():
        return 0
    return sum(
        1
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in {".yml", ".yaml"}
    )


def count_pytest_cases() -> tuple[int, str | None]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not integration",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    output = f"{result.stdout}\n{result.stderr}"

    if result.returncode not in {0, 5}:
        return 0, "Unable to collect pytest cases."

    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if "test" in stripped and "collected" in stripped:
            # Handles formats such as "22/24 tests collected (2 deselected)"
            first_token = stripped.split()[0]
            if "/" in first_token:
                selected = first_token.split("/")[0]
                if selected.isdigit():
                    return int(selected), None

        if stripped.endswith("tests collected"):
            first = stripped.split()[0]
            if first.isdigit():
                return int(first), None

    # Reliable fallback: count collected node IDs.
    node_ids = [
        line
        for line in result.stdout.splitlines()
        if "::test_" in line
    ]
    return len(node_ids), None


def age_seconds(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return round(max(0.0, (now_utc() - value).total_seconds()), 2)


def prepare_features(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    prepared = df.copy()
    encoder = bundle["market_regime_encoder"]
    known = set(encoder.classes_)
    prepared["market_regime_safe"] = prepared["market_regime"].apply(
        lambda value: value if value in known else encoder.classes_[0]
    )
    prepared["market_regime_encoded"] = encoder.transform(
        prepared["market_regime_safe"].astype(str)
    )
    return prepared[bundle["features"]]


def benchmark_inference(
    df: pd.DataFrame,
    iterations: int,
) -> dict[str, Any]:
    empty = {
        "engineered_features": 0,
        "feature_names": [],
        "inference_batch_size": len(df),
        "return_model_batch_latency_ms_median": None,
        "direction_model_batch_latency_ms_median": None,
        "hybrid_batch_latency_ms_median": None,
        "hybrid_batch_latency_ms_p95": None,
        "hybrid_per_row_latency_ms_median": None,
        "hybrid_throughput_rows_per_second": None,
    }

    if df.empty or not RETURN_MODEL_PATH.exists() or not DIRECTION_MODEL_PATH.exists():
        return empty

    return_bundle = joblib.load(RETURN_MODEL_PATH)
    direction_bundle = joblib.load(DIRECTION_MODEL_PATH)

    return_x = prepare_features(df, return_bundle)
    direction_x = prepare_features(df, direction_bundle)

    feature_names = sorted(
        set(return_bundle["features"]) | set(direction_bundle["features"])
    )

    # Warm-up.
    for _ in range(10):
        return_bundle["model"].predict(return_x)
        direction_bundle["model"].predict_proba(direction_x)

    return_ms: list[float] = []
    direction_ms: list[float] = []
    hybrid_ms: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        return_bundle["model"].predict(return_x)
        return_ms.append((time.perf_counter_ns() - start) / 1_000_000)

        start = time.perf_counter_ns()
        direction_bundle["model"].predict_proba(direction_x)
        direction_ms.append((time.perf_counter_ns() - start) / 1_000_000)

        start = time.perf_counter_ns()
        return_bundle["model"].predict(return_x)
        direction_bundle["model"].predict_proba(direction_x)
        hybrid_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    hybrid_sorted = sorted(hybrid_ms)
    p95_index = min(
        len(hybrid_sorted) - 1,
        max(0, round(0.95 * len(hybrid_sorted)) - 1),
    )
    median_hybrid = statistics.median(hybrid_ms)
    batch_size = len(df)

    return {
        "engineered_features": len(feature_names),
        "feature_names": feature_names,
        "inference_batch_size": batch_size,
        "return_model_batch_latency_ms_median": round(
            statistics.median(return_ms), 3
        ),
        "direction_model_batch_latency_ms_median": round(
            statistics.median(direction_ms), 3
        ),
        "hybrid_batch_latency_ms_median": round(median_hybrid, 3),
        "hybrid_batch_latency_ms_p95": round(hybrid_sorted[p95_index], 3),
        "hybrid_per_row_latency_ms_median": round(
            median_hybrid / batch_size, 3
        ),
        "hybrid_throughput_rows_per_second": round(
            batch_size / (median_hybrid / 1000), 2
        ),
    }


def benchmark_pipeline() -> tuple[float | None, str, str | None]:
    started = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ml.quantlab_orchestrator", "--once"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None, "FAIL", "Pipeline one-cycle benchmark timed out."

    elapsed = round(time.perf_counter() - started, 2)

    if result.returncode == 0:
        return elapsed, "PASS", None

    error_tail = "\n".join(result.stderr.splitlines()[-5:])
    return None, "FAIL", f"Pipeline benchmark failed: {error_tail}"


def collect_database_metrics(conn) -> tuple[dict[str, Any], pd.DataFrame]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT symbol FROM market_trades ORDER BY symbol;"
        )
        assets = [row[0] for row in cursor.fetchall()]

        values = {
            "live_assets": len(assets),
            "asset_universe": assets,
            "validated_observations": int(
                scalar(
                    cursor,
                    """
                    SELECT COUNT(*) FROM signal_validation
                    WHERE future_return_5m IS NOT NULL;
                    """,
                )
                or 0
            ),
            "market_trade_rows": int(
                scalar(cursor, "SELECT COUNT(*) FROM market_trades;") or 0
            ),
            "quant_metric_rows": int(
                scalar(cursor, "SELECT COUNT(*) FROM quant_metrics;") or 0
            ),
            "live_signal_rows": int(
                scalar(cursor, "SELECT COUNT(*) FROM live_return_signals;") or 0
            ),
            "final_decision_rows": int(
                scalar(cursor, "SELECT COUNT(*) FROM final_strategy_decisions;")
                or 0
            ),
            "open_positions": int(
                scalar(
                    cursor,
                    """
                    SELECT COUNT(*) FROM paper_positions
                    WHERE position_status = 'OPEN';
                    """,
                )
                or 0
            ),
            "closed_trades": int(
                scalar(cursor, "SELECT COUNT(*) FROM closed_paper_trades;") or 0
            ),
            "latest_market_data_age_seconds": age_seconds(
                scalar(cursor, "SELECT MAX(event_time) FROM market_trades;")
            ),
            "latest_signal_age_seconds": age_seconds(
                scalar(cursor, "SELECT MAX(created_at) FROM live_return_signals;")
            ),
        }

        cursor.execute(FEATURE_QUERY)
        rows = cursor.fetchall()

    columns = [
        "symbol",
        "metric_time",
        "rsi",
        "z_score",
        "rolling_volatility",
        "liquidity_pressure",
        "market_regime",
        "momentum_5m",
        "momentum_15m",
        "momentum_30m",
    ]
    return values, pd.DataFrame(rows, columns=columns)


def print_metrics(metrics: Metrics) -> None:
    print()
    print("=" * 70)
    print("QUANTLAB VERIFIED SYSTEM METRICS")
    print("=" * 70)

    for key, value in asdict(metrics).items():
        if key == "notes":
            continue
        label = key.replace("_", " ").title()
        print(f"{label:<42}: {value}")

    if metrics.notes:
        print("-" * 70)
        print("NOTES")
        for note in metrics.notes:
            print(f"- {note}")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--benchmark-pipeline", action="store_true")
    parser.add_argument("--store-db", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notes: list[str] = []

    test_count, test_note = count_pytest_cases()
    if test_note:
        notes.append(test_note)

    with psycopg2.connect(**DB_CONFIG) as conn:
        db_values, feature_df = collect_database_metrics(conn)
        inference_values = benchmark_inference(
            feature_df,
            max(20, args.iterations),
        )

        pipeline_seconds = None
        pipeline_status = "NOT_MEASURED"

        if args.benchmark_pipeline:
            pipeline_seconds, pipeline_status, pipeline_note = benchmark_pipeline()
            if pipeline_note:
                notes.append(pipeline_note)

        market_age = db_values["latest_market_data_age_seconds"]
        signal_age = db_values["latest_signal_age_seconds"]

        if market_age is not None and market_age > 300:
            notes.append("Latest market data is older than five minutes.")
        if signal_age is not None and signal_age > 300:
            notes.append("Latest live signal is older than five minutes.")

        metrics = Metrics(
            generated_at_utc=now_utc().isoformat(),
            pipeline_stages=len(
                __import__(
                    "ml.quantlab_orchestrator",
                    fromlist=["PIPELINE_STEPS"],
                ).PIPELINE_STEPS
            ),
            python_modules=count_python_modules(),
            automated_tests_collected=test_count,
            github_workflows=count_workflows(),
            pipeline_cycle_seconds=pipeline_seconds,
            pipeline_status=pipeline_status,
            notes=notes,
            **db_values,
            **inference_values,
        )

        if args.store_db:
            payload = json.dumps(asdict(metrics))
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_metrics_history (
                        id BIGSERIAL PRIMARY KEY,
                        generated_at TIMESTAMPTZ NOT NULL,
                        metrics_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO system_metrics_history (
                        generated_at,
                        metrics_json
                    )
                    VALUES (%s, %s::jsonb);
                    """,
                    (
                        datetime.fromisoformat(metrics.generated_at_utc),
                        payload,
                    ),
                )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    print_metrics(metrics)
    print(f"JSON snapshot: {args.json_output}")


if __name__ == "__main__":
    main()
