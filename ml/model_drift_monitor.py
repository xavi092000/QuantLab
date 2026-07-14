
from typing import Any

import psycopg2
from psycopg2 import sql

from configs.database import DB_CONFIG


FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
]

LIVE_WINDOW_MINUTES = 60
BASELINE_WINDOW_DAYS = 30
MIN_BASELINE_ROWS = 100
MIN_LIVE_ROWS = 5

RELATIVE_WARNING_THRESHOLD = 0.25
RELATIVE_CRITICAL_THRESHOLD = 0.50

ZSCORE_WARNING_ABS_DIFF = 1.5
ZSCORE_CRITICAL_ABS_DIFF = 3.0

NEAR_ZERO_EPSILON = 1e-6


def classify_drift(
    feature_name: str,
    absolute_difference: float,
    relative_difference: float | None,
) -> str:
    """Classify feature drift using feature-appropriate thresholds."""
    if feature_name == "z_score":
        if absolute_difference >= ZSCORE_CRITICAL_ABS_DIFF:
            return "CRITICAL_DRIFT"
        if absolute_difference >= ZSCORE_WARNING_ABS_DIFF:
            return "WARNING_DRIFT"
        return "NO_DRIFT"

    if relative_difference is None:
        return "INSUFFICIENT_BASELINE"

    if relative_difference >= RELATIVE_CRITICAL_THRESHOLD:
        return "CRITICAL_DRIFT"

    if relative_difference >= RELATIVE_WARNING_THRESHOLD:
        return "WARNING_DRIFT"

    return "NO_DRIFT"


def fetch_feature_statistics(
    cursor: Any,
    feature_name: str,
) -> dict[str, Any]:
    """Fetch separated baseline and live statistics for one feature."""
    query = sql.SQL(
        """
        SELECT
            AVG({feature}) FILTER (
                WHERE metric_time >= NOW() - INTERVAL '{baseline_days} days'
                  AND metric_time < NOW() - INTERVAL '{live_minutes} minutes'
            ) AS historical_avg,

            COUNT({feature}) FILTER (
                WHERE metric_time >= NOW() - INTERVAL '{baseline_days} days'
                  AND metric_time < NOW() - INTERVAL '{live_minutes} minutes'
            ) AS historical_count,

            AVG({feature}) FILTER (
                WHERE metric_time >= NOW() - INTERVAL '{live_minutes} minutes'
            ) AS live_avg,

            COUNT({feature}) FILTER (
                WHERE metric_time >= NOW() - INTERVAL '{live_minutes} minutes'
            ) AS live_count

        FROM quant_metrics
        WHERE {feature} IS NOT NULL;
        """
    ).format(
        feature=sql.Identifier(feature_name),
        baseline_days=sql.Literal(BASELINE_WINDOW_DAYS),
        live_minutes=sql.Literal(LIVE_WINDOW_MINUTES),
    )

    cursor.execute(query)

    (
        historical_avg,
        historical_count,
        live_avg,
        live_count,
    ) = cursor.fetchone()

    return {
        "historical_avg": (
            float(historical_avg)
            if historical_avg is not None
            else None
        ),
        "historical_count": int(historical_count or 0),
        "live_avg": (
            float(live_avg)
            if live_avg is not None
            else None
        ),
        "live_count": int(live_count or 0),
    }


def evaluate_feature_drift(
    feature_name: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate drift only when sufficient baseline and live data exist."""
    historical_avg = statistics["historical_avg"]
    historical_count = statistics["historical_count"]
    live_avg = statistics["live_avg"]
    live_count = statistics["live_count"]

    if historical_count < MIN_BASELINE_ROWS:
        return {
            "feature_name": feature_name,
            "historical_avg": historical_avg,
            "live_avg": live_avg,
            "absolute_difference": None,
            "relative_difference": None,
            "drift_status": "INSUFFICIENT_BASELINE",
            "drift_method": "NOT_EVALUATED",
            "historical_count": historical_count,
            "live_count": live_count,
        }

    if live_count < MIN_LIVE_ROWS:
        return {
            "feature_name": feature_name,
            "historical_avg": historical_avg,
            "live_avg": live_avg,
            "absolute_difference": None,
            "relative_difference": None,
            "drift_status": "INSUFFICIENT_LIVE_DATA",
            "drift_method": "NOT_EVALUATED",
            "historical_count": historical_count,
            "live_count": live_count,
        }

    if historical_avg is None or live_avg is None:
        return {
            "feature_name": feature_name,
            "historical_avg": historical_avg,
            "live_avg": live_avg,
            "absolute_difference": None,
            "relative_difference": None,
            "drift_status": "MISSING_STATISTICS",
            "drift_method": "NOT_EVALUATED",
            "historical_count": historical_count,
            "live_count": live_count,
        }

    absolute_difference = abs(live_avg - historical_avg)

    if feature_name == "z_score":
        relative_difference = None
        drift_method = "ABSOLUTE_MEAN_SHIFT"
    elif abs(historical_avg) < NEAR_ZERO_EPSILON:
        relative_difference = None
        drift_method = "RELATIVE_DIFF_UNAVAILABLE"
    else:
        relative_difference = (
            absolute_difference / abs(historical_avg)
        )
        drift_method = "RELATIVE_MEAN_SHIFT"

    drift_status = classify_drift(
        feature_name=feature_name,
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
    )

    return {
        "feature_name": feature_name,
        "historical_avg": historical_avg,
        "live_avg": live_avg,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "drift_status": drift_status,
        "drift_method": drift_method,
        "historical_count": historical_count,
        "live_count": live_count,
    }


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_drift_monitor (
                    id BIGSERIAL PRIMARY KEY,
                    feature_name TEXT NOT NULL,
                    historical_avg DOUBLE PRECISION,
                    live_avg DOUBLE PRECISION,
                    absolute_difference DOUBLE PRECISION,
                    relative_difference DOUBLE PRECISION,
                    drift_status TEXT NOT NULL,
                    drift_method TEXT NOT NULL,
                    historical_count INTEGER,
                    live_count INTEGER,
                    baseline_window_days INTEGER,
                    live_window_minutes INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE model_drift_monitor
                ADD COLUMN IF NOT EXISTS historical_count INTEGER,
                ADD COLUMN IF NOT EXISTS live_count INTEGER,
                ADD COLUMN IF NOT EXISTS baseline_window_days INTEGER,
                ADD COLUMN IF NOT EXISTS live_window_minutes INTEGER;
                """
            )

            results = []

            for feature_name in FEATURES:
                statistics = fetch_feature_statistics(
                    cursor=cursor,
                    feature_name=feature_name,
                )

                result = evaluate_feature_drift(
                    feature_name=feature_name,
                    statistics=statistics,
                )

                results.append(result)

            cursor.executemany(
                """
                INSERT INTO model_drift_monitor (
                    feature_name,
                    historical_avg,
                    live_avg,
                    absolute_difference,
                    relative_difference,
                    drift_status,
                    drift_method,
                    historical_count,
                    live_count,
                    baseline_window_days,
                    live_window_minutes
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                );
                """,
                [
                    (
                        result["feature_name"],
                        result["historical_avg"],
                        result["live_avg"],
                        result["absolute_difference"],
                        result["relative_difference"],
                        result["drift_status"],
                        result["drift_method"],
                        result["historical_count"],
                        result["live_count"],
                        BASELINE_WINDOW_DAYS,
                        LIVE_WINDOW_MINUTES,
                    )
                    for result in results
                ],
            )

            critical_count = sum(
                result["drift_status"] == "CRITICAL_DRIFT"
                for result in results
            )

            warning_count = sum(
                result["drift_status"] == "WARNING_DRIFT"
                for result in results
            )

            print("==============================")
            print("MODEL DRIFT MONITOR V3")
            print("==============================")
            print(
                "Baseline window:",
                f"{BASELINE_WINDOW_DAYS} days",
            )
            print(
                "Live window:",
                f"{LIVE_WINDOW_MINUTES} minutes",
            )
            print("Critical features:", critical_count)
            print("Warning features:", warning_count)

            for result in results:
                historical_avg = result["historical_avg"]
                live_avg = result["live_avg"]
                absolute_difference = result["absolute_difference"]
                relative_difference = result["relative_difference"]

                historical_text = (
                    "N/A"
                    if historical_avg is None
                    else f"{historical_avg:.8f}"
                )

                live_text = (
                    "N/A"
                    if live_avg is None
                    else f"{live_avg:.8f}"
                )

                absolute_text = (
                    "N/A"
                    if absolute_difference is None
                    else f"{absolute_difference:.8f}"
                )

                relative_text = (
                    "N/A"
                    if relative_difference is None
                    else f"{relative_difference * 100:.2f}%"
                )

                print(
                    f"{result['feature_name']} | "
                    f"historical={historical_text} | "
                    f"live={live_text} | "
                    f"abs_diff={absolute_text} | "
                    f"rel_diff={relative_text} | "
                    f"baseline_n={result['historical_count']} | "
                    f"live_n={result['live_count']} | "
                    f"status={result['drift_status']} | "
                    f"method={result['drift_method']}"
                )


if __name__ == "__main__":
    main()
