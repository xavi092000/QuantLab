from __future__ import annotations

import joblib
import pandas as pd
import psycopg2

from configs.database import DB_CONFIG


RETURN_MODEL_PATH = "ml/return_prediction_model.pkl"
DIRECTION_MODEL_PATH = "ml/return_direction_model.pkl"

MIN_PROBABILITY_UP = 0.60
MAX_NEGATIVE_RETURN_FOR_SUPPORT = -0.0001


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
        bar_time,
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
    COALESCE(m.momentum_5m, 0) AS momentum_5m,
    COALESCE(m.momentum_15m, 0) AS momentum_15m,
    COALESCE(m.momentum_30m, 0) AS momentum_30m
FROM latest_quant q
LEFT JOIN latest_momentum m
    ON q.symbol = m.symbol;
"""


def load_scalar(conn, query: str, column: str):
    df = pd.read_sql_query(query, conn)
    if df.empty:
        raise RuntimeError(f"No configuration row found for {column}.")
    return df[column].iloc[0]


def prepare_features(df, bundle):
    encoder = bundle["market_regime_encoder"]
    features = bundle["features"]
    known_classes = set(encoder.classes_)

    prepared = df.copy()
    prepared["market_regime_safe"] = prepared["market_regime"].apply(
        lambda value: value if value in known_classes else encoder.classes_[0]
    )
    prepared["market_regime_encoded"] = encoder.transform(
        prepared["market_regime_safe"].astype(str)
    )
    return prepared[features]


def hybrid_research_signal(
    predicted_return: float,
    probability_up: float,
    return_threshold: float,
) -> tuple[str, str]:
    if (
        probability_up >= MIN_PROBABILITY_UP
        and predicted_return >= return_threshold
    ):
        return (
            "BUY_CANDIDATE",
            "Direction and return models both support upside",
        )

    if (
        probability_up >= MIN_PROBABILITY_UP
        and predicted_return >= MAX_NEGATIVE_RETURN_FOR_SUPPORT
    ):
        return (
            "WATCH",
            "Direction model supports upside but return estimate is weak",
        )

    if predicted_return > return_threshold:
        return (
            "WATCH",
            "Return model is positive but direction confidence is insufficient",
        )

    return (
        "AVOID",
        "Hybrid model confirmation not met",
    )


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        return_threshold = float(
            load_scalar(
                conn,
                """
                SELECT return_threshold
                FROM production_strategy_config
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                "return_threshold",
            )
        )

        risk_decision = str(
            load_scalar(
                conn,
                """
                SELECT risk_decision
                FROM risk_budgeting_v2
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                "risk_decision",
            )
        )

        df = pd.read_sql_query(FEATURE_QUERY, conn)

        if df.empty:
            print("[ERROR] No live feature rows found.")
            return

        return_bundle = joblib.load(RETURN_MODEL_PATH)
        direction_bundle = joblib.load(DIRECTION_MODEL_PATH)

        return_X = prepare_features(df, return_bundle)
        direction_X = prepare_features(df, direction_bundle)

        df["predicted_return_5m"] = return_bundle["model"].predict(return_X)
        df["probability_up"] = direction_bundle["model"].predict_proba(
            direction_X
        )[:, 1]

        signals = df.apply(
            lambda row: hybrid_research_signal(
                predicted_return=float(row["predicted_return_5m"]),
                probability_up=float(row["probability_up"]),
                return_threshold=return_threshold,
            ),
            axis=1,
        )

        df["research_signal"] = [value[0] for value in signals]
        df["decision_reason"] = [value[1] for value in signals]

        def final_decision(row):
            if risk_decision == "NO_RISK_ALLOCATED":
                return "NO_TRADE"
            if row["research_signal"] == "BUY_CANDIDATE":
                return "BUY"
            if row["research_signal"] == "WATCH":
                return "WATCH"
            return "NO_TRADE"

        df["final_decision"] = df.apply(final_decision, axis=1)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS live_return_signals (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT,
                    metric_time TIMESTAMPTZ,
                    market_regime TEXT,
                    predicted_return_5m DOUBLE PRECISION,
                    probability_up DOUBLE PRECISION,
                    return_threshold DOUBLE PRECISION,
                    probability_threshold DOUBLE PRECISION,
                    research_signal TEXT,
                    risk_decision TEXT,
                    final_decision TEXT,
                    decision_reason TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE live_return_signals
                ADD COLUMN IF NOT EXISTS probability_up DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS probability_threshold
                    DOUBLE PRECISION;
                """
            )

            cursor.execute("DELETE FROM live_return_signals;")

            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO live_return_signals (
                        symbol,
                        metric_time,
                        market_regime,
                        predicted_return_5m,
                        probability_up,
                        return_threshold,
                        probability_threshold,
                        research_signal,
                        risk_decision,
                        final_decision,
                        decision_reason
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
                    """,
                    (
                        row["symbol"],
                        row["metric_time"],
                        row["market_regime"],
                        float(row["predicted_return_5m"]),
                        float(row["probability_up"]),
                        return_threshold,
                        MIN_PROBABILITY_UP,
                        row["research_signal"],
                        risk_decision,
                        row["final_decision"],
                        row["decision_reason"],
                    ),
                )

        print("==============================")
        print("HYBRID LIVE RETURN SIGNAL ENGINE")
        print("==============================")
        print("Rows processed       :", len(df))
        print("Return threshold     :", return_threshold)
        print("Probability threshold:", MIN_PROBABILITY_UP)
        print("Risk decision        :", risk_decision)
        print(
            df[
                [
                    "symbol",
                    "market_regime",
                    "predicted_return_5m",
                    "probability_up",
                    "research_signal",
                    "final_decision",
                ]
            ]
        )


if __name__ == "__main__":
    main()
