from configs.database import DB_CONFIG
import time
import joblib
import psycopg2
import pandas as pd


MODEL_PATH = "ml/signal_success_predictor_model.pkl"
POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_model_bundle():
    return joblib.load(MODEL_PATH)


def fetch_latest_signal(conn):
    query = """
        SELECT
            metric_time,
            symbol,
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            signal_coherence_score,
            market_regime,
            COALESCE(alert_level, 'UNKNOWN') AS alert_level
        FROM quant_metrics
        WHERE rsi IS NOT NULL
          AND z_score IS NOT NULL
          AND rolling_volatility IS NOT NULL
          AND liquidity_pressure IS NOT NULL
          AND signal_coherence_score IS NOT NULL
          AND market_regime IS NOT NULL
        ORDER BY metric_time DESC
        LIMIT 1;
    """

    return pd.read_sql(query, conn)


def encode_value(encoder, value):
    value = str(value)

    if value in encoder.classes_:
        return int(encoder.transform([value])[0])

    return 0


def create_prediction_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS live_signal_success_predictions (
            id BIGSERIAL PRIMARY KEY,
            prediction_time TIMESTAMPTZ,
            symbol TEXT,
            market_regime TEXT,
            alert_level TEXT,
            success_probability DOUBLE PRECISION,
            predicted_success BOOLEAN,
            recommendation TEXT,
            rsi DOUBLE PRECISION,
            z_score DOUBLE PRECISION,
            rolling_volatility DOUBLE PRECISION,
            liquidity_pressure DOUBLE PRECISION,
            signal_coherence_score DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )


def save_prediction(cursor, row, success_probability, predicted_success, recommendation):
    cursor.execute(
        """
        INSERT INTO live_signal_success_predictions (
            prediction_time,
            symbol,
            market_regime,
            alert_level,
            success_probability,
            predicted_success,
            recommendation,
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            signal_coherence_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            row["metric_time"],
            row["symbol"],
            row["market_regime"],
            row["alert_level"],
            float(success_probability),
            bool(predicted_success),
            recommendation,
            float(row["rsi"]),
            float(row["z_score"]),
            float(row["rolling_volatility"]),
            float(row["liquidity_pressure"]),
            float(row["signal_coherence_score"]),
        ),
    )


def get_recommendation(success_probability):
    if success_probability >= 0.75:
        return "TAKE"
    if success_probability >= 0.60:
        return "WATCH"
    return "AVOID"


def main():
    print("[INFO] Loading Signal Success Predictor...")

    bundle = load_model_bundle()

    model = bundle["model"]
    features = bundle["features"]
    regime_encoder = bundle["regime_encoder"]
    alert_encoder = bundle["alert_encoder"]

    print("[INFO] Model loaded.")

    conn = get_connection()
    cursor = conn.cursor()

    create_prediction_table(cursor)
    conn.commit()

    while True:
        try:
            df = fetch_latest_signal(conn)

            if len(df) == 0:
                print("[INFO] No signal found yet")
                time.sleep(POLL_SECONDS)
                continue

            row = df.iloc[0].copy()

            row["market_regime_encoded"] = encode_value(
                regime_encoder,
                row["market_regime"],
            )

            row["alert_level_encoded"] = encode_value(
                alert_encoder,
                row["alert_level"],
            )

            X = pd.DataFrame([row[features]])

            success_probability = model.predict_proba(X)[0][1]
            predicted_success = success_probability >= 0.50
            recommendation = get_recommendation(success_probability)

            save_prediction(
                cursor,
                row,
                success_probability,
                predicted_success,
                recommendation,
            )

            conn.commit()

            print(
                f"[SIGNAL SUCCESS] "
                f"time={row['metric_time']} "
                f"regime={row['market_regime']} "
                f"alert={row['alert_level']} "
                f"success_probability={success_probability:.2%} "
                f"recommendation={recommendation}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


