from configs.database import DB_CONFIG
import time
import joblib
import psycopg2
import pandas as pd


POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_model():
    return joblib.load("ml/regime_predictor_model.pkl")


def fetch_latest_metrics(conn):
    query = """
    SELECT
        metric_time,
        rsi,
        z_score,
        rolling_volatility,
        liquidity_pressure,
        signal_coherence_score
    FROM quant_metrics
    ORDER BY metric_time DESC
    LIMIT 1;
    """

    return pd.read_sql(query, conn)


def save_prediction(
    cursor,
    prediction_time,
    predicted_regime,
    confidence,
    rsi,
    z_score,
    rolling_volatility,
    liquidity_pressure,
    signal_coherence_score,
):
    cursor.execute(
        """
        INSERT INTO live_regime_predictions (
            prediction_time,
            predicted_regime,
            confidence,
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            signal_coherence_score
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
        """,
        (
            prediction_time,
            predicted_regime,
            confidence,
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            signal_coherence_score,
        ),
    )


def main():

    print("[INFO] Loading model...")

    model = load_model()

    print("[INFO] Model loaded.")

    conn = get_connection()
    cursor = conn.cursor()

    while True:

        try:

            df = fetch_latest_metrics(conn)

            if len(df) == 0:
                time.sleep(POLL_SECONDS)
                continue

            features = df[
                [
                    "rsi",
                    "z_score",
                    "rolling_volatility",
                    "liquidity_pressure",
                    "signal_coherence_score",
                ]
            ]

            prediction = model.predict(features)[0]

            confidence = max(model.predict_proba(features)[0])

            row = df.iloc[0]

            save_prediction(
                cursor,
                row["metric_time"],
                prediction,
                float(confidence),
                float(row["rsi"]),
                float(row["z_score"]),
                float(row["rolling_volatility"]),
                float(row["liquidity_pressure"]),
                float(row["signal_coherence_score"]),
            )

            conn.commit()

            print(
                f"[PREDICTION] "
                f"regime={prediction} "
                f"confidence={confidence:.2%}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


