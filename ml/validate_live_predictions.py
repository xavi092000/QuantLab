from configs.database import DB_CONFIG
import time
import psycopg2


POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_unvalidated_predictions(cursor):
    cursor.execute(
        """
        SELECT
            p.id,
            p.prediction_time,
            p.predicted_regime,
            p.confidence
        FROM live_regime_predictions p
        LEFT JOIN ml_prediction_validation v
            ON p.id = v.prediction_id
        WHERE v.id IS NULL
          AND p.prediction_time <= NOW() - INTERVAL '5 minutes'
        ORDER BY p.prediction_time ASC
        LIMIT 50;
        """
    )
    return cursor.fetchall()


def fetch_actual_regime_5m(cursor, prediction_time):
    cursor.execute(
        """
        SELECT market_regime
        FROM quant_metrics
        WHERE metric_time >= %s + INTERVAL '5 minutes'
        ORDER BY metric_time ASC
        LIMIT 1;
        """,
        (prediction_time,),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return row[0]


def insert_validation(
    cursor,
    prediction_id,
    prediction_time,
    predicted_regime,
    actual_regime_5m,
    prediction_success,
    confidence,
):
    cursor.execute(
        """
        INSERT INTO ml_prediction_validation (
            prediction_id,
            prediction_time,
            predicted_regime,
            actual_regime_5m,
            prediction_success,
            confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            prediction_id,
            prediction_time,
            predicted_regime,
            actual_regime_5m,
            prediction_success,
            confidence,
        ),
    )


def main():
    conn = get_connection()
    cursor = conn.cursor()

    print("[INFO] ML Prediction Validation Engine started")

    while True:
        try:
            predictions = fetch_unvalidated_predictions(cursor)

            if not predictions:
                print("[INFO] No mature predictions to validate yet")
                time.sleep(POLL_SECONDS)
                continue

            for prediction in predictions:
                prediction_id, prediction_time, predicted_regime, confidence = prediction

                actual_regime_5m = fetch_actual_regime_5m(cursor, prediction_time)

                if actual_regime_5m is None:
                    continue

                prediction_success = predicted_regime == actual_regime_5m

                insert_validation(
                    cursor,
                    prediction_id,
                    prediction_time,
                    predicted_regime,
                    actual_regime_5m,
                    prediction_success,
                    confidence,
                )

                print(
                    f"[ML VALIDATION] "
                    f"time={prediction_time} "
                    f"predicted={predicted_regime} "
                    f"actual_5m={actual_regime_5m} "
                    f"confidence={float(confidence):.2%} "
                    f"success={prediction_success}"
                )

            conn.commit()

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


