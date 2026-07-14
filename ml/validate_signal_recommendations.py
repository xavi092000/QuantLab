from configs.database import DB_CONFIG
import time
import psycopg2

POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signal_recommendation_validation (
        id BIGSERIAL PRIMARY KEY,
        prediction_id BIGINT,
        prediction_time TIMESTAMPTZ,
        recommendation TEXT,
        success_probability DOUBLE PRECISION,
        actual_signal_success BOOLEAN,
        validation_success BOOLEAN,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)


def fetch_unvalidated_predictions(cursor):
    cursor.execute("""
    SELECT
        p.id,
        p.prediction_time,
        p.recommendation,
        p.success_probability
    FROM live_signal_success_predictions p
    LEFT JOIN signal_recommendation_validation v
        ON p.id = v.prediction_id
    WHERE v.id IS NULL
      AND p.prediction_time <= NOW() - INTERVAL '5 minutes'
    ORDER BY p.prediction_time ASC
    LIMIT 50;
    """)

    return cursor.fetchall()


def fetch_actual_result(cursor, prediction_time):
    cursor.execute("""
    SELECT signal_success
    FROM signal_validation
    WHERE signal_time >= %s + INTERVAL '5 minutes'
      AND signal_time <= %s + INTERVAL '6 minutes'
    ORDER BY signal_time ASC
    LIMIT 1;
    """, (prediction_time, prediction_time))

    row = cursor.fetchone()

    if row is None:
        return None

    return bool(row[0])


def validate_recommendation(recommendation, actual_success):
    if recommendation == "TAKE":
        return actual_success

    if recommendation == "AVOID":
        return not actual_success

    if recommendation == "WATCH":
        return True

    return False


def insert_validation(
    cursor,
    prediction_id,
    prediction_time,
    recommendation,
    probability,
    actual_success,
    validation_success
):
    cursor.execute("""
    INSERT INTO signal_recommendation_validation (
        prediction_id,
        prediction_time,
        recommendation,
        success_probability,
        actual_signal_success,
        validation_success
    )
    VALUES (%s,%s,%s,%s,%s,%s);
    """,
    (
        prediction_id,
        prediction_time,
        recommendation,
        probability,
        actual_success,
        validation_success,
    ))


def main():
    conn = get_connection()
    cursor = conn.cursor()

    create_table(cursor)
    conn.commit()

    print("[INFO] Recommendation Validation Engine started")

    while True:
        try:
            predictions = fetch_unvalidated_predictions(cursor)

            if not predictions:
                print("[INFO] No mature recommendations to validate yet")
                time.sleep(POLL_SECONDS)
                continue

            for pred in predictions:
                (
                    prediction_id,
                    prediction_time,
                    recommendation,
                    probability,
                ) = pred

                actual_success = fetch_actual_result(
                    cursor,
                    prediction_time,
                )

                if actual_success is None:
                    print(
                        f"[INFO] No validation result yet "
                        f"for prediction_time={prediction_time}"
                    )
                    continue

                validation_success = validate_recommendation(
                    recommendation,
                    actual_success,
                )

                insert_validation(
                    cursor,
                    prediction_id,
                    prediction_time,
                    recommendation,
                    probability,
                    actual_success,
                    validation_success,
                )

                print(
                    f"[VALIDATED] "
                    f"time={prediction_time} "
                    f"recommendation={recommendation} "
                    f"probability={float(probability):.2%} "
                    f"actual={actual_success} "
                    f"success={validation_success}"
                )

            conn.commit()

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


