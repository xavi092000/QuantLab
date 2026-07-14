from configs.database import DB_CONFIG
import os
import joblib
import psycopg2
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


MODEL_DIR = "ml"
MODEL_PATH = "ml/signal_success_retrained_model.pkl"

FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "signal_coherence_score",
    "market_regime_encoded",
    "alert_level_encoded",
]


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_pending_job(cursor):
    cursor.execute("""
        SELECT job_id
        FROM retraining_jobs
        WHERE job_status = 'PENDING'
        ORDER BY requested_at ASC
        LIMIT 1;
    """)
    return cursor.fetchone()


def mark_job_running(cursor, job_id):
    cursor.execute("""
        UPDATE retraining_jobs
        SET job_status = 'RUNNING',
            notes = 'Retraining started'
        WHERE job_id = %s;
    """, (job_id,))


def mark_job_completed(cursor, job_id, accuracy):
    cursor.execute("""
        UPDATE retraining_jobs
        SET job_status = 'COMPLETED',
            notes = %s
        WHERE job_id = %s;
    """, (
        f"Retraining completed successfully. New accuracy: {accuracy:.2f}%",
        job_id,
    ))


def mark_job_failed(cursor, job_id, error_message):
    cursor.execute("""
        UPDATE retraining_jobs
        SET job_status = 'FAILED',
            notes = %s
        WHERE job_id = %s;
    """, (
        f"Retraining failed: {error_message}",
        job_id,
    ))


def load_training_data(conn):
    query = """
        SELECT
            qm.rsi,
            qm.z_score,
            qm.rolling_volatility,
            qm.liquidity_pressure,
            qm.signal_coherence_score,
            qm.market_regime,
            COALESCE(qm.alert_level, 'UNKNOWN') AS alert_level,
            sv.signal_success
        FROM signal_validation sv
        JOIN quant_metrics qm
            ON sv.signal_time = qm.metric_time
           AND sv.symbol = qm.symbol
        WHERE qm.rsi IS NOT NULL
          AND qm.z_score IS NOT NULL
          AND qm.rolling_volatility IS NOT NULL
          AND qm.liquidity_pressure IS NOT NULL
          AND qm.signal_coherence_score IS NOT NULL
          AND qm.market_regime IS NOT NULL
          AND sv.signal_success IS NOT NULL
        ORDER BY qm.metric_time ASC;
    """

    return pd.read_sql(query, conn)


def prepare_features(df):
    regime_encoder = LabelEncoder()
    alert_encoder = LabelEncoder()

    df["market_regime_encoded"] = regime_encoder.fit_transform(
        df["market_regime"].astype(str)
    )

    df["alert_level_encoded"] = alert_encoder.fit_transform(
        df["alert_level"].astype(str)
    )

    X = df[FEATURES]
    y = df["signal_success"].astype(int)

    return X, y, regime_encoder, alert_encoder


def save_retraining_metrics(cursor, job_id, accuracy, train_rows, test_rows):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrained_model_metrics (
            id BIGSERIAL PRIMARY KEY,
            job_id INTEGER,
            model_name TEXT,
            accuracy_pct DOUBLE PRECISION,
            train_rows INTEGER,
            test_rows INTEGER,
            model_path TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        INSERT INTO retrained_model_metrics (
            job_id,
            model_name,
            accuracy_pct,
            train_rows,
            test_rows,
            model_path
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (
        job_id,
        "RandomForest_Retrained_Signal_Success",
        accuracy,
        train_rows,
        test_rows,
        MODEL_PATH,
    ))


def main():
    conn = get_connection()
    cursor = conn.cursor()

    job = fetch_pending_job(cursor)

    if not job:
        print("[INFO] No pending retraining job found.")
        cursor.close()
        conn.close()
        return

    job_id = job[0]

    print("==============================")
    print("RETRAINING JOB STARTED")
    print("==============================")
    print(f"Job ID: {job_id}")

    try:
        mark_job_running(cursor, job_id)
        conn.commit()

        print("[INFO] Loading training data...")
        df = load_training_data(conn)

        print(f"[INFO] Rows loaded: {len(df)}")

        if len(df) < 1000:
            raise ValueError("Not enough training data for retraining.")

        X, y, regime_encoder, alert_encoder = prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        )

        print("[INFO] Training new model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "regime_encoder": regime_encoder,
                "alert_encoder": alert_encoder,
            },
            MODEL_PATH,
        )

        save_retraining_metrics(
            cursor=cursor,
            job_id=job_id,
            accuracy=accuracy,
            train_rows=len(X_train),
            test_rows=len(X_test),
        )

        mark_job_completed(cursor, job_id, accuracy)
        conn.commit()

        print("==============================")
        print("RETRAINING COMPLETED")
        print("==============================")
        print(f"New Accuracy: {accuracy:.2f}%")
        print(f"Saved Model : {MODEL_PATH}")

    except Exception as error:
        conn.rollback()
        mark_job_failed(cursor, job_id, str(error))
        conn.commit()

        print("==============================")
        print("RETRAINING FAILED")
        print("==============================")
        print(error)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()


