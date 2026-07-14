from configs.database import DB_CONFIG
import os

import joblib
import pandas as pd
import psycopg2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "signal_coherence_score",
]

TARGET = "future_market_regime_5m"

MODEL_DIR = "ml"
BEST_MODEL_PATH = "ml/future_regime_predictor_model.pkl"


def load_future_training_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            qm.metric_time,
            qm.rsi,
            qm.z_score,
            qm.rolling_volatility,
            qm.liquidity_pressure,
            qm.signal_coherence_score,
            future_qm.market_regime AS future_market_regime_5m
        FROM quant_metrics qm
        JOIN LATERAL (
            SELECT market_regime
            FROM quant_metrics qm_future
            WHERE qm_future.metric_time >= qm.metric_time + INTERVAL '5 minutes'
            ORDER BY qm_future.metric_time ASC
            LIMIT 1
        ) future_qm ON TRUE
        WHERE qm.rsi IS NOT NULL
          AND qm.z_score IS NOT NULL
          AND qm.rolling_volatility IS NOT NULL
          AND qm.liquidity_pressure IS NOT NULL
          AND qm.signal_coherence_score IS NOT NULL
          AND qm.market_regime IS NOT NULL;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest_Future_Regime": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting_Future_Regime": GradientBoostingClassifier(
            random_state=42,
        ),
        "LogisticRegression_Future_Regime": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    }

    results = []

    for model_name, model in models.items():
        print("\n==============================")
        print(f"Training: {model_name}")
        print("==============================")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        results.append(
            {
                "model_name": model_name,
                "model": model,
                "accuracy": accuracy,
            }
        )

    best_result = max(results, key=lambda item: item["accuracy"])

    return results, best_result


def save_model_metrics(results, best_result):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS future_model_metrics (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT,
            accuracy DOUBLE PRECISION,
            is_best_model BOOLEAN,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    for result in results:
        cursor.execute(
            """
            INSERT INTO future_model_metrics (
                model_name,
                accuracy,
                is_best_model
            )
            VALUES (%s, %s, %s);
            """,
            (
                result["model_name"],
                float(result["accuracy"]),
                result["model_name"] == best_result["model_name"],
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()


def main():
    print("[INFO] Loading future regime training data...")

    df = load_future_training_data()

    print(f"[INFO] Rows loaded: {len(df)}")
    print("\n[INFO] Future regime distribution:")
    print(df[TARGET].value_counts())

    if len(df) < 500:
        print("[ERROR] Not enough data to train a future regime model.")
        return

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results, best_result = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_result["model"], BEST_MODEL_PATH)

    save_model_metrics(results, best_result)

    print("\n==============================")
    print("BEST FUTURE REGIME MODEL")
    print("==============================")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()


