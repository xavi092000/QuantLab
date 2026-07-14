from configs.database import DB_CONFIG
import os

import joblib
import pandas as pd
import psycopg2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "signal_coherence_score",
    "market_regime_encoded",
    "alert_level_encoded",
    "trend_regime_encoded",
    "ema20",
    "ema50",
    "ema200",
]

TARGET = "signal_success"
MODEL_PATH = "ml/signal_success_predictor_v2_fast_model.pkl"


def load_training_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        WITH latest_trend AS (
            SELECT
                trend_regime,
                ema20,
                ema50,
                ema200
            FROM trend_classification
            ORDER BY event_time DESC
            LIMIT 1
        )
        SELECT
            qm.rsi,
            qm.z_score,
            qm.rolling_volatility,
            qm.liquidity_pressure,
            qm.signal_coherence_score,
            qm.market_regime,
            COALESCE(qm.alert_level, 'UNKNOWN') AS alert_level,
            COALESCE(lt.trend_regime, 'UNKNOWN') AS trend_regime,
            COALESCE(lt.ema20, qm.price) AS ema20,
            COALESCE(lt.ema50, qm.price) AS ema50,
            COALESCE(lt.ema200, qm.price) AS ema200,
            sv.signal_success
        FROM signal_validation sv
        JOIN quant_metrics qm
            ON sv.signal_time = qm.metric_time
            AND sv.symbol = qm.symbol
        CROSS JOIN latest_trend lt
        WHERE qm.rsi IS NOT NULL
          AND qm.z_score IS NOT NULL
          AND qm.rolling_volatility IS NOT NULL
          AND qm.liquidity_pressure IS NOT NULL
          AND qm.signal_coherence_score IS NOT NULL
          AND qm.market_regime IS NOT NULL
          AND sv.signal_success IS NOT NULL;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def prepare_features(df):
    regime_encoder = LabelEncoder()
    alert_encoder = LabelEncoder()
    trend_encoder = LabelEncoder()

    df["market_regime_encoded"] = regime_encoder.fit_transform(df["market_regime"].astype(str))
    df["alert_level_encoded"] = alert_encoder.fit_transform(df["alert_level"].astype(str))
    df["trend_regime_encoded"] = trend_encoder.fit_transform(df["trend_regime"].astype(str))

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    return X, y, regime_encoder, alert_encoder, trend_encoder


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest_Signal_Success_V2_FAST": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting_Signal_Success_V2_FAST": GradientBoostingClassifier(
            random_state=42,
        ),
        "LogisticRegression_Signal_Success_V2_FAST": LogisticRegression(
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

    return results, max(results, key=lambda item: item["accuracy"])


def main():
    print("[INFO] Loading FAST V2 training data...")

    df = load_training_data()

    print(f"[INFO] Rows loaded: {len(df)}")
    print("\n[INFO] Target distribution:")
    print(df[TARGET].value_counts())

    print("\n[INFO] Trend distribution:")
    print(df["trend_regime"].value_counts())

    if len(df) < 500:
        print("[ERROR] Not enough data.")
        return

    X, y, regime_encoder, alert_encoder, trend_encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results, best_result = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    os.makedirs("ml", exist_ok=True)

    joblib.dump(
        {
            "model": best_result["model"],
            "features": FEATURES,
            "regime_encoder": regime_encoder,
            "alert_encoder": alert_encoder,
            "trend_encoder": trend_encoder,
        },
        MODEL_PATH,
    )

    print("\n==============================")
    print("BEST SIGNAL SUCCESS MODEL V2 FAST")
    print("==============================")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


