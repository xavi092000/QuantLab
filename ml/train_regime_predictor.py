from configs.database import DB_CONFIG
import os

import joblib
import pandas as pd
import psycopg2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "signal_coherence_score",
]

TARGET = "market_regime"
MODEL_PATH = "ml/regime_predictor_model.pkl"


def load_training_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            signal_coherence_score,
            market_regime
        FROM quant_metrics
        WHERE rsi IS NOT NULL
          AND z_score IS NOT NULL
          AND rolling_volatility IS NOT NULL
          AND liquidity_pressure IS NOT NULL
          AND signal_coherence_score IS NOT NULL
          AND market_regime IS NOT NULL;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def main():
    print("[INFO] Loading QuantLab training data...")

    df = load_training_data()

    print(f"[INFO] Rows loaded: {len(df)}")
    print("[INFO] Regime distribution:")
    print(df[TARGET].value_counts())

    if len(df) < 100:
        print("[ERROR] Not enough data to train a reliable model.")
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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    print("[INFO] Training Random Forest regime predictor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n[INFO] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


