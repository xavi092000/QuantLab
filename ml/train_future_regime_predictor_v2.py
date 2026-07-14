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


BASE_FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "signal_coherence_score",
]

TEMPORAL_FEATURES = [
    "rsi_change",
    "zscore_change",
    "volatility_change",
    "liquidity_change",
    "previous_regime_encoded",
]

FEATURES = BASE_FEATURES + TEMPORAL_FEATURES

TARGET = "future_market_regime_5m"

MODEL_DIR = "ml"
BEST_MODEL_PATH = "ml/future_regime_predictor_v2_model.pkl"


def load_raw_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            qm.metric_time,
            qm.rsi,
            qm.z_score,
            qm.rolling_volatility,
            qm.liquidity_pressure,
            qm.signal_coherence_score,
            qm.market_regime,
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
          AND qm.market_regime IS NOT NULL
        ORDER BY qm.metric_time ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def add_temporal_features(df):
    df = df.sort_values("metric_time").copy()

    df["rsi_change"] = df["rsi"].diff()
    df["zscore_change"] = df["z_score"].diff()
    df["volatility_change"] = df["rolling_volatility"].diff()
    df["liquidity_change"] = df["liquidity_pressure"].diff()

    df["previous_regime"] = df["market_regime"].shift(1)

    label_encoder = LabelEncoder()
    df["previous_regime_encoded"] = label_encoder.fit_transform(
        df["previous_regime"].astype(str)
    )

    df = df.dropna(subset=FEATURES + [TARGET])

    return df, label_encoder


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest_Future_Regime_V2": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting_Future_Regime_V2": GradientBoostingClassifier(
            random_state=42,
        ),
        "LogisticRegression_Future_Regime_V2": LogisticRegression(
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
        CREATE TABLE IF NOT EXISTS future_model_metrics_v2 (
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
            INSERT INTO future_model_metrics_v2 (
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


def save_feature_importance(best_result):
    model = best_result["model"]

    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature_name": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS future_model_feature_importance_v2 (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT,
            feature_name TEXT,
            importance DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    for _, row in importance_df.iterrows():
        cursor.execute(
            """
            INSERT INTO future_model_feature_importance_v2 (
                model_name,
                feature_name,
                importance
            )
            VALUES (%s, %s, %s);
            """,
            (
                best_result["model_name"],
                row["feature_name"],
                float(row["importance"]),
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()

    print("\nFeature Importance:")
    print(importance_df)


def main():
    print("[INFO] Loading raw future regime data...")

    df = load_raw_data()

    print(f"[INFO] Raw rows loaded: {len(df)}")

    df, label_encoder = add_temporal_features(df)

    print(f"[INFO] Rows after temporal feature engineering: {len(df)}")

    print("\n[INFO] Future regime distribution:")
    print(df[TARGET].value_counts())

    if len(df) < 500:
        print("[ERROR] Not enough data to train a future regime model V2.")
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

    joblib.dump(
        {
            "model": best_result["model"],
            "label_encoder": label_encoder,
            "features": FEATURES,
        },
        BEST_MODEL_PATH,
    )

    save_model_metrics(results, best_result)
    save_feature_importance(best_result)

    print("\n==============================")
    print("BEST FUTURE REGIME MODEL V2")
    print("==============================")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()


