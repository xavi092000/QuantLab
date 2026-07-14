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
]

TARGET = "signal_success"

MODEL_PATH = "ml/signal_success_predictor_model.pkl"


def load_training_data():
    conn = psycopg2.connect(**DB_CONFIG)

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
          AND sv.signal_success IS NOT NULL;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


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
    y = df[TARGET].astype(int)

    return X, y, regime_encoder, alert_encoder


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest_Signal_Success": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting_Signal_Success": GradientBoostingClassifier(
            random_state=42,
        ),
        "LogisticRegression_Signal_Success": LogisticRegression(
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
        CREATE TABLE IF NOT EXISTS signal_success_model_metrics (
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
            INSERT INTO signal_success_model_metrics (
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
        CREATE TABLE IF NOT EXISTS signal_success_feature_importance (
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
            INSERT INTO signal_success_feature_importance (
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
    print("[INFO] Loading signal validation training data...")

    df = load_training_data()

    print(f"[INFO] Rows loaded: {len(df)}")
    print("\n[INFO] Target distribution:")
    print(df[TARGET].value_counts())

    if len(df) < 500:
        print("[ERROR] Not enough validated signals to train reliably.")
        return

    X, y, regime_encoder, alert_encoder = prepare_features(df)

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

    os.makedirs("ml", exist_ok=True)

    joblib.dump(
        {
            "model": best_result["model"],
            "features": FEATURES,
            "regime_encoder": regime_encoder,
            "alert_encoder": alert_encoder,
        },
        MODEL_PATH,
    )

    save_model_metrics(results, best_result)
    save_feature_importance(best_result)

    print("\n==============================")
    print("BEST SIGNAL SUCCESS MODEL")
    print("==============================")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()



