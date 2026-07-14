from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)

RETURN_MODEL_PATH = "ml/return_prediction_model.pkl"
DIRECTION_MODEL_PATH = "ml/return_direction_model.pkl"


def safe_encode_market_regime(df, encoder):
    known_classes = set(encoder.classes_)

    df["market_regime_safe"] = df["market_regime"].apply(
        lambda value: value if value in known_classes else encoder.classes_[0]
    )

    df["market_regime_encoded"] = encoder.transform(
        df["market_regime_safe"].astype(str)
    )

    return df


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        qm.rsi,
        qm.z_score,
        qm.rolling_volatility,
        qm.liquidity_pressure,
        qm.market_regime,

        COALESCE(mf.momentum_5m,0)  AS momentum_5m,
        COALESCE(mf.momentum_15m,0) AS momentum_15m,
        COALESCE(mf.momentum_30m,0) AS momentum_30m,

        sv.future_return_5m

    FROM signal_validation sv

    JOIN quant_metrics qm
    ON sv.symbol = qm.symbol
    AND sv.signal_time = qm.metric_time

    LEFT JOIN bar_momentum_features mf
    ON qm.symbol = mf.symbol
    AND date_trunc('minute', qm.metric_time) = mf.bar_time

    WHERE sv.future_return_5m IS NOT NULL;
    """

    print("[INFO] Loading evaluation dataset...")

    df = pd.read_sql(query, conn)
    df = df.dropna()

    print("[INFO] Rows loaded:", len(df))

    if df.empty:
        print("[ERROR] No evaluation data found.")
        conn.close()
        return

    return_bundle = joblib.load(RETURN_MODEL_PATH)
    direction_bundle = joblib.load(DIRECTION_MODEL_PATH)

    return_model = return_bundle["model"]
    return_features = return_bundle["features"]
    return_encoder = return_bundle["market_regime_encoder"]

    direction_model = direction_bundle["model"]
    direction_features = direction_bundle["features"]
    direction_encoder = direction_bundle["market_regime_encoder"]

    return_df = df.copy()
    return_df = safe_encode_market_regime(return_df, return_encoder)

    direction_df = df.copy()
    direction_df = safe_encode_market_regime(direction_df, direction_encoder)

    X_return = return_df[return_features]
    y_return = return_df["future_return_5m"]

    X_direction = direction_df[direction_features]
    y_direction = (direction_df["future_return_5m"] > 0).astype(int)

    return_predictions = return_model.predict(X_return)
    direction_predictions = direction_model.predict(X_direction)
    direction_probabilities = direction_model.predict_proba(X_direction)[:, 1]

    mae = mean_absolute_error(y_return, return_predictions)
    rmse = np.sqrt(mean_squared_error(y_return, return_predictions))
    r2 = r2_score(y_return, return_predictions)

    directional_accuracy_from_return = accuracy_score(
        (y_return > 0).astype(int),
        (return_predictions > 0).astype(int)
    ) * 100

    direction_accuracy = accuracy_score(
        y_direction,
        direction_predictions
    ) * 100

    avg_actual_return = float(y_return.mean())
    avg_predicted_return = float(np.mean(return_predictions))

    prediction_bias = avg_predicted_return - avg_actual_return

    avg_probability_up = float(np.mean(direction_probabilities) * 100)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluation_framework (
            id BIGSERIAL PRIMARY KEY,
            evaluation_name TEXT,
            rows_used INTEGER,
            return_model_path TEXT,
            direction_model_path TEXT,
            mae DOUBLE PRECISION,
            rmse DOUBLE PRECISION,
            r2_score DOUBLE PRECISION,
            directional_accuracy_from_return_pct DOUBLE PRECISION,
            direction_model_accuracy_pct DOUBLE PRECISION,
            avg_actual_return DOUBLE PRECISION,
            avg_predicted_return DOUBLE PRECISION,
            prediction_bias DOUBLE PRECISION,
            avg_probability_up_pct DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        INSERT INTO model_evaluation_framework (
            evaluation_name,
            rows_used,
            return_model_path,
            direction_model_path,
            mae,
            rmse,
            r2_score,
            directional_accuracy_from_return_pct,
            direction_model_accuracy_pct,
            avg_actual_return,
            avg_predicted_return,
            prediction_bias,
            avg_probability_up_pct
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, (
        "Return + Direction Model Evaluation",
        int(len(df)),
        RETURN_MODEL_PATH,
        DIRECTION_MODEL_PATH,
        float(mae),
        float(rmse),
        float(r2),
        float(directional_accuracy_from_return),
        float(direction_accuracy),
        float(avg_actual_return),
        float(avg_predicted_return),
        float(prediction_bias),
        float(avg_probability_up),
    ))

    conn.commit()

    print("==============================")
    print("MODEL EVALUATION FRAMEWORK")
    print("==============================")
    print("Rows Used                         :", len(df))
    print("MAE                               :", round(mae, 8))
    print("RMSE                              :", round(rmse, 8))
    print("R2 Score                          :", round(r2, 4))
    print("Directional Accuracy from Return %:", round(directional_accuracy_from_return, 2))
    print("Direction Model Accuracy %        :", round(direction_accuracy, 2))
    print("Avg Actual Return                 :", round(avg_actual_return, 8))
    print("Avg Predicted Return              :", round(avg_predicted_return, 8))
    print("Prediction Bias                   :", round(prediction_bias, 8))
    print("Avg Probability Up %              :", round(avg_probability_up, 2))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


