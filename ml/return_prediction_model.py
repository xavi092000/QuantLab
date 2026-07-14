from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from configs.database import DB_CONFIG


MODEL_PATH = "ml/return_prediction_model.pkl"
LOWER_TARGET_QUANTILE = 0.01
UPPER_TARGET_QUANTILE = 0.99
TRAIN_FRACTION = 0.75


QUERY = """
SELECT
    sv.signal_time,
    qm.rsi,
    qm.z_score,
    qm.rolling_volatility,
    qm.liquidity_pressure,
    qm.market_regime,
    COALESCE(mf.momentum_5m, 0) AS momentum_5m,
    COALESCE(mf.momentum_15m, 0) AS momentum_15m,
    COALESCE(mf.momentum_30m, 0) AS momentum_30m,
    sv.future_return_5m
FROM signal_validation sv
JOIN quant_metrics qm
    ON sv.symbol = qm.symbol
   AND sv.signal_time = qm.metric_time
LEFT JOIN bar_momentum_features mf
    ON qm.symbol = mf.symbol
   AND date_trunc('minute', qm.metric_time) = mf.bar_time
WHERE sv.future_return_5m IS NOT NULL
ORDER BY sv.signal_time;
"""


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        print("[INFO] Loading dataset...")
        df = pd.read_sql_query(QUERY, conn).dropna()

        if len(df) < 100:
            raise RuntimeError(
                f"Not enough rows to train a reliable model: {len(df)} rows."
            )

        print("[INFO] Rows loaded:", len(df))

        encoder = LabelEncoder()
        df["market_regime_encoded"] = encoder.fit_transform(
            df["market_regime"].astype(str)
        )

        features = [
            "rsi",
            "z_score",
            "rolling_volatility",
            "liquidity_pressure",
            "market_regime_encoded",
            "momentum_5m",
            "momentum_15m",
            "momentum_30m",
        ]

        raw_target = df["future_return_5m"].astype(float)

        lower_bound = raw_target.quantile(LOWER_TARGET_QUANTILE)
        upper_bound = raw_target.quantile(UPPER_TARGET_QUANTILE)

        df["target_clipped"] = raw_target.clip(
            lower=lower_bound,
            upper=upper_bound,
        )

        print("==============================")
        print("TARGET DIAGNOSTICS")
        print("==============================")
        print("Mean raw              :", float(raw_target.mean()))
        print("Median raw            :", float(raw_target.median()))
        print("Positive rate %       :", float((raw_target > 0).mean() * 100))
        print("Lower clip bound      :", float(lower_bound))
        print("Upper clip bound      :", float(upper_bound))
        print("Mean clipped          :", float(df["target_clipped"].mean()))

        split_index = int(len(df) * TRAIN_FRACTION)

        if split_index <= 0 or split_index >= len(df):
            raise RuntimeError("Invalid temporal split index.")

        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

        X_train = train_df[features]
        X_test = test_df[features]
        y_train = train_df["target_clipped"]
        y_test = test_df["future_return_5m"].astype(float)

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        print("[INFO] Training robust return prediction model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        direction_accuracy = np.mean(
            (predictions > 0) == (y_test.to_numpy() > 0)
        )
        prediction_positive_rate = np.mean(predictions > 0)
        actual_positive_rate = np.mean(y_test.to_numpy() > 0)

        joblib.dump(
            {
                "model": model,
                "features": features,
                "market_regime_encoder": encoder,
                "target_clip_bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                },
                "training_metadata": {
                    "rows_used": int(len(df)),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "split_type": "temporal",
                },
            },
            MODEL_PATH,
        )

        print("==============================")
        print("RETURN PREDICTION COMPLETE")
        print("==============================")
        print("Rows used                  :", len(df))
        print("Train rows                 :", len(train_df))
        print("Test rows                  :", len(test_df))
        print("MAE                        :", round(float(mae), 8))
        print(
            "Direction accuracy %       :",
            round(float(direction_accuracy * 100), 2),
        )
        print(
            "Actual positive rate %     :",
            round(float(actual_positive_rate * 100), 2),
        )
        print(
            "Prediction positive rate % :",
            round(float(prediction_positive_rate * 100), 2),
        )
        print("Prediction mean            :", round(float(predictions.mean()), 8))
        print("Prediction min             :", round(float(predictions.min()), 8))
        print("Prediction max             :", round(float(predictions.max()), 8))
        print("Saved                      :", MODEL_PATH)


if __name__ == "__main__":
    main()
