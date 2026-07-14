from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "ml/return_prediction_model.pkl"

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

WHERE sv.future_return_5m IS NOT NULL
"""

print("[INFO] Loading dataset...")

df = pd.read_sql(query, conn)

print("[INFO] Rows loaded:", len(df))

df = df.dropna()

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
    "momentum_30m"
]

X = df[features]

y = df["future_return_5m"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("[INFO] Training return prediction model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "market_regime_encoder": encoder
    },
    MODEL_PATH
)

print("==============================")
print("RETURN PREDICTION COMPLETE")
print("==============================")
print("MAE :", round(mae,6))
print("Saved :", MODEL_PATH)

conn.close()


