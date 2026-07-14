from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

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
WHERE sv.future_return_5m IS NOT NULL;
"""

df = pd.read_sql(query, conn).dropna()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoder = bundle["market_regime_encoder"]
features = bundle["features"]

df["market_regime_encoded"] = encoder.transform(df["market_regime"].astype(str))
df["predicted_return_5m"] = model.predict(df[features])

print("==============================")
print("RETURN PREDICTION DIAGNOSTICS")
print("==============================")
print("Rows:", len(df))
print("Actual min:", df["future_return_5m"].min())
print("Actual avg:", df["future_return_5m"].mean())
print("Actual max:", df["future_return_5m"].max())
print("Actual positive %:", (df["future_return_5m"] > 0).mean() * 100)
print("------------------------------")
print("Pred min:", df["predicted_return_5m"].min())
print("Pred avg:", df["predicted_return_5m"].mean())
print("Pred max:", df["predicted_return_5m"].max())
print("Pred positive %:", (df["predicted_return_5m"] > 0).mean() * 100)

conn.close()


