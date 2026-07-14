from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "ml/return_prediction_oos.pkl"

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    sv.signal_time,

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

ORDER BY sv.signal_time
"""

print("[INFO] Loading dataset...")

df = pd.read_sql(query, conn)

print("[INFO] Rows:", len(df))

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

split_index = int(len(df) * 0.70)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("[INFO] Train rows:", len(train_df))
print("[INFO] Test rows :", len(test_df))

X_train = train_df[features]
y_train = train_df["future_return_5m"]

X_test = test_df[features]
y_test = test_df["future_return_5m"]

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("[INFO] Training OOS model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

test_df = test_df.copy()

test_df["predicted_return"] = predictions

trades = test_df[
    test_df["predicted_return"] > 0
]

wins = trades[
    trades["future_return_5m"] > 0
]

losses = trades[
    trades["future_return_5m"] < 0
]

profit_factor = (
    wins["future_return_5m"].sum()
    /
    abs(losses["future_return_5m"].sum())
)

win_rate = (
    len(wins)
    /
    len(trades)
) * 100

total_return = trades[
    "future_return_5m"
].sum()

print("==============================")
print("OUT OF SAMPLE RESULTS")
print("==============================")
print("MAE           :", round(mae,6))
print("Trades        :", len(trades))
print("Win Rate      :", round(win_rate,2))
print("Profit Factor :", round(profit_factor,4))
print("Total Return  :", round(total_return,4))

joblib.dump(
    model,
    MODEL_PATH
)

conn.close()


