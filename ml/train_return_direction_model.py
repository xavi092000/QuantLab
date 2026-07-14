from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "ml/return_direction_model.pkl"

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

print("[INFO] Loading direction dataset...")

df = pd.read_sql(query, conn)

df = df.dropna()

print("[INFO] Rows loaded:", len(df))

df["target_up"] = (df["future_return_5m"] > 0).astype(int)

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
y = df["target_up"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("[INFO] Training return direction classifier...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

joblib.dump(
    {
        "model": model,
        "features": features,
        "market_regime_encoder": encoder
    },
    MODEL_PATH
)

cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS return_direction_model_metrics;")

cursor.execute("""
CREATE TABLE return_direction_model_metrics (
    id BIGSERIAL PRIMARY KEY,
    rows_used INTEGER,
    accuracy_pct DOUBLE PRECISION,
    positive_rate_pct DOUBLE PRECISION,
    avg_probability_up DOUBLE PRECISION,
    model_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
INSERT INTO return_direction_model_metrics (
    rows_used,
    accuracy_pct,
    positive_rate_pct,
    avg_probability_up,
    model_path
)
VALUES (%s,%s,%s,%s,%s);
""", (
    len(df),
    float(accuracy * 100),
    float(df["target_up"].mean() * 100),
    float(probabilities.mean() * 100),
    MODEL_PATH
))

conn.commit()

print("==============================")
print("RETURN DIRECTION MODEL COMPLETE")
print("==============================")
print("Rows used          :", len(df))
print("Accuracy %         :", round(accuracy * 100, 2))
print("Positive Rate %    :", round(df["target_up"].mean() * 100, 2))
print("Avg Probability Up :", round(probabilities.mean() * 100, 2))
print("Saved              :", MODEL_PATH)
print("------------------------------")
print(classification_report(y_test, predictions))

cursor.close()
conn.close()


