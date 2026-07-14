from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "ml/challenger_extra_trees_signal_success_model.pkl"

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

    sv.signal_success

FROM signal_validation sv

JOIN quant_metrics qm
ON sv.symbol = qm.symbol
AND sv.signal_time = qm.metric_time

LEFT JOIN bar_momentum_features mf
ON qm.symbol = mf.symbol
AND date_trunc('minute', qm.metric_time) = mf.bar_time

WHERE sv.signal_success IS NOT NULL;
"""

print("[INFO] Loading challenger dataset...")

df = pd.read_sql(query, conn)
df = df.dropna()

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

X = df[features]
y = df["signal_success"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("[INFO] Training challenger ExtraTreesClassifier...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100
precision = precision_score(y_test, predictions, zero_division=0) * 100
recall = recall_score(y_test, predictions, zero_division=0) * 100
f1 = f1_score(y_test, predictions, zero_division=0) * 100

joblib.dump(
    {
        "model": model,
        "features": features,
        "market_regime_encoder": encoder,
        "model_type": "ExtraTreesClassifier",
    },
    MODEL_PATH
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS retrained_model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT,
    accuracy_pct DOUBLE PRECISION,
    model_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
ALTER TABLE retrained_model_metrics
ADD COLUMN IF NOT EXISTS model_type TEXT,
ADD COLUMN IF NOT EXISTS precision_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS recall_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS f1_score_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS training_rows INTEGER,
ADD COLUMN IF NOT EXISTS feature_set_name TEXT,
ADD COLUMN IF NOT EXISTS validation_method TEXT;
""")

cursor.execute("""
INSERT INTO retrained_model_metrics (
    model_name,
    model_type,
    accuracy_pct,
    precision_pct,
    recall_pct,
    f1_score_pct,
    training_rows,
    feature_set_name,
    validation_method,
    model_path
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
""", (
    "ExtraTrees_Challenger_Signal_Success",
    "ExtraTreesClassifier",
    float(accuracy),
    float(precision),
    float(recall),
    float(f1),
    int(len(df)),
    "Quant Metrics + Momentum Features",
    "Holdout validation 75/25 stratified split",
    MODEL_PATH,
))

conn.commit()

print("==============================")
print("CHALLENGER MODEL COMPLETE")
print("==============================")
print("Model        : ExtraTrees_Challenger_Signal_Success")
print("Type         : ExtraTreesClassifier")
print("Rows used    :", len(df))
print("Accuracy %   :", round(accuracy, 2))
print("Precision %  :", round(precision, 2))
print("Recall %     :", round(recall, 2))
print("F1 Score %   :", round(f1, 2))
print("Saved        :", MODEL_PATH)
print("------------------------------")
print(classification_report(y_test, predictions, zero_division=0))

cursor.close()
conn.close()


