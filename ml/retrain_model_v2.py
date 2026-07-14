from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "ml/signal_success_model_v2.pkl"

FEATURES = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "market_regime_encoded",
    "momentum_5m",
    "momentum_15m",
    "momentum_30m",
]


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        qm.rsi,
        qm.z_score,
        qm.rolling_volatility,
        qm.liquidity_pressure,
        qm.market_regime,
        COALESCE(mf.momentum_5m, 0) AS momentum_5m,
        COALESCE(mf.momentum_15m, 0) AS momentum_15m,
        COALESCE(mf.momentum_30m, 0) AS momentum_30m,
        sv.signal_success
    FROM signal_validation sv
    JOIN quant_metrics qm
        ON sv.symbol = qm.symbol
       AND sv.signal_time = qm.metric_time
    LEFT JOIN bar_momentum_features mf
        ON qm.symbol = mf.symbol
       AND date_trunc('minute', qm.metric_time) = mf.bar_time
    WHERE sv.signal_success IS NOT NULL
      AND qm.rsi IS NOT NULL
      AND qm.z_score IS NOT NULL
      AND qm.rolling_volatility IS NOT NULL
      AND qm.liquidity_pressure IS NOT NULL
      AND qm.market_regime IS NOT NULL;
    """

    print("[INFO] Loading dataset...")

    df = pd.read_sql(query, conn)

    print(f"[INFO] Rows loaded: {len(df)}")

    if df.empty:
        print("[ERROR] No data loaded.")
        conn.close()
        return

    df = df.dropna()

    encoder = LabelEncoder()
    df["market_regime_encoded"] = encoder.fit_transform(
        df["market_regime"].astype(str)
    )

    X = df[FEATURES]
    y = df["signal_success"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )

    print("[INFO] Training V2 model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "market_regime_encoder": encoder,
        },
        MODEL_PATH,
    )

    cursor = conn.cursor()

    cursor.execute("""
    DROP TABLE IF EXISTS model_v2_metrics;
    """)

    cursor.execute("""
    CREATE TABLE model_v2_metrics (
        id BIGSERIAL PRIMARY KEY,
        model_name TEXT,
        accuracy_pct DOUBLE PRECISION,
        train_rows INTEGER,
        test_rows INTEGER,
        model_path TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)

    cursor.execute("""
    INSERT INTO model_v2_metrics (
        model_name,
        accuracy_pct,
        train_rows,
        test_rows,
        model_path
    )
    VALUES (%s, %s, %s, %s, %s);
    """, (
        "RandomForest_Momentum_V2",
        accuracy,
        len(X_train),
        len(X_test),
        MODEL_PATH,
    ))

    conn.commit()

    print("==============================")
    print("MODEL V2 COMPLETE")
    print("==============================")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Train rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")
    print(f"Saved    : {MODEL_PATH}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


