from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    print("[INFO] Loading V2 dataset...")
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

    print("[INFO] Training feature importance V2 model...")
    model.fit(X_train, y_train)

    importances = model.feature_importances_

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS feature_importance_v2;")

    cursor.execute("""
        CREATE TABLE feature_importance_v2 (
            id BIGSERIAL PRIMARY KEY,
            feature_name TEXT,
            importance DOUBLE PRECISION,
            importance_pct DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    rows = []

    for feature_name, importance in zip(FEATURES, importances):
        rows.append((
            feature_name,
            float(importance),
            float(importance * 100)
        ))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    for row in rows:
        cursor.execute("""
            INSERT INTO feature_importance_v2 (
                feature_name,
                importance,
                importance_pct
            )
            VALUES (%s, %s, %s);
        """, row)

    conn.commit()

    print("==============================")
    print("FEATURE IMPORTANCE V2 COMPLETE")
    print("==============================")

    for feature_name, importance, importance_pct in rows:
        print(f"{feature_name}: {importance_pct:.2f}%")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


