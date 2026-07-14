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
    "signal_coherence_score",
    "market_regime_encoded",
    "alert_level_encoded",
]


def main():
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

    if df.empty:
        print("[ERROR] No training data found.")
        conn.close()
        return

    regime_encoder = LabelEncoder()
    alert_encoder = LabelEncoder()

    df["market_regime_encoded"] = regime_encoder.fit_transform(df["market_regime"].astype(str))
    df["alert_level_encoded"] = alert_encoder.fit_transform(df["alert_level"].astype(str))
    df["target"] = df["signal_success"].astype(int)

    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS feature_importance_analysis;")

    cursor.execute("""
        CREATE TABLE feature_importance_analysis (
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
            INSERT INTO feature_importance_analysis (
                feature_name,
                importance,
                importance_pct
            )
            VALUES (%s,%s,%s);
        """, row)

    conn.commit()

    print("==============================")
    print("FEATURE IMPORTANCE COMPLETE")
    print("==============================")

    for feature_name, importance, importance_pct in rows:
        print(f"{feature_name}: {importance_pct:.2f}%")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


