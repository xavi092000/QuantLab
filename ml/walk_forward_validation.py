from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

TRAIN_SIZE = 1000
TEST_SIZE = 300


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        qm.metric_time,
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
      AND sv.signal_success IS NOT NULL
    ORDER BY qm.metric_time ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def prepare_data(df):
    regime_encoder = LabelEncoder()
    alert_encoder = LabelEncoder()

    df["market_regime_encoded"] = regime_encoder.fit_transform(
        df["market_regime"].astype(str)
    )

    df["alert_level_encoded"] = alert_encoder.fit_transform(
        df["alert_level"].astype(str)
    )

    df["target"] = df["signal_success"].astype(int)

    return df


def save_results(results):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS walk_forward_validation_results;")

    cursor.execute("""
        CREATE TABLE walk_forward_validation_results (
            window_id INTEGER PRIMARY KEY,
            train_start TIMESTAMPTZ,
            train_end TIMESTAMPTZ,
            test_start TIMESTAMPTZ,
            test_end TIMESTAMPTZ,
            train_rows INTEGER,
            test_rows INTEGER,
            accuracy_pct DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.executemany("""
        INSERT INTO walk_forward_validation_results (
            window_id,
            train_start,
            train_end,
            test_start,
            test_end,
            train_rows,
            test_rows,
            accuracy_pct
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
    """, results)

    conn.commit()
    cursor.close()
    conn.close()


def main():
    print("[INFO] Loading walk-forward dataset...")

    df = load_data()
    print(f"[INFO] Rows loaded: {len(df)}")

    if len(df) < TRAIN_SIZE + TEST_SIZE:
        print("[ERROR] Not enough rows for walk-forward validation.")
        return

    df = prepare_data(df)

    results = []
    window_id = 1

    start = 0

    while start + TRAIN_SIZE + TEST_SIZE <= len(df):
        train_df = df.iloc[start:start + TRAIN_SIZE]
        test_df = df.iloc[start + TRAIN_SIZE:start + TRAIN_SIZE + TEST_SIZE]

        X_train = train_df[FEATURES]
        y_train = train_df["target"]

        X_test = test_df[FEATURES]
        y_test = test_df["target"]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100

        results.append((
            window_id,
            train_df["metric_time"].min(),
            train_df["metric_time"].max(),
            test_df["metric_time"].min(),
            test_df["metric_time"].max(),
            len(train_df),
            len(test_df),
            float(accuracy),
        ))

        print(
            f"[WALK-FORWARD] "
            f"window={window_id} "
            f"accuracy={accuracy:.2f}% "
            f"train_rows={len(train_df)} "
            f"test_rows={len(test_df)}"
        )

        window_id += 1
        start += TEST_SIZE

    save_results(results)

    print("=================================")
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=================================")
    print(f"Windows tested : {len(results)}")
    print("Saved table    : walk_forward_validation_results")


if __name__ == "__main__":
    main()


