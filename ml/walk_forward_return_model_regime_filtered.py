from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

TRAIN_SIZE = 1000
TEST_SIZE = 300
RETURN_THRESHOLD = 0.0

ALLOWED_REGIMES = [
    "NORMAL",
    "VOLATILE_MOMENTUM",
]


def compute_profit_factor(trades):
    wins = trades[trades["future_return_5m"] > 0]
    losses = trades[trades["future_return_5m"] < 0]

    if len(losses) == 0:
        return 999.0

    loss_sum = abs(losses["future_return_5m"].sum())

    if loss_sum == 0:
        return 999.0

    return wins["future_return_5m"].sum() / loss_sum


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        sv.signal_time,
        qm.rsi,
        qm.z_score,
        qm.rolling_volatility,
        qm.liquidity_pressure,
        qm.market_regime,
        COALESCE(mf.momentum_5m,0) AS momentum_5m,
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
      AND qm.rsi IS NOT NULL
      AND qm.z_score IS NOT NULL
      AND qm.rolling_volatility IS NOT NULL
      AND qm.liquidity_pressure IS NOT NULL
      AND qm.market_regime IS NOT NULL
    ORDER BY sv.signal_time ASC;
    """

    print("[INFO] Loading regime-filtered walk-forward dataset...")
    df = pd.read_sql(query, conn)
    print(f"[INFO] Rows loaded: {len(df)}")

    if len(df) < TRAIN_SIZE + TEST_SIZE:
        print("[ERROR] Not enough rows.")
        conn.close()
        return

    df = df.dropna().reset_index(drop=True)

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

    results = []
    window_id = 1
    start = 0

    print("======================================")
    print("REGIME-FILTERED WALK-FORWARD MODEL")
    print("======================================")

    while start + TRAIN_SIZE + TEST_SIZE <= len(df):
        train_df = df.iloc[start:start + TRAIN_SIZE]
        test_df = df.iloc[start + TRAIN_SIZE:start + TRAIN_SIZE + TEST_SIZE].copy()

        X_train = train_df[features]
        y_train = train_df["future_return_5m"]

        X_test = test_df[features]
        y_test = test_df["future_return_5m"]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_df["predicted_return"] = predictions

        trades = test_df[
            (test_df["predicted_return"] > RETURN_THRESHOLD)
            &
            (test_df["market_regime"].isin(ALLOWED_REGIMES))
        ].copy()

        mae = mean_absolute_error(y_test, predictions)

        if len(trades) == 0:
            win_rate = 0.0
            profit_factor = 0.0
            total_return = 0.0
            avg_return = 0.0
        else:
            wins = trades[trades["future_return_5m"] > 0]
            win_rate = (len(wins) / len(trades)) * 100
            profit_factor = compute_profit_factor(trades)
            total_return = trades["future_return_5m"].sum()
            avg_return = trades["future_return_5m"].mean()

        results.append((
            window_id,
            train_df["signal_time"].min(),
            train_df["signal_time"].max(),
            test_df["signal_time"].min(),
            test_df["signal_time"].max(),
            len(train_df),
            len(test_df),
            len(trades),
            float(mae),
            float(avg_return),
            float(total_return),
            float(win_rate),
            float(profit_factor),
        ))

        print(
            f"Window {window_id:02d} | "
            f"Trades {len(trades)} | "
            f"Avg {avg_return:.6f} | "
            f"Total {total_return:.4f} | "
            f"WinRate {win_rate:.2f}% | "
            f"PF {profit_factor:.4f}"
        )

        window_id += 1
        start += TEST_SIZE

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS walk_forward_return_regime_filtered;")

    cursor.execute("""
        CREATE TABLE walk_forward_return_regime_filtered (
            window_id INTEGER PRIMARY KEY,
            train_start TIMESTAMPTZ,
            train_end TIMESTAMPTZ,
            test_start TIMESTAMPTZ,
            test_end TIMESTAMPTZ,
            train_rows INTEGER,
            test_rows INTEGER,
            trades_taken INTEGER,
            mae DOUBLE PRECISION,
            avg_return_pct DOUBLE PRECISION,
            total_return_pct DOUBLE PRECISION,
            win_rate_pct DOUBLE PRECISION,
            profit_factor DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.executemany("""
        INSERT INTO walk_forward_return_regime_filtered (
            window_id,
            train_start,
            train_end,
            test_start,
            test_end,
            train_rows,
            test_rows,
            trades_taken,
            mae,
            avg_return_pct,
            total_return_pct,
            win_rate_pct,
            profit_factor
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, results)

    cursor.execute("""
        CREATE OR REPLACE VIEW walk_forward_return_regime_summary AS
        SELECT
            COUNT(*) AS windows_tested,
            ROUND(AVG(profit_factor)::numeric, 4) AS avg_profit_factor,
            ROUND(MIN(profit_factor)::numeric, 4) AS worst_profit_factor,
            ROUND(MAX(profit_factor)::numeric, 4) AS best_profit_factor,
            ROUND(AVG(win_rate_pct)::numeric, 2) AS avg_win_rate_pct,
            ROUND(SUM(total_return_pct)::numeric, 4) AS total_return_pct,
            ROUND(AVG(trades_taken)::numeric, 2) AS avg_trades_per_window
        FROM walk_forward_return_regime_filtered;
    """)

    conn.commit()

    print("======================================")
    print("REGIME-FILTERED WALK-FORWARD COMPLETE")
    print("======================================")
    print(f"Windows tested: {len(results)}")
    print("Saved table: walk_forward_return_regime_filtered")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


