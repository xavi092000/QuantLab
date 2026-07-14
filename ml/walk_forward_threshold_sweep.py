from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

TRAIN_SIZE = 1000
TEST_SIZE = 300

THRESHOLDS = [
    0.0000,
    0.0001,
    0.0002,
    0.0003,
    0.0004,
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

    print("[INFO] Loading dataset...")
    df = pd.read_sql(query, conn)
    print(f"[INFO] Rows loaded: {len(df)}")

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

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS threshold_sweep_results;")

    cursor.execute("""
        CREATE TABLE threshold_sweep_results (
            threshold DOUBLE PRECISION,
            windows_tested INTEGER,
            profitable_windows INTEGER,
            avg_profit_factor DOUBLE PRECISION,
            total_return DOUBLE PRECISION,
            avg_trades_per_window DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    print("==============================")
    print("WALK-FORWARD THRESHOLD SWEEP")
    print("==============================")

    for threshold in THRESHOLDS:
        start = 0
        total_windows = 0
        profitable_windows = 0

        pf_list = []
        return_list = []
        trades_list = []

        while start + TRAIN_SIZE + TEST_SIZE <= len(df):
            train_df = df.iloc[start:start + TRAIN_SIZE]
            test_df = df.iloc[
                start + TRAIN_SIZE:
                start + TRAIN_SIZE + TEST_SIZE
            ].copy()

            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                train_df[features],
                train_df["future_return_5m"]
            )

            test_df["predicted_return"] = model.predict(
                test_df[features]
            )

            trades = test_df[
                test_df["predicted_return"] > threshold
            ].copy()

            total_windows += 1

            if len(trades) > 0:
                pf = compute_profit_factor(trades)
                total_ret = trades["future_return_5m"].sum()

                if total_ret > 0:
                    profitable_windows += 1

                pf_list.append(float(pf))
                return_list.append(float(total_ret))
                trades_list.append(int(len(trades)))

            start += TEST_SIZE

        avg_pf = float(sum(pf_list) / len(pf_list)) if pf_list else 0.0
        total_return = float(sum(return_list)) if return_list else 0.0
        avg_trades = float(sum(trades_list) / len(trades_list)) if trades_list else 0.0

        cursor.execute("""
            INSERT INTO threshold_sweep_results (
                threshold,
                windows_tested,
                profitable_windows,
                avg_profit_factor,
                total_return,
                avg_trades_per_window
            )
            VALUES (%s,%s,%s,%s,%s,%s);
        """, (
            float(threshold),
            int(total_windows),
            int(profitable_windows),
            float(avg_pf),
            float(total_return),
            float(avg_trades),
        ))

        print(
            f"Threshold={threshold:.4f} | "
            f"PF={avg_pf:.4f} | "
            f"PositiveWindows={profitable_windows}/{total_windows} | "
            f"TotalReturn={total_return:.4f} | "
            f"AvgTrades={avg_trades:.2f}"
        )

    conn.commit()

    print("==============================")
    print("THRESHOLD SWEEP COMPLETE")
    print("==============================")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


