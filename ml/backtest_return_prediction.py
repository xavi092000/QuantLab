from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

MODEL_PATH = "ml/return_prediction_model.pkl"

RETURN_THRESHOLDS = [
    0.0000,
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
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
      AND qm.rsi IS NOT NULL
      AND qm.z_score IS NOT NULL
      AND qm.rolling_volatility IS NOT NULL
      AND qm.liquidity_pressure IS NOT NULL
      AND qm.market_regime IS NOT NULL;
    """

    print("[INFO] Loading backtest dataset...")
    df = pd.read_sql(query, conn)
    print(f"[INFO] Rows loaded: {len(df)}")

    if df.empty:
        print("[ERROR] No rows loaded.")
        conn.close()
        return

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    encoder = model_bundle["market_regime_encoder"]
    features = model_bundle["features"]

    df["market_regime_encoded"] = encoder.transform(
        df["market_regime"].astype(str)
    )

    X = df[features]

    df["predicted_return_5m"] = model.predict(X)

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS return_prediction_backtest;")

    cursor.execute("""
        CREATE TABLE return_prediction_backtest (
            id BIGSERIAL PRIMARY KEY,
            return_threshold DOUBLE PRECISION,
            trades_taken INTEGER,
            avg_predicted_return DOUBLE PRECISION,
            avg_actual_return DOUBLE PRECISION,
            total_actual_return DOUBLE PRECISION,
            win_rate_pct DOUBLE PRECISION,
            profit_factor DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    print("==============================")
    print("RETURN PREDICTION BACKTEST")
    print("==============================")

    for threshold in RETURN_THRESHOLDS:
        trades = df[df["predicted_return_5m"] > threshold].copy()

        if len(trades) == 0:
            avg_pred = 0.0
            avg_actual = 0.0
            total_actual = 0.0
            win_rate = 0.0
            profit_factor = 0.0
        else:
            avg_pred = trades["predicted_return_5m"].mean()
            avg_actual = trades["future_return_5m"].mean()
            total_actual = trades["future_return_5m"].sum()

            wins = trades[trades["future_return_5m"] > 0]
            losses = trades[trades["future_return_5m"] < 0]

            win_rate = (len(wins) / len(trades)) * 100

            if len(losses) == 0:
                profit_factor = 999.0
            else:
                profit_factor = (
                    wins["future_return_5m"].sum()
                    / abs(losses["future_return_5m"].sum())
                )

        cursor.execute("""
            INSERT INTO return_prediction_backtest (
                return_threshold,
                trades_taken,
                avg_predicted_return,
                avg_actual_return,
                total_actual_return,
                win_rate_pct,
                profit_factor
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            threshold,
            len(trades),
            float(avg_pred),
            float(avg_actual),
            float(total_actual),
            float(win_rate),
            float(profit_factor),
        ))

        print(
            f"Threshold {threshold:.4f} | "
            f"Trades {len(trades)} | "
            f"AvgActual {avg_actual:.6f} | "
            f"Total {total_actual:.4f} | "
            f"WinRate {win_rate:.2f}% | "
            f"PF {profit_factor:.4f}"
        )

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


