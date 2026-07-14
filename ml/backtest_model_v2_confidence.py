from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

MODEL_PATH = "ml/signal_success_model_v2.pkl"

CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
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
      AND qm.market_regime IS NOT NULL;
    """

    print("[INFO] Loading dataset...")
    df = pd.read_sql(query, conn)
    print(f"[INFO] Rows loaded: {len(df)}")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    encoder = model_bundle["market_regime_encoder"]

    df["market_regime_encoded"] = encoder.transform(
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

    probabilities = model.predict_proba(X)[:, 1]
    df["success_probability"] = probabilities

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS backtest_v2_confidence_results;")

    cursor.execute("""
        CREATE TABLE backtest_v2_confidence_results (
            id BIGSERIAL PRIMARY KEY,
            confidence_threshold DOUBLE PRECISION,
            trades_taken INTEGER,
            avg_return_pct DOUBLE PRECISION,
            total_return_pct DOUBLE PRECISION,
            win_rate_pct DOUBLE PRECISION,
            profit_factor DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    print("==============================")
    print("MODEL V2 CONFIDENCE BACKTEST")
    print("==============================")

    for threshold in CONFIDENCE_THRESHOLDS:
        trades = df[df["success_probability"] >= threshold].copy()

        if len(trades) == 0:
            avg_return = 0.0
            total_return = 0.0
            win_rate = 0.0
            profit_factor = 0.0
        else:
            avg_return = trades["future_return_5m"].mean()
            total_return = trades["future_return_5m"].sum()

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
            INSERT INTO backtest_v2_confidence_results (
                confidence_threshold,
                trades_taken,
                avg_return_pct,
                total_return_pct,
                win_rate_pct,
                profit_factor
            )
            VALUES (%s,%s,%s,%s,%s,%s);
        """, (
            threshold,
            len(trades),
            float(avg_return),
            float(total_return),
            float(win_rate),
            float(profit_factor),
        ))

        print(
            f"Threshold {threshold:.2f} | "
            f"Trades {len(trades)} | "
            f"Avg {avg_return:.5f} | "
            f"Total {total_return:.4f} | "
            f"WinRate {win_rate:.2f}% | "
            f"PF {profit_factor:.4f}"
        )

    conn.commit()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


