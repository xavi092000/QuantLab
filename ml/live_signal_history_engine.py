from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

MODEL_PATH = "ml/return_prediction_model.pkl"


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    config_df = pd.read_sql("""
        SELECT return_threshold
        FROM production_strategy_config
        ORDER BY created_at DESC
        LIMIT 1;
    """, conn)

    return_threshold = float(config_df["return_threshold"].iloc[0])

    risk_df = pd.read_sql("""
        SELECT risk_decision
        FROM risk_budgeting_engine
        ORDER BY created_at DESC
        LIMIT 1;
    """, conn)

    risk_decision = risk_df["risk_decision"].iloc[0]

    live_df = pd.read_sql("""
        WITH latest_quant AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                metric_time,
                rsi,
                z_score,
                rolling_volatility,
                liquidity_pressure,
                market_regime
            FROM quant_metrics
            WHERE rsi IS NOT NULL
              AND z_score IS NOT NULL
              AND rolling_volatility IS NOT NULL
              AND liquidity_pressure IS NOT NULL
              AND market_regime IS NOT NULL
            ORDER BY symbol, metric_time DESC
        ),
        latest_momentum AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                bar_time,
                momentum_5m,
                momentum_15m,
                momentum_30m
            FROM bar_momentum_features
            ORDER BY symbol, bar_time DESC
        )
        SELECT
            q.symbol,
            q.metric_time,
            q.rsi,
            q.z_score,
            q.rolling_volatility,
            q.liquidity_pressure,
            q.market_regime,
            COALESCE(m.momentum_5m, 0) AS momentum_5m,
            COALESCE(m.momentum_15m, 0) AS momentum_15m,
            COALESCE(m.momentum_30m, 0) AS momentum_30m
        FROM latest_quant q
        LEFT JOIN latest_momentum m
            ON q.symbol = m.symbol;
    """, conn)

    if live_df.empty:
        print("[ERROR] No live rows found.")
        conn.close()
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    encoder = bundle["market_regime_encoder"]
    features = bundle["features"]

    known_classes = set(encoder.classes_)

    live_df["market_regime_safe"] = live_df["market_regime"].apply(
        lambda value: value if value in known_classes else encoder.classes_[0]
    )

    live_df["market_regime_encoded"] = encoder.transform(
        live_df["market_regime_safe"].astype(str)
    )

    live_df["predicted_return_5m"] = model.predict(live_df[features])

    def get_research_signal(predicted_return):
        if predicted_return > return_threshold:
            return "BUY_CANDIDATE"
        if predicted_return > 0:
            return "WATCH"
        return "AVOID"

    live_df["research_signal"] = live_df["predicted_return_5m"].apply(
        get_research_signal
    )

    def get_final_decision(row):
        if risk_decision == "NO_RISK_ALLOCATED":
            return "NO_TRADE"

        if row["predicted_return_5m"] > return_threshold:
            return "BUY"

        if row["predicted_return_5m"] > 0:
            return "WATCH"

        return "NO_TRADE"

    live_df["final_decision"] = live_df.apply(get_final_decision, axis=1)

    live_df["decision_reason"] = live_df.apply(
        lambda row: (
            "Risk budget blocks live trading"
            if risk_decision == "NO_RISK_ALLOCATED"
            else "Predicted return passed threshold"
            if row["predicted_return_5m"] > return_threshold
            else "Predicted return below threshold"
        ),
        axis=1,
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_signal_history (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            metric_time TIMESTAMPTZ,
            market_regime TEXT,
            predicted_return_5m DOUBLE PRECISION,
            return_threshold DOUBLE PRECISION,
            research_signal TEXT,
            risk_decision TEXT,
            final_decision TEXT,
            decision_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    for _, row in live_df.iterrows():
        cursor.execute("""
            INSERT INTO live_signal_history (
                symbol,
                metric_time,
                market_regime,
                predicted_return_5m,
                return_threshold,
                research_signal,
                risk_decision,
                final_decision,
                decision_reason
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            row["symbol"],
            row["metric_time"],
            row["market_regime"],
            float(row["predicted_return_5m"]),
            return_threshold,
            row["research_signal"],
            risk_decision,
            row["final_decision"],
            row["decision_reason"],
        ))

    conn.commit()

    print("==============================")
    print("LIVE SIGNAL HISTORY UPDATED")
    print("==============================")
    print(f"Rows inserted : {len(live_df)}")
    print(f"Threshold     : {return_threshold}")
    print(f"Risk decision : {risk_decision}")

    print(live_df[[
        "symbol",
        "market_regime",
        "predicted_return_5m",
        "research_signal",
        "final_decision"
    ]])

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


