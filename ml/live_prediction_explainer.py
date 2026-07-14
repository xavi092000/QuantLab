from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

def explain_signal(row):
    reasons = []

    if row["predicted_return_5m"] <= 0:
        reasons.append("Predicted return is negative")

    if row["predicted_return_5m"] < row["return_threshold"]:
        reasons.append("Predicted return is below strategy threshold")

    if row["market_regime"] in [
        "LIQUIDITY_EVENT",
        "STATISTICAL_ANOMALY",
        "VWAP_DISLOCATION",
    ]:
        reasons.append(f"Market regime is high-risk: {row['market_regime']}")

    if row["rsi"] >= 80:
        reasons.append("RSI indicates overbought conditions")

    if row["rsi"] <= 20:
        reasons.append("RSI indicates oversold/extreme weakness")

    if abs(row["z_score"]) >= 3:
        reasons.append("Z-score indicates statistical anomaly")

    if row["liquidity_pressure"] >= 2:
        reasons.append("Liquidity pressure is elevated")

    if not reasons:
        reasons.append("No major risk flag detected")

    return " | ".join(reasons)


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
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
    )
    SELECT
        l.symbol,
        l.metric_time,
        q.rsi,
        q.z_score,
        q.rolling_volatility,
        q.liquidity_pressure,
        l.market_regime,
        l.predicted_return_5m,
        l.return_threshold,
        l.research_signal,
        l.final_decision
    FROM live_return_signals l
    JOIN latest_quant q
        ON l.symbol = q.symbol
    ORDER BY l.predicted_return_5m DESC;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No live signals found.")
        conn.close()
        return

    df["explanation"] = df.apply(explain_signal, axis=1)

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS live_prediction_explanations;")

    cursor.execute("""
        CREATE TABLE live_prediction_explanations (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            metric_time TIMESTAMPTZ,
            market_regime TEXT,
            predicted_return_5m DOUBLE PRECISION,
            return_threshold DOUBLE PRECISION,
            research_signal TEXT,
            final_decision TEXT,
            explanation TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO live_prediction_explanations (
                symbol,
                metric_time,
                market_regime,
                predicted_return_5m,
                return_threshold,
                research_signal,
                final_decision,
                explanation
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            row["symbol"],
            row["metric_time"],
            row["market_regime"],
            float(row["predicted_return_5m"]),
            float(row["return_threshold"]),
            row["research_signal"],
            row["final_decision"],
            row["explanation"],
        ))

    conn.commit()

    print("==============================")
    print("LIVE PREDICTION EXPLAINER")
    print("==============================")

    print(df[[
        "symbol",
        "market_regime",
        "predicted_return_5m",
        "research_signal",
        "final_decision",
        "explanation"
    ]])

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


