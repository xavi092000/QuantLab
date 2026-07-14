from configs.database import DB_CONFIG
import psycopg2

BUY_Z_THRESHOLD = -2.5
BUY_RSI_THRESHOLD = 30
WATCH_Z_THRESHOLD = -2.0
WATCH_RSI_THRESHOLD = 35

HIGH_RISK_REGIMES = [
    "LIQUIDITY_EVENT",
    "STATISTICAL_ANOMALY",
    "VWAP_DISLOCATION",
]


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS strategy_analyzer_results;")

    cursor.execute("""
        CREATE TABLE strategy_analyzer_results (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            metric_time TIMESTAMPTZ,
            rsi DOUBLE PRECISION,
            z_score DOUBLE PRECISION,
            market_regime TEXT,
            strategy_signal TEXT,
            signal_quality_score DOUBLE PRECISION,
            blocking_reason TEXT,
            detailed_explanation TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT DISTINCT ON (symbol)
            symbol,
            metric_time,
            rsi,
            z_score,
            market_regime
        FROM quant_metrics
        WHERE rsi IS NOT NULL
          AND z_score IS NOT NULL
          AND market_regime IS NOT NULL
        ORDER BY symbol, metric_time DESC;
    """)

    rows = cursor.fetchall()
    inserted = 0

    for row in rows:
        symbol = row[0]
        metric_time = row[1]
        rsi = float(row[2])
        z_score = float(row[3])
        market_regime = row[4]

        signal = "AVOID"
        blocking_reason = "Strategy conditions not met"

        quality_score = 0

        rsi_distance = BUY_RSI_THRESHOLD - rsi
        z_distance = BUY_Z_THRESHOLD - z_score

        if market_regime not in HIGH_RISK_REGIMES:
            quality_score += 25

        if rsi < BUY_RSI_THRESHOLD:
            quality_score += 30
        elif rsi < WATCH_RSI_THRESHOLD:
            quality_score += 15

        if z_score < BUY_Z_THRESHOLD:
            quality_score += 35
        elif z_score < WATCH_Z_THRESHOLD:
            quality_score += 20

        if rsi < BUY_RSI_THRESHOLD and z_score < BUY_Z_THRESHOLD:
            quality_score += 10

        if market_regime in HIGH_RISK_REGIMES:
            signal = "NO_TRADE"
            blocking_reason = f"High-risk regime: {market_regime}"

        elif z_score < BUY_Z_THRESHOLD and rsi < BUY_RSI_THRESHOLD:
            signal = "BUY"
            blocking_reason = "No blocker"

        elif z_score < WATCH_Z_THRESHOLD and rsi < WATCH_RSI_THRESHOLD:
            signal = "WATCH"
            blocking_reason = "Near BUY threshold"

        else:
            signal = "AVOID"
            blocking_reason = "RSI and/or Z-score not close enough"

        details = []

        if market_regime in HIGH_RISK_REGIMES:
            details.append(f"Market regime blocks trade: {market_regime}")
        else:
            details.append(f"Market regime acceptable: {market_regime}")

        if rsi < BUY_RSI_THRESHOLD:
            details.append(f"RSI condition passed: {rsi:.2f} < {BUY_RSI_THRESHOLD}")
        else:
            details.append(f"RSI condition failed: {rsi:.2f} >= {BUY_RSI_THRESHOLD}")

        if z_score < BUY_Z_THRESHOLD:
            details.append(f"Z-score condition passed: {z_score:.3f} < {BUY_Z_THRESHOLD}")
        else:
            details.append(f"Z-score condition failed: {z_score:.3f} >= {BUY_Z_THRESHOLD}")

        if signal == "WATCH":
            details.append("Asset is close to BUY but not fully confirmed.")

        if signal == "BUY":
            details.append("Both mean reversion production conditions are satisfied.")

        if signal == "AVOID":
            details.append("Signal is too weak for current production strategy.")

        detailed_explanation = " | ".join(details)

        cursor.execute("""
            INSERT INTO strategy_analyzer_results (
                symbol,
                metric_time,
                rsi,
                z_score,
                market_regime,
                strategy_signal,
                signal_quality_score,
                blocking_reason,
                detailed_explanation
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            metric_time,
            rsi,
            z_score,
            market_regime,
            signal,
            quality_score,
            blocking_reason,
            detailed_explanation,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("STRATEGY ANALYZER")
    print("==============================")
    print("Rows processed:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            ROUND(rsi::numeric, 2),
            ROUND(z_score::numeric, 3),
            market_regime,
            strategy_signal,
            ROUND(signal_quality_score::numeric, 2),
            blocking_reason
        FROM strategy_analyzer_results
        ORDER BY signal_quality_score DESC;
    """)

    for r in cursor.fetchall():
        print(
            f"{r[0]} | RSI={r[1]} | Z={r[2]} | "
            f"regime={r[3]} | signal={r[4]} | "
            f"quality={r[5]} | blocker={r[6]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


