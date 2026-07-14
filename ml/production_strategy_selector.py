from configs.database import DB_CONFIG
import psycopg2

BEST_STRATEGY = "Mean_Reversion_z-2.5_rsi30"


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS production_strategy_signals;")

    cursor.execute("""
        CREATE TABLE production_strategy_signals (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            metric_time TIMESTAMPTZ,
            rsi DOUBLE PRECISION,
            z_score DOUBLE PRECISION,
            market_regime TEXT,
            strategy_name TEXT,
            strategy_signal TEXT,
            decision_reason TEXT,
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

        strategy_signal = "AVOID"
        decision_reason = "Strategy conditions not met"

        if market_regime in [
            "LIQUIDITY_EVENT",
            "STATISTICAL_ANOMALY",
            "VWAP_DISLOCATION"
        ]:
            strategy_signal = "NO_TRADE"
            decision_reason = f"High-risk regime: {market_regime}"

        elif z_score < -2.5 and rsi < 30:
            strategy_signal = "BUY"
            decision_reason = "Mean reversion conditions met: z_score < -2.5 and RSI < 30"

        elif z_score < -2.0 and rsi < 35:
            strategy_signal = "WATCH"
            decision_reason = "Near mean reversion setup"

        cursor.execute("""
            INSERT INTO production_strategy_signals (
                symbol,
                metric_time,
                rsi,
                z_score,
                market_regime,
                strategy_name,
                strategy_signal,
                decision_reason
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            metric_time,
            rsi,
            z_score,
            market_regime,
            BEST_STRATEGY,
            strategy_signal,
            decision_reason,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("PRODUCTION STRATEGY SELECTOR")
    print("==============================")
    print("Strategy:", BEST_STRATEGY)
    print("Rows processed:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            ROUND(rsi::numeric, 2),
            ROUND(z_score::numeric, 3),
            market_regime,
            strategy_signal,
            decision_reason
        FROM production_strategy_signals
        ORDER BY
            CASE
                WHEN strategy_signal = 'BUY' THEN 1
                WHEN strategy_signal = 'WATCH' THEN 2
                WHEN strategy_signal = 'AVOID' THEN 3
                ELSE 4
            END,
            symbol;
    """)

    for r in cursor.fetchall():
        print(
            f"{r[0]} | RSI={r[1]} | Z={r[2]} | "
            f"regime={r[3]} | signal={r[4]} | reason={r[5]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


