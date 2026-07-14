from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS momentum_strategy_signals (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            market_regime TEXT,
            rsi DOUBLE PRECISION,
            z_score DOUBLE PRECISION,
            momentum_signal TEXT,
            signal_quality DOUBLE PRECISION,
            decision_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("DELETE FROM momentum_strategy_signals;")

    cursor.execute("""
        SELECT symbol
        FROM strategy_router_results
        WHERE selected_strategy = 'MOMENTUM';
    """)

    symbols = [row[0] for row in cursor.fetchall()]
    inserted = 0

    for symbol in symbols:
        cursor.execute("""
            SELECT
                symbol,
                market_regime,
                rsi,
                z_score
            FROM quant_metrics
            WHERE symbol = %s
            ORDER BY metric_time DESC
            LIMIT 1;
        """, (symbol,))

        row = cursor.fetchone()

        if row is None:
            continue

        symbol, market_regime, rsi, z_score = row

        rsi = float(rsi)
        z_score = float(z_score)

        quality = 0.0

        if market_regime == "BULLISH_MOMENTUM":
            quality += 10

        if rsi >= 70:
            quality += 50

        if z_score > 1:
            quality += 40

        if quality >= 90:
            signal = "BUY"
            reason = "Strong bullish momentum confirmed"

        elif quality >= 50:
            signal = "WATCH"
            reason = "Momentum building but not fully confirmed"

        else:
            signal = "AVOID"
            reason = "Momentum insufficient"

        cursor.execute("""
            INSERT INTO momentum_strategy_signals (
                symbol,
                market_regime,
                rsi,
                z_score,
                momentum_signal,
                signal_quality,
                decision_reason
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            market_regime,
            rsi,
            z_score,
            signal,
            quality,
            reason,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("MOMENTUM STRATEGY ENGINE")
    print("==============================")
    print("Assets processed:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            market_regime,
            ROUND(rsi::numeric, 2),
            ROUND(z_score::numeric, 3),
            momentum_signal,
            signal_quality,
            decision_reason
        FROM momentum_strategy_signals
        ORDER BY signal_quality DESC;
    """)

    for r in cursor.fetchall():
        print(
            f"{r[0]} | regime={r[1]} | RSI={r[2]} | Z={r[3]} | "
            f"signal={r[4]} | quality={r[5]} | reason={r[6]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


