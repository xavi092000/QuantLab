from configs.database import DB_CONFIG
import psycopg2

def main():

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
    DROP TABLE IF EXISTS paper_trading_v2;
    """)

    cursor.execute("""
    CREATE TABLE paper_trading_v2 (

        trade_id BIGSERIAL PRIMARY KEY,

        symbol TEXT,

        decision TEXT,

        entry_time TIMESTAMPTZ,

        entry_price DOUBLE PRECISION,

        position_size_usd DOUBLE PRECISION,

        trade_status TEXT,

        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """)

    cursor.execute("""
    SELECT
        symbol,
        combined_decision
    FROM combined_signal_engine
    WHERE combined_decision = 'BUY';
    """)

    buy_signals = cursor.fetchall()

    inserted = 0

    for signal in buy_signals:

        symbol = signal[0]
        decision = signal[1]

        cursor.execute("""
        SELECT price
        FROM market_trades
        WHERE symbol = %s
        ORDER BY event_time DESC
        LIMIT 1;
        """, (symbol,))

        price_row = cursor.fetchone()

        if price_row is None:
            continue

        entry_price = float(price_row[0])

        cursor.execute("""
        INSERT INTO paper_trading_v2 (

            symbol,
            decision,
            entry_time,
            entry_price,
            position_size_usd,
            trade_status

        )
        VALUES (
            %s,
            %s,
            NOW(),
            %s,
            %s,
            %s
        );
        """,
        (
            symbol,
            decision,
            entry_price,
            10000.0,
            "OPEN"
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("PAPER TRADING ENGINE")
    print("==============================")
    print("Trades Opened:", inserted)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


