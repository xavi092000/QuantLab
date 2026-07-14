from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS ohlc_1m_bars;")

    cursor.execute("""
        CREATE TABLE ohlc_1m_bars AS
        WITH base AS (
            SELECT
                symbol,
                date_trunc('minute', event_time) AS bar_time,
                event_time,
                price,
                quantity
            FROM market_trades
        ),
        ranked AS (
            SELECT
                *,
                FIRST_VALUE(price) OVER (
                    PARTITION BY symbol, bar_time
                    ORDER BY event_time ASC
                ) AS open_price,
                FIRST_VALUE(price) OVER (
                    PARTITION BY symbol, bar_time
                    ORDER BY event_time DESC
                ) AS close_price
            FROM base
        )
        SELECT
            symbol,
            bar_time,
            MAX(open_price) AS open_price,
            MAX(price) AS high_price,
            MIN(price) AS low_price,
            MAX(close_price) AS close_price,
            SUM(quantity) AS volume,
            COUNT(*) AS trade_count
        FROM ranked
        GROUP BY symbol, bar_time
        ORDER BY symbol, bar_time;
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlc_1m_symbol_time
        ON ohlc_1m_bars(symbol, bar_time);
    """)

    conn.commit()

    cursor.execute("""
        SELECT COUNT(*)
        FROM ohlc_1m_bars;
    """)

    count = cursor.fetchone()[0]

    print("==========================")
    print("OHLC 1M BAR ENGINE")
    print("==========================")
    print(f"Bars created : {count}")
    print("Saved table  : ohlc_1m_bars")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


