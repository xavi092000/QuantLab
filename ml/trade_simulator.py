from configs.database import DB_CONFIG
import time
import psycopg2

POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_table(cursor):

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trade_simulation_results (

        id BIGSERIAL PRIMARY KEY,

        prediction_time TIMESTAMPTZ,

        recommendation TEXT,

        entry_price DOUBLE PRECISION,

        exit_price DOUBLE PRECISION,

        pnl_pct DOUBLE PRECISION,

        profitable_trade BOOLEAN,

        created_at TIMESTAMPTZ DEFAULT NOW()

    );
    """)


def fetch_predictions(cursor):

    cursor.execute("""
    SELECT
        prediction_time,
        recommendation
    FROM live_signal_success_predictions
    ORDER BY prediction_time ASC;
    """)

    return cursor.fetchall()


def fetch_prices(cursor, prediction_time):

    cursor.execute("""
    SELECT price
    FROM market_trades
    WHERE event_time >= %s
    ORDER BY event_time ASC
    LIMIT 1;
    """, (prediction_time,))

    entry_row = cursor.fetchone()

    cursor.execute("""
    SELECT price
    FROM market_trades
    WHERE event_time >= %s + INTERVAL '5 minutes'
    ORDER BY event_time ASC
    LIMIT 1;
    """, (prediction_time,))

    exit_row = cursor.fetchone()

    if not entry_row or not exit_row:
        return None, None

    return float(entry_row[0]), float(exit_row[0])


def insert_trade(
    cursor,
    prediction_time,
    recommendation,
    entry_price,
    exit_price,
    pnl_pct,
    profitable_trade
):

    cursor.execute("""
    INSERT INTO trade_simulation_results (

        prediction_time,
        recommendation,
        entry_price,
        exit_price,
        pnl_pct,
        profitable_trade

    )
    VALUES (%s,%s,%s,%s,%s,%s);
    """,
    (
        prediction_time,
        recommendation,
        entry_price,
        exit_price,
        pnl_pct,
        profitable_trade
    ))


def main():

    conn = get_connection()
    cursor = conn.cursor()

    create_table(cursor)
    conn.commit()

    print("[INFO] Trade Simulator Started")

    predictions = fetch_predictions(cursor)

    for prediction_time, recommendation in predictions:

        entry_price, exit_price = fetch_prices(
            cursor,
            prediction_time
        )

        if entry_price is None:
            continue

        pnl_pct = (
            (exit_price - entry_price)
            / entry_price
        ) * 100

        profitable_trade = pnl_pct > 0

        insert_trade(
            cursor,
            prediction_time,
            recommendation,
            entry_price,
            exit_price,
            pnl_pct,
            profitable_trade
        )

        print(
            f"[TRADE] "
            f"{recommendation} "
            f"PnL={pnl_pct:.4f}%"
        )

    conn.commit()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


