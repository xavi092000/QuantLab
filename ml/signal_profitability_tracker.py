from configs.database import DB_CONFIG
import psycopg2

HORIZON_MINUTES = 5


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_profitability_tracking (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            signal_time TIMESTAMPTZ,
            decision TEXT,
            predicted_return_5m DOUBLE PRECISION,
            probability_up DOUBLE PRECISION,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            actual_return_5m DOUBLE PRECISION,
            profitable BOOLEAN,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            symbol,
            created_at AS signal_time,
            combined_decision,
            predicted_return_5m,
            probability_up
        FROM combined_signal_engine
        WHERE combined_decision = 'BUY';
    """)

    signals = cursor.fetchall()

    if not signals:
        print("==============================")
        print("SIGNAL PROFITABILITY TRACKER")
        print("==============================")
        print("No BUY signals found.")
        print("Nothing to validate yet.")
        cursor.close()
        conn.close()
        return

    inserted = 0

    for signal in signals:
        symbol = signal[0]
        signal_time = signal[1]
        decision = signal[2]
        predicted_return = float(signal[3])
        probability_up = float(signal[4])

        cursor.execute("""
            SELECT price
            FROM market_trades
            WHERE symbol = %s
              AND event_time <= %s
            ORDER BY event_time DESC
            LIMIT 1;
        """, (symbol, signal_time))

        entry_row = cursor.fetchone()

        cursor.execute("""
            SELECT price
            FROM market_trades
            WHERE symbol = %s
              AND event_time >= %s + INTERVAL '5 minutes'
            ORDER BY event_time ASC
            LIMIT 1;
        """, (symbol, signal_time))

        exit_row = cursor.fetchone()

        if entry_row is None or exit_row is None:
            continue

        entry_price = float(entry_row[0])
        exit_price = float(exit_row[0])

        actual_return = (exit_price - entry_price) / entry_price
        profitable = actual_return > 0

        cursor.execute("""
            INSERT INTO signal_profitability_tracking (
                symbol,
                signal_time,
                decision,
                predicted_return_5m,
                probability_up,
                entry_price,
                exit_price,
                actual_return_5m,
                profitable
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            signal_time,
            decision,
            predicted_return,
            probability_up,
            entry_price,
            exit_price,
            actual_return,
            profitable,
        ))

        inserted += 1

    conn.commit()

    cursor.execute("""
        SELECT
            COUNT(*),
            AVG(CASE WHEN profitable THEN 1 ELSE 0 END) * 100,
            AVG(actual_return_5m),
            SUM(actual_return_5m)
        FROM signal_profitability_tracking;
    """)

    summary = cursor.fetchone()

    print("==============================")
    print("SIGNAL PROFITABILITY TRACKER")
    print("==============================")
    print("Rows inserted       :", inserted)
    print("Total BUY signals   :", summary[0])
    print("Win Rate %          :", round(float(summary[1] or 0), 2))
    print("Avg Return 5m       :", round(float(summary[2] or 0), 6))
    print("Total Sim Return    :", round(float(summary[3] or 0), 6))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


