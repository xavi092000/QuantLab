from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_accuracy_tracking (
            id BIGSERIAL PRIMARY KEY,
            signal_id BIGINT,
            symbol TEXT,
            signal_time TIMESTAMPTZ,
            predicted_return_5m DOUBLE PRECISION,
            actual_return_5m DOUBLE PRECISION,
            research_signal TEXT,
            prediction_correct BOOLEAN,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        INSERT INTO signal_accuracy_tracking (
            signal_id,
            symbol,
            signal_time,
            predicted_return_5m,
            actual_return_5m,
            research_signal,
            prediction_correct
        )
        SELECT
            h.id AS signal_id,
            h.symbol,
            h.created_at AS signal_time,
            h.predicted_return_5m,
            (
                future.price - current_price.price
            ) / NULLIF(current_price.price, 0) AS actual_return_5m,
            h.research_signal,
            CASE
                WHEN h.predicted_return_5m > 0
                 AND (
                    future.price - current_price.price
                 ) / NULLIF(current_price.price, 0) > 0
                THEN TRUE

                WHEN h.predicted_return_5m <= 0
                 AND (
                    future.price - current_price.price
                 ) / NULLIF(current_price.price, 0) <= 0
                THEN TRUE

                ELSE FALSE
            END AS prediction_correct
        FROM live_signal_history h

        JOIN LATERAL (
            SELECT price
            FROM market_trades mt
            WHERE mt.symbol = h.symbol
              AND mt.event_time <= h.created_at
            ORDER BY mt.event_time DESC
            LIMIT 1
        ) current_price ON TRUE

        JOIN LATERAL (
            SELECT price
            FROM market_trades mt
            WHERE mt.symbol = h.symbol
              AND mt.event_time >= h.created_at + INTERVAL '5 minutes'
            ORDER BY mt.event_time ASC
            LIMIT 1
        ) future ON TRUE

        WHERE NOT EXISTS (
            SELECT 1
            FROM signal_accuracy_tracking sat
            WHERE sat.signal_id = h.id
        );
    """)

    conn.commit()

    cursor.execute("""
        SELECT
            COUNT(*) AS validated_signals,
            ROUND(
                100.0 * AVG(
                    CASE WHEN prediction_correct THEN 1 ELSE 0 END
                )::numeric,
                2
            ) AS live_accuracy_pct,
            ROUND(AVG(actual_return_5m)::numeric, 6) AS avg_actual_return
        FROM signal_accuracy_tracking;
    """)

    result = cursor.fetchone()

    print("==============================")
    print("SIGNAL ACCURACY TRACKER")
    print("==============================")
    print("Validated Signals :", result[0])
    print("Live Accuracy %   :", result[1])
    print("Avg Actual Return :", result[2])

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


