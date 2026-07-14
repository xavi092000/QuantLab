from configs.database import DB_CONFIG
import psycopg2

def classify_trade(pnl_pct):
    pnl = float(pnl_pct)

    if pnl >= 0.15:
        return "TOP_WINNER"

    if pnl > 0:
        return "SMALL_WINNER"

    if pnl <= -0.15:
        return "WORST_LOSER"

    return "SMALL_LOSER"


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS trade_quality_engine;")

    cursor.execute("""
        CREATE TABLE trade_quality_engine (
            id BIGSERIAL PRIMARY KEY,
            prediction_time TIMESTAMPTZ,
            recommendation TEXT,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            pnl_pct DOUBLE PRECISION,
            trade_quality TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            prediction_time,
            recommendation,
            entry_price,
            exit_price,
            pnl_pct
        FROM trade_simulation_results
        ORDER BY prediction_time ASC;
    """)

    rows = cursor.fetchall()

    for row in rows:
        prediction_time, recommendation, entry_price, exit_price, pnl_pct = row

        trade_quality = classify_trade(pnl_pct)

        cursor.execute("""
            INSERT INTO trade_quality_engine (
                prediction_time,
                recommendation,
                entry_price,
                exit_price,
                pnl_pct,
                trade_quality
            )
            VALUES (%s,%s,%s,%s,%s,%s);
        """, (
            prediction_time,
            recommendation,
            entry_price,
            exit_price,
            float(pnl_pct),
            trade_quality,
        ))

    conn.commit()

    print("==============================")
    print("TRADE QUALITY ENGINE COMPLETE")
    print("==============================")
    print(f"Trades processed : {len(rows)}")
    print("Saved table      : trade_quality_engine")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


