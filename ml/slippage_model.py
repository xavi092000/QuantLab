from configs.database import DB_CONFIG
import psycopg2

BASE_SLIPPAGE_PCT = 0.03


def calculate_slippage_pct(recommendation, gross_return_pct):
    abs_move = abs(float(gross_return_pct))

    slippage = BASE_SLIPPAGE_PCT

    if recommendation == "TAKE":
        slippage += 0.02

    if abs_move > 0.10:
        slippage += 0.03

    if abs_move > 0.20:
        slippage += 0.05

    return slippage


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS slippage_analysis;")

    cursor.execute("""
        CREATE TABLE slippage_analysis (
            id BIGSERIAL PRIMARY KEY,
            prediction_time TIMESTAMPTZ,
            recommendation TEXT,
            gross_return_pct DOUBLE PRECISION,
            slippage_pct DOUBLE PRECISION,
            net_after_slippage_pct DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            prediction_time,
            recommendation,
            pnl_pct
        FROM trade_simulation_results
        ORDER BY prediction_time;
    """)

    rows = cursor.fetchall()

    inserted = 0

    for prediction_time, recommendation, gross_return_pct in rows:
        gross_return = float(gross_return_pct)
        slippage_pct = calculate_slippage_pct(recommendation, gross_return)
        net_after_slippage = gross_return - slippage_pct

        cursor.execute("""
            INSERT INTO slippage_analysis (
                prediction_time,
                recommendation,
                gross_return_pct,
                slippage_pct,
                net_after_slippage_pct
            )
            VALUES (%s, %s, %s, %s, %s);
        """, (
            prediction_time,
            recommendation,
            gross_return,
            slippage_pct,
            net_after_slippage,
        ))

        inserted += 1

    conn.commit()

    print("==========================")
    print("SLIPPAGE MODEL COMPLETE")
    print("==========================")
    print(f"Trades processed : {inserted}")
    print(f"Base slippage    : {BASE_SLIPPAGE_PCT}%")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


