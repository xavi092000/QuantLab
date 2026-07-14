from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

INITIAL_CAPITAL = 10000.0


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def main():
    conn = get_connection()

    query = """
        SELECT
            prediction_time,
            pnl_pct
        FROM trade_simulation_results
        ORDER BY prediction_time;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No trades found in trade_simulation_results.")
        conn.close()
        return

    capital = INITIAL_CAPITAL
    equity_values = []

    for pnl_pct in df["pnl_pct"]:
        capital = capital * (1 + float(pnl_pct) / 100)
        equity_values.append(capital)

    df["equity_value"] = equity_values

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS equity_curve;")

    cursor.execute("""
        CREATE TABLE equity_curve (
            id BIGSERIAL PRIMARY KEY,
            prediction_time TIMESTAMPTZ,
            pnl_pct DOUBLE PRECISION,
            equity_value DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO equity_curve (
                prediction_time,
                pnl_pct,
                equity_value
            )
            VALUES (%s, %s, %s);
        """, (
            row["prediction_time"],
            float(row["pnl_pct"]),
            float(row["equity_value"]),
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print("================================")
    print("EQUITY CURVE GENERATED")
    print("================================")
    print(f"Initial Capital : {INITIAL_CAPITAL:.2f}")
    print(f"Final Capital   : {capital:.2f}")
    print(f"Total Trades    : {len(df)}")
    print("Saved to table  : equity_curve")


if __name__ == "__main__":
    main()


