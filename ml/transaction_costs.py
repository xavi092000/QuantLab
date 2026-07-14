from configs.database import DB_CONFIG
import psycopg2

TRANSACTION_COST_PCT = 0.10

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS transaction_cost_analysis;
""")

cursor.execute("""
CREATE TABLE transaction_cost_analysis (
    id BIGSERIAL PRIMARY KEY,
    prediction_time TIMESTAMPTZ,
    gross_return_pct DOUBLE PRECISION,
    transaction_cost_pct DOUBLE PRECISION,
    net_return_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
SELECT
    prediction_time,
    pnl_pct
FROM trade_simulation_results
ORDER BY prediction_time;
""")

rows = cursor.fetchall()

for row in rows:

    prediction_time = row[0]
    gross_return = float(row[1])

    net_return = gross_return - TRANSACTION_COST_PCT

    cursor.execute("""
    INSERT INTO transaction_cost_analysis (
        prediction_time,
        gross_return_pct,
        transaction_cost_pct,
        net_return_pct
    )
    VALUES (%s,%s,%s,%s)
    """,
    (
        prediction_time,
        gross_return,
        TRANSACTION_COST_PCT,
        net_return
    ))

conn.commit()

print("==========================")
print("TRANSACTION COST ANALYSIS")
print("==========================")
print(f"Trades processed : {len(rows)}")
print(f"Cost per trade   : {TRANSACTION_COST_PCT}%")

cursor.close()
conn.close()


