from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    symbol,
    recommendation,
    success_probability
FROM live_signal_success_predictions
WHERE recommendation = 'TAKE'
"""

df = pd.read_sql(query, conn)

if len(df) == 0:
    print("No TAKE signals found.")
    quit()

df["score"] = df["success_probability"] * 100

total_score = df["score"].sum()

df["allocation_pct"] = (
    df["score"] / total_score
) * 100

df = df.sort_values(
    "allocation_pct",
    ascending=False
)

cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS portfolio_optimization;
""")

cursor.execute("""
CREATE TABLE portfolio_optimization (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT,
    score DOUBLE PRECISION,
    allocation_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

for _, row in df.iterrows():

    cursor.execute("""
    INSERT INTO portfolio_optimization (
        symbol,
        score,
        allocation_pct
    )
    VALUES (%s,%s,%s)
    """,
    (
        row["symbol"],
        float(row["score"]),
        float(row["allocation_pct"])
    ))

conn.commit()

print("==========================")
print("PORTFOLIO OPTIMIZATION")
print("==========================")
print(f"Signals : {len(df)}")
print(f"Total Allocation : {df['allocation_pct'].sum():.2f}%")

cursor.close()
conn.close()


