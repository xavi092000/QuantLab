from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    qm.market_regime AS market_regime,
    sv.future_return_5m AS future_return_5m
FROM signal_validation sv
JOIN quant_metrics qm
    ON sv.symbol = qm.symbol
   AND sv.signal_time = qm.metric_time
WHERE sv.future_return_5m IS NOT NULL
  AND qm.market_regime IS NOT NULL;
"""

df = pd.read_sql(query, conn)

summary = (
    df.groupby("market_regime")
      .agg(
        trades=("future_return_5m", "count"),
        avg_return=("future_return_5m", "mean"),
        total_return=("future_return_5m", "sum")
      )
      .reset_index()
)

summary["recommendation"] = summary["avg_return"].apply(
    lambda x: "TRADE" if x > 0 else "BLOCK"
)

cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS market_regime_filter;")

cursor.execute("""
CREATE TABLE market_regime_filter (
    market_regime TEXT,
    trades INTEGER,
    avg_return DOUBLE PRECISION,
    total_return DOUBLE PRECISION,
    recommendation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

for _, row in summary.iterrows():
    cursor.execute("""
    INSERT INTO market_regime_filter (
        market_regime,
        trades,
        avg_return,
        total_return,
        recommendation
    )
    VALUES (%s, %s, %s, %s, %s);
    """, (
        row["market_regime"],
        int(row["trades"]),
        float(row["avg_return"]),
        float(row["total_return"]),
        row["recommendation"]
    ))

conn.commit()

print("========================")
print("MARKET REGIME FILTER")
print("========================")
print(summary)

cursor.close()
conn.close()


