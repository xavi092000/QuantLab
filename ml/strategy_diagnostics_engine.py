from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS strategy_diagnostics;
""")

cursor.execute("""
CREATE TABLE strategy_diagnostics (
    id BIGSERIAL PRIMARY KEY,
    diagnostic_name TEXT,
    diagnostic_value DOUBLE PRECISION,
    diagnostic_comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

# Win Rate
cursor.execute("""
SELECT
100.0 *
SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)
/
COUNT(*)
FROM trade_simulation_results;
""")

win_rate = float(cursor.fetchone()[0])

# Avg Win
cursor.execute("""
SELECT AVG(pnl_pct)
FROM trade_simulation_results
WHERE pnl_pct > 0;
""")

avg_win = float(cursor.fetchone()[0])

# Avg Loss
cursor.execute("""
SELECT ABS(AVG(pnl_pct))
FROM trade_simulation_results
WHERE pnl_pct < 0;
""")

avg_loss = float(cursor.fetchone()[0])

# Best Trade
cursor.execute("""
SELECT MAX(pnl_pct)
FROM trade_simulation_results;
""")

best_trade = float(cursor.fetchone()[0])

# Worst Trade
cursor.execute("""
SELECT MIN(pnl_pct)
FROM trade_simulation_results;
""")

worst_trade = float(cursor.fetchone()[0])

# Profit Factor
cursor.execute("""
SELECT
COALESCE(SUM(CASE WHEN pnl_pct > 0 THEN pnl_pct END),0)
/
NULLIF(
ABS(
SUM(CASE WHEN pnl_pct < 0 THEN pnl_pct END)
),
0
)
FROM trade_simulation_results;
""")

profit_factor = float(cursor.fetchone()[0])

diagnostics = [

(
"WIN_RATE",
win_rate,
"Above 50% is generally desirable"
),

(
"AVG_WIN",
avg_win,
"Average winning trade"
),

(
"AVG_LOSS",
avg_loss,
"Average losing trade"
),

(
"BEST_TRADE",
best_trade,
"Best trade observed"
),

(
"WORST_TRADE",
worst_trade,
"Worst trade observed"
),

(
"PROFIT_FACTOR",
profit_factor,
"Above 1.0 indicates positive expectancy"
)

]

for row in diagnostics:

    cursor.execute("""
    INSERT INTO strategy_diagnostics (
        diagnostic_name,
        diagnostic_value,
        diagnostic_comment
    )
    VALUES (%s,%s,%s)
    """, row)

conn.commit()

print("==============================")
print("STRATEGY DIAGNOSTICS COMPLETE")
print("==============================")

for row in diagnostics:
    print(row[0], "=", round(row[1],4))

cursor.close()
conn.close()


