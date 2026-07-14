from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(
    host=DB_CONFIG["host"],
    database="quantlab",
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    port="5432"
)

cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS worst_trade_analysis;
""")

cursor.execute("""
CREATE TABLE worst_trade_analysis AS
SELECT *
FROM trade_quality_engine
WHERE trade_quality = 'WORST_LOSER'
ORDER BY pnl_pct ASC;
""")

conn.commit()

cursor.execute("""
SELECT
COUNT(*) ,
AVG(pnl_pct),
MIN(pnl_pct),
MAX(pnl_pct)
FROM worst_trade_analysis;
""")

result = cursor.fetchone()

print("==========================")
print("WORST TRADE ANALYSIS")
print("==========================")
print("Trades :", result[0])
print("Avg Loss :", round(result[1],4))
print("Worst :", round(result[2],4))
print("Best :", round(result[3],4))

cursor.close()
conn.close()




