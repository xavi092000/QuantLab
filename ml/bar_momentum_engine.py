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
DROP TABLE IF EXISTS bar_momentum_features;
""")

cursor.execute("""
CREATE TABLE bar_momentum_features AS

SELECT

symbol,
bar_time,
close_price,

ROUND(
(
close_price
/
LAG(close_price,5)
OVER(
PARTITION BY symbol
ORDER BY bar_time
)
- 1
)::numeric,
6
) AS momentum_5m,

ROUND(
(
close_price
/
LAG(close_price,15)
OVER(
PARTITION BY symbol
ORDER BY bar_time
)
- 1
)::numeric,
6
) AS momentum_15m,

ROUND(
(
close_price
/
LAG(close_price,30)
OVER(
PARTITION BY symbol
ORDER BY bar_time
)
- 1
)::numeric,
6
) AS momentum_30m

FROM ohlc_1m_bars;
""")

conn.commit()

cursor.execute("""
SELECT COUNT(*)
FROM bar_momentum_features;
""")

count = cursor.fetchone()[0]

print("==========================")
print("BAR MOMENTUM ENGINE")
print("==========================")
print("Rows :", count)

cursor.close()
conn.close()




