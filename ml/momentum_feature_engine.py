from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS momentum_features;
""")

cursor.execute("""
CREATE TABLE momentum_features AS

WITH price_series AS (

    SELECT
        id,
        symbol,
        event_time,
        price,

        LAG(price, 5)
            OVER (
                PARTITION BY symbol
                ORDER BY event_time
            ) AS price_5,

        LAG(price, 15)
            OVER (
                PARTITION BY symbol
                ORDER BY event_time
            ) AS price_15,

        LAG(price, 30)
            OVER (
                PARTITION BY symbol
                ORDER BY event_time
            ) AS price_30,

        LAG(price, 60)
            OVER (
                PARTITION BY symbol
                ORDER BY event_time
            ) AS price_60

    FROM market_trades

)

SELECT

    id,
    symbol,
    event_time,
    price,

    ROUND(
        (
            (price - price_5)
            /
            NULLIF(price_5,0)
        )::numeric,
        6
    ) AS momentum_5,

    ROUND(
        (
            (price - price_15)
            /
            NULLIF(price_15,0)
        )::numeric,
        6
    ) AS momentum_15,

    ROUND(
        (
            (price - price_30)
            /
            NULLIF(price_30,0)
        )::numeric,
        6
    ) AS momentum_30,

    ROUND(
        (
            (price - price_60)
            /
            NULLIF(price_60,0)
        )::numeric,
        6
    ) AS momentum_60

FROM price_series;
""")

conn.commit()

cursor.execute("""
SELECT COUNT(*)
FROM momentum_features;
""")

count = cursor.fetchone()[0]

print("==========================")
print("MOMENTUM FEATURE ENGINE")
print("==========================")
print("Rows created :", count)

cursor.close()
conn.close()


