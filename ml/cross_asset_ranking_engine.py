from configs.database import DB_CONFIG
import psycopg2

MIN_TRADES_REQUIRED = 50


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS cross_asset_rankings;")

    cursor.execute("""
        CREATE TABLE cross_asset_rankings (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            trade_count BIGINT,
            avg_price DOUBLE PRECISION,
            price_min DOUBLE PRECISION,
            price_max DOUBLE PRECISION,
            price_range_pct DOUBLE PRECISION,
            total_quantity DOUBLE PRECISION,
            activity_score DOUBLE PRECISION,
            liquidity_score DOUBLE PRECISION,
            volatility_score DOUBLE PRECISION,
            cross_asset_score DOUBLE PRECISION,
            asset_rank INTEGER,
            recommendation TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        WITH asset_stats AS (
            SELECT
                symbol,
                COUNT(*) AS trade_count,
                AVG(price) AS avg_price,
                MIN(price) AS price_min,
                MAX(price) AS price_max,
                SUM(quantity) AS total_quantity,
                CASE
                    WHEN AVG(price) = 0 THEN 0
                    ELSE ((MAX(price) - MIN(price)) / AVG(price)) * 100
                END AS price_range_pct
            FROM market_trades
            GROUP BY symbol
            HAVING COUNT(*) >= %s
        ),
        normalized AS (
            SELECT
                symbol,
                trade_count,
                avg_price,
                price_min,
                price_max,
                price_range_pct,
                total_quantity,

                CASE
                    WHEN MAX(trade_count) OVER () = 0 THEN 0
                    ELSE trade_count::DOUBLE PRECISION / MAX(trade_count) OVER ()
                END AS activity_score,

                CASE
                    WHEN MAX(total_quantity) OVER () = 0 THEN 0
                    ELSE total_quantity / MAX(total_quantity) OVER ()
                END AS liquidity_score,

                CASE
                    WHEN MAX(price_range_pct) OVER () = 0 THEN 0
                    ELSE price_range_pct / MAX(price_range_pct) OVER ()
                END AS volatility_score

            FROM asset_stats
        ),
        scored AS (
            SELECT
                *,
                (
                    activity_score * 0.35
                    + liquidity_score * 0.35
                    + volatility_score * 0.30
                ) * 100 AS cross_asset_score
            FROM normalized
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    ORDER BY cross_asset_score DESC
                ) AS asset_rank
            FROM scored
        )
        SELECT
            symbol,
            trade_count,
            avg_price,
            price_min,
            price_max,
            price_range_pct,
            total_quantity,
            activity_score,
            liquidity_score,
            volatility_score,
            cross_asset_score,
            asset_rank,
            CASE
                WHEN asset_rank = 1 THEN 'TOP_ASSET'
                WHEN asset_rank <= 3 THEN 'WATCHLIST'
                ELSE 'LOW_PRIORITY'
            END AS recommendation
        FROM ranked
        ORDER BY asset_rank ASC;
    """, (MIN_TRADES_REQUIRED,))

    rows = cursor.fetchall()

    for row in rows:
        cursor.execute("""
            INSERT INTO cross_asset_rankings (
                symbol,
                trade_count,
                avg_price,
                price_min,
                price_max,
                price_range_pct,
                total_quantity,
                activity_score,
                liquidity_score,
                volatility_score,
                cross_asset_score,
                asset_rank,
                recommendation
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, row)

    conn.commit()

    print("==============================")
    print("CROSS-ASSET RANKING COMPLETE")
    print("==============================")
    print(f"Assets ranked : {len(rows)}")
    print(f"Minimum trades required : {MIN_TRADES_REQUIRED}")
    print("Saved table : cross_asset_rankings")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


