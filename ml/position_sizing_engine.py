from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            portfolio_value_usd,
            max_allocation_pct
        FROM portfolio_config
        ORDER BY created_at DESC
        LIMIT 1;
    """)

    config = cursor.fetchone()

    if config is None:
        print("[ERROR] No portfolio_config found.")
        cursor.close()
        conn.close()
        return

    portfolio_value = float(config[0])
    max_allocation_pct = float(config[1])

    cursor.execute("""
        DROP TABLE IF EXISTS position_sizing_results;
    """)

    cursor.execute("""
        CREATE TABLE position_sizing_results (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            opportunity_score DOUBLE PRECISION,
            recommendation TEXT,
            combined_decision TEXT,
            allocation_pct DOUBLE PRECISION,
            portfolio_value_usd DOUBLE PRECISION,
            position_size_usd DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            symbol,
            opportunity_score,
            recommendation,
            combined_decision
        FROM opportunity_ranking_engine
        ORDER BY opportunity_score DESC;
    """)

    rows = cursor.fetchall()

    inserted = 0

    for row in rows:
        symbol = row[0]
        opportunity_score = float(row[1])
        recommendation = row[2]
        combined_decision = row[3]

        allocation_pct = 0.0

        if recommendation == "BUY" and combined_decision == "BUY":
            allocation_pct = opportunity_score / 10.0
            allocation_pct = min(allocation_pct, max_allocation_pct)

        position_size_usd = portfolio_value * (allocation_pct / 100.0)

        cursor.execute("""
            INSERT INTO position_sizing_results (
                symbol,
                opportunity_score,
                recommendation,
                combined_decision,
                allocation_pct,
                portfolio_value_usd,
                position_size_usd
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            opportunity_score,
            recommendation,
            combined_decision,
            allocation_pct,
            portfolio_value,
            position_size_usd,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("POSITION SIZING ENGINE")
    print("==============================")
    print("Portfolio Value :", portfolio_value)
    print("Max Allocation %:", max_allocation_pct)
    print("Assets Processed:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            ROUND(opportunity_score::numeric, 2),
            recommendation,
            combined_decision,
            ROUND(allocation_pct::numeric, 2),
            ROUND(position_size_usd::numeric, 2)
        FROM position_sizing_results
        ORDER BY opportunity_score DESC;
    """)

    results = cursor.fetchall()

    for r in results:
        print(
            f"{r[0]} | "
            f"score={r[1]} | "
            f"recommendation={r[2]} | "
            f"combined={r[3]} | "
            f"allocation={r[4]}% | "
            f"position=${r[5]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


