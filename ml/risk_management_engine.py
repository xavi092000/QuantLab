from configs.database import DB_CONFIG
import psycopg2

STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 4.0


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS risk_management_results;")

    cursor.execute("""
        CREATE TABLE risk_management_results (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            recommendation TEXT,
            combined_decision TEXT,
            opportunity_score DOUBLE PRECISION,
            allocation_pct DOUBLE PRECISION,
            position_size_usd DOUBLE PRECISION,
            stop_loss_pct DOUBLE PRECISION,
            take_profit_pct DOUBLE PRECISION,
            max_loss_usd DOUBLE PRECISION,
            target_profit_usd DOUBLE PRECISION,
            risk_reward_ratio DOUBLE PRECISION,
            risk_score TEXT,
            final_trade_plan TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            symbol,
            recommendation,
            combined_decision,
            opportunity_score,
            allocation_pct,
            position_size_usd
        FROM position_sizing_results
        ORDER BY opportunity_score DESC;
    """)

    rows = cursor.fetchall()
    inserted = 0

    for row in rows:
        symbol = row[0]
        recommendation = row[1]
        combined_decision = row[2]
        opportunity_score = float(row[3])
        allocation_pct = float(row[4])
        position_size_usd = float(row[5])

        if position_size_usd > 0:
            stop_loss_pct = STOP_LOSS_PCT
            take_profit_pct = TAKE_PROFIT_PCT
            max_loss_usd = position_size_usd * (stop_loss_pct / 100)
            target_profit_usd = position_size_usd * (take_profit_pct / 100)
            risk_reward_ratio = take_profit_pct / stop_loss_pct

            if allocation_pct <= 3:
                risk_score = "LOW"
            elif allocation_pct <= 7:
                risk_score = "MEDIUM"
            else:
                risk_score = "HIGH"

            final_trade_plan = "TRADE_ALLOWED"

        else:
            stop_loss_pct = 0.0
            take_profit_pct = 0.0
            max_loss_usd = 0.0
            target_profit_usd = 0.0
            risk_reward_ratio = 0.0
            risk_score = "NO_POSITION"
            final_trade_plan = "NO_TRADE"

        cursor.execute("""
            INSERT INTO risk_management_results (
                symbol,
                recommendation,
                combined_decision,
                opportunity_score,
                allocation_pct,
                position_size_usd,
                stop_loss_pct,
                take_profit_pct,
                max_loss_usd,
                target_profit_usd,
                risk_reward_ratio,
                risk_score,
                final_trade_plan
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            recommendation,
            combined_decision,
            opportunity_score,
            allocation_pct,
            position_size_usd,
            stop_loss_pct,
            take_profit_pct,
            max_loss_usd,
            target_profit_usd,
            risk_reward_ratio,
            risk_score,
            final_trade_plan,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("RISK MANAGEMENT ENGINE")
    print("==============================")
    print("Assets Processed:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            ROUND(opportunity_score::numeric, 2),
            recommendation,
            combined_decision,
            ROUND(allocation_pct::numeric, 2),
            ROUND(position_size_usd::numeric, 2),
            stop_loss_pct,
            take_profit_pct,
            ROUND(max_loss_usd::numeric, 2),
            ROUND(target_profit_usd::numeric, 2),
            risk_reward_ratio,
            risk_score,
            final_trade_plan
        FROM risk_management_results
        ORDER BY opportunity_score DESC;
    """)

    results = cursor.fetchall()

    for r in results:
        print(
            f"{r[0]} | "
            f"score={r[1]} | "
            f"rec={r[2]} | "
            f"combined={r[3]} | "
            f"alloc={r[4]}% | "
            f"position=${r[5]} | "
            f"SL={r[6]}% | "
            f"TP={r[7]}% | "
            f"max_loss=${r[8]} | "
            f"target=${r[9]} | "
            f"RR={r[10]} | "
            f"risk={r[11]} | "
            f"plan={r[12]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


