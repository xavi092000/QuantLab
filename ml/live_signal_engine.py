from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS live_signal_engine;")

    cursor.execute("""
        CREATE TABLE live_signal_engine (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            asset_rank INTEGER,
            ml_probability_pct DOUBLE PRECISION,
            asset_recommendation TEXT,
            approved_risk_pct DOUBLE PRECISION,
            final_decision TEXT,
            decision_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            approved_risk_pct,
            risk_decision
        FROM risk_budgeting_engine
        ORDER BY created_at DESC
        LIMIT 1;
    """)

    risk_row = cursor.fetchone()

    if risk_row is None:
        print("[ERROR] No risk budgeting result found.")
        cursor.close()
        conn.close()
        return

    approved_risk_pct = float(risk_row[0])
    risk_decision = risk_row[1]

    cursor.execute("""
        SELECT
            symbol,
            asset_rank,
            ml_probability_pct,
            recommendation
        FROM cross_asset_ml_rankings
        ORDER BY asset_rank ASC;
    """)

    assets = cursor.fetchall()

    inserted = 0

    for symbol, asset_rank, ml_probability_pct, recommendation in assets:
        probability = float(ml_probability_pct)

        if approved_risk_pct <= 0:
            final_decision = "NO_TRADE"
            decision_reason = "Risk budget blocks trading"

        elif recommendation == "STRONG_BUY" and probability >= 70:
            final_decision = "BUY"
            decision_reason = "Top-ranked asset with strong probability"

        elif recommendation == "WATCHLIST" and probability >= 55:
            final_decision = "WATCH"
            decision_reason = "Asset is on watchlist but below buy threshold"

        else:
            final_decision = "NO_TRADE"
            decision_reason = "Signal does not meet decision threshold"

        cursor.execute("""
            INSERT INTO live_signal_engine (
                symbol,
                asset_rank,
                ml_probability_pct,
                asset_recommendation,
                approved_risk_pct,
                final_decision,
                decision_reason
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            asset_rank,
            probability,
            recommendation,
            approved_risk_pct,
            final_decision,
            decision_reason,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("LIVE SIGNAL ENGINE COMPLETE")
    print("==============================")
    print(f"Assets processed : {inserted}")
    print(f"Risk decision    : {risk_decision}")
    print(f"Approved risk %  : {approved_risk_pct}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


