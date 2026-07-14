from configs.database import DB_CONFIG
import psycopg2

MIN_PROFITABLE_WINDOW_RATIO = 0.55
MIN_TOTAL_RETURN = 0.0
MAX_RESEARCH_RISK_PCT = 1.0


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS risk_budgeting_v2;")

    cursor.execute("""
        CREATE TABLE risk_budgeting_v2 (
            id BIGSERIAL PRIMARY KEY,
            strategy_name TEXT,
            strategy_status TEXT,
            profitable_windows INTEGER,
            windows_tested INTEGER,
            profitable_window_ratio DOUBLE PRECISION,
            total_return_pct DOUBLE PRECISION,
            return_threshold DOUBLE PRECISION,
            approved_risk_pct DOUBLE PRECISION,
            risk_decision TEXT,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            strategy_name,
            strategy_status,
            profitable_windows,
            windows_tested,
            total_return_pct,
            return_threshold
        FROM production_strategy_config
        ORDER BY created_at DESC
        LIMIT 1;
    """)

    row = cursor.fetchone()

    if row is None:
        print("[ERROR] No production strategy config found.")
        conn.close()
        return

    strategy_name = row[0]
    strategy_status = row[1]
    profitable_windows = int(row[2])
    windows_tested = int(row[3])
    total_return_pct = float(row[4])
    return_threshold = float(row[5])

    profitable_window_ratio = profitable_windows / windows_tested

    if (
        strategy_status == "RESEARCH_APPROVED"
        and profitable_window_ratio >= MIN_PROFITABLE_WINDOW_RATIO
        and total_return_pct > MIN_TOTAL_RETURN
    ):
        approved_risk_pct = MAX_RESEARCH_RISK_PCT
        risk_decision = "RESEARCH_PAPER_RISK_APPROVED"
        notes = "Research-only paper risk approved based on walk-forward threshold sweep."
    else:
        approved_risk_pct = 0.0
        risk_decision = "NO_RISK_ALLOCATED"
        notes = "Strategy did not meet research risk approval conditions."

    cursor.execute("""
        INSERT INTO risk_budgeting_v2 (
            strategy_name,
            strategy_status,
            profitable_windows,
            windows_tested,
            profitable_window_ratio,
            total_return_pct,
            return_threshold,
            approved_risk_pct,
            risk_decision,
            notes
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, (
        strategy_name,
        strategy_status,
        profitable_windows,
        windows_tested,
        profitable_window_ratio,
        total_return_pct,
        return_threshold,
        approved_risk_pct,
        risk_decision,
        notes,
    ))

    conn.commit()

    print("==============================")
    print("RISK BUDGETING V2 COMPLETE")
    print("==============================")
    print("Strategy :", strategy_name)
    print("Window Ratio :", round(profitable_window_ratio, 4))
    print("Total Return :", round(total_return_pct, 4))
    print("Approved Risk % :", approved_risk_pct)
    print("Decision :", risk_decision)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


