from configs.database import DB_CONFIG
import psycopg2

MIN_TRADES_REQUIRED = 5


def classify_strategy(trades, win_rate, total_pnl, avg_pnl):
    if trades < MIN_TRADES_REQUIRED:
        return "INSUFFICIENT_DATA"

    if total_pnl > 0 and win_rate >= 50 and avg_pnl > 0:
        return "FAVOR"

    if total_pnl < 0 and win_rate < 40:
        return "AVOID"

    return "NEUTRAL"


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta_strategy_scores (
            strategy_name TEXT PRIMARY KEY,
            trades INTEGER,
            win_rate DOUBLE PRECISION,
            total_pnl DOUBLE PRECISION,
            avg_pnl DOUBLE PRECISION,
            meta_decision TEXT,
            confidence_score DOUBLE PRECISION,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("DELETE FROM meta_strategy_scores;")

    cursor.execute("""
        SELECT
            strategy_name,
            trades,
            win_rate,
            total_pnl,
            avg_pnl
        FROM strategy_performance;
    """)

    rows = cursor.fetchall()

    for row in rows:
        strategy_name = row[0]
        trades = int(row[1])
        win_rate = float(row[2])
        total_pnl = float(row[3])
        avg_pnl = float(row[4])

        meta_decision = classify_strategy(
            trades,
            win_rate,
            total_pnl,
            avg_pnl
        )

        confidence_score = 0.0

        if meta_decision == "FAVOR":
            confidence_score = min(100, win_rate + max(avg_pnl, 0) / 10)

        elif meta_decision == "AVOID":
            confidence_score = max(0, 100 - abs(total_pnl) / 100)

        elif meta_decision == "NEUTRAL":
            confidence_score = 50.0

        else:
            confidence_score = 0.0

        cursor.execute("""
            INSERT INTO meta_strategy_scores (
                strategy_name,
                trades,
                win_rate,
                total_pnl,
                avg_pnl,
                meta_decision,
                confidence_score
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            strategy_name,
            trades,
            win_rate,
            total_pnl,
            avg_pnl,
            meta_decision,
            confidence_score,
        ))

    conn.commit()

    print("==============================")
    print("META STRATEGY ENGINE")
    print("==============================")

    cursor.execute("""
        SELECT
            strategy_name,
            trades,
            win_rate,
            total_pnl,
            avg_pnl,
            meta_decision,
            confidence_score
        FROM meta_strategy_scores
        ORDER BY confidence_score DESC;
    """)

    for r in cursor.fetchall():
        print(
            f"{r[0]} | trades={r[1]} | win_rate={r[2]:.2f}% | "
            f"total_pnl=${r[3]:.2f} | avg_pnl=${r[4]:.2f} | "
            f"decision={r[5]} | confidence={r[6]:.2f}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


