from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regime_performance (
            market_regime TEXT PRIMARY KEY,
            trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate DOUBLE PRECISION,
            total_pnl DOUBLE PRECISION,
            avg_pnl DOUBLE PRECISION,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            COALESCE(market_regime, 'UNKNOWN') AS market_regime,
            pnl_usd
        FROM closed_paper_trades;
    """)

    rows = cursor.fetchall()
    stats = {}

    for market_regime, pnl in rows:
        if market_regime not in stats:
            stats[market_regime] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
            }

        stats[market_regime]["trades"] += 1
        stats[market_regime]["total_pnl"] += float(pnl)

        if pnl > 0:
            stats[market_regime]["wins"] += 1
        else:
            stats[market_regime]["losses"] += 1

    cursor.execute("DELETE FROM regime_performance;")

    for market_regime, s in stats.items():
        trades = s["trades"]
        wins = s["wins"]
        losses = s["losses"]
        total_pnl = s["total_pnl"]
        avg_pnl = total_pnl / trades if trades > 0 else 0
        win_rate = wins / trades * 100 if trades > 0 else 0

        cursor.execute("""
            INSERT INTO regime_performance (
                market_regime,
                trades,
                wins,
                losses,
                win_rate,
                total_pnl,
                avg_pnl
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            market_regime,
            trades,
            wins,
            losses,
            win_rate,
            total_pnl,
            avg_pnl,
        ))

    conn.commit()

    print("==============================")
    print("REGIME ANALYTICS ENGINE")
    print("==============================")

    cursor.execute("""
        SELECT
            market_regime,
            trades,
            wins,
            losses,
            win_rate,
            avg_pnl,
            total_pnl
        FROM regime_performance
        ORDER BY total_pnl DESC;
    """)

    for row in cursor.fetchall():
        print(
            f"{row[0]} | "
            f"trades={row[1]} | "
            f"wins={row[2]} | "
            f"losses={row[3]} | "
            f"win_rate={row[4]:.2f}% | "
            f"avg_pnl=${row[5]:.2f} | "
            f"total_pnl=${row[6]:.2f}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


