from configs.database import DB_CONFIG
import psycopg2

STARTING_CAPITAL = 100000.0


def main():

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_equity_curve (
            id BIGSERIAL PRIMARY KEY,
            equity_value DOUBLE PRECISION,
            realized_pnl DOUBLE PRECISION,
            unrealized_pnl DOUBLE PRECISION,
            open_positions INTEGER,
            cash_remaining DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # PnL réalisé

    cursor.execute("""
        SELECT COALESCE(SUM(pnl_usd),0)
        FROM closed_paper_trades;
    """)

    realized_pnl = float(cursor.fetchone()[0])

    # PnL non réalisé

    cursor.execute("""
        SELECT
            symbol,
            entry_price,
            position_size_usd
        FROM paper_positions
        WHERE position_status='OPEN';
    """)

    open_positions = cursor.fetchall()

    unrealized_pnl = 0.0

    for position in open_positions:

        symbol = position[0]
        entry_price = float(position[1])
        position_size = float(position[2])

        cursor.execute("""
            SELECT price
            FROM market_trades
            WHERE symbol=%s
            ORDER BY event_time DESC
            LIMIT 1;
        """, (symbol,))

        latest = cursor.fetchone()

        if latest is None:
            continue

        current_price = float(latest[0])

        pct_return = (
            (current_price - entry_price)
            / entry_price
        )

        unrealized_pnl += (
            position_size * pct_return
        )

    cursor.execute("""
        SELECT COALESCE(SUM(position_size_usd),0)
        FROM paper_positions
        WHERE position_status='OPEN';
    """)

    capital_deployed = float(cursor.fetchone()[0])

    cash_remaining = (
        STARTING_CAPITAL
        + realized_pnl
        - capital_deployed
    )

    equity_value = (
        STARTING_CAPITAL
        + realized_pnl
        + unrealized_pnl
    )

    cursor.execute("""
        INSERT INTO portfolio_equity_curve (
            equity_value,
            realized_pnl,
            unrealized_pnl,
            open_positions,
            cash_remaining
        )
        VALUES (%s,%s,%s,%s,%s);
    """, (
        equity_value,
        realized_pnl,
        unrealized_pnl,
        len(open_positions),
        cash_remaining,
    ))

    conn.commit()

    print("==============================")
    print("EQUITY CURVE ENGINE")
    print("==============================")
    print(f"Starting Capital : ${STARTING_CAPITAL:,.2f}")
    print(f"Realized PnL     : ${realized_pnl:,.2f}")
    print(f"Unrealized PnL   : ${unrealized_pnl:,.2f}")
    print(f"Open Positions   : {len(open_positions)}")
    print(f"Cash Remaining   : ${cash_remaining:,.2f}")
    print(f"Portfolio Equity : ${equity_value:,.2f}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


