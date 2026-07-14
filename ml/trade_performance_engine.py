from typing import Any

import psycopg2

from configs.database import DB_CONFIG


def calculate_performance(
    pnls: list[float],
    holdings: list[float],
) -> dict[str, Any]:
    """Calculate aggregate performance metrics for closed trades."""
    trades_closed = len(pnls)

    winners_list = [pnl for pnl in pnls if pnl > 0]
    losers_list = [pnl for pnl in pnls if pnl < 0]
    breakeven_list = [pnl for pnl in pnls if pnl == 0]

    winners = len(winners_list)
    losers = len(losers_list)
    breakeven = len(breakeven_list)

    win_rate_pct = (
        winners / trades_closed * 100.0
        if trades_closed > 0
        else 0.0
    )

    total_pnl_usd = sum(pnls)
    gross_profit_usd = sum(winners_list)
    gross_loss_usd = abs(sum(losers_list))

    avg_win_usd = (
        gross_profit_usd / winners
        if winners > 0
        else 0.0
    )

    avg_loss_usd = (
        sum(losers_list) / losers
        if losers > 0
        else 0.0
    )

    if gross_loss_usd > 0:
        profit_factor = gross_profit_usd / gross_loss_usd
    elif gross_profit_usd > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    payoff_ratio = (
        avg_win_usd / abs(avg_loss_usd)
        if avg_win_usd > 0 and avg_loss_usd < 0
        else None
    )

    expectancy_usd = (
        total_pnl_usd / trades_closed
        if trades_closed > 0
        else 0.0
    )

    avg_holding_minutes = (
        sum(holdings) / len(holdings)
        if holdings
        else 0.0
    )

    return {
        "trades_closed": trades_closed,
        "winners": winners,
        "losers": losers,
        "breakeven": breakeven,
        "win_rate_pct": win_rate_pct,
        "total_pnl_usd": total_pnl_usd,
        "gross_profit_usd": gross_profit_usd,
        "gross_loss_usd": gross_loss_usd,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy_usd": expectancy_usd,
        "best_trade_usd": max(pnls),
        "worst_trade_usd": min(pnls),
        "avg_holding_minutes": avg_holding_minutes,
    }


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_performance_summary (
                    id BIGSERIAL PRIMARY KEY,
                    trades_closed INTEGER,
                    winners INTEGER,
                    losers INTEGER,
                    breakeven INTEGER,
                    win_rate_pct DOUBLE PRECISION,
                    total_pnl_usd DOUBLE PRECISION,
                    gross_profit_usd DOUBLE PRECISION,
                    gross_loss_usd DOUBLE PRECISION,
                    avg_win_usd DOUBLE PRECISION,
                    avg_loss_usd DOUBLE PRECISION,
                    profit_factor DOUBLE PRECISION,
                    payoff_ratio DOUBLE PRECISION,
                    expectancy_usd DOUBLE PRECISION,
                    best_trade_usd DOUBLE PRECISION,
                    worst_trade_usd DOUBLE PRECISION,
                    avg_holding_minutes DOUBLE PRECISION,
                    latest_trade_id BIGINT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE trade_performance_summary
                ADD COLUMN IF NOT EXISTS breakeven INTEGER,
                ADD COLUMN IF NOT EXISTS gross_profit_usd
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS gross_loss_usd
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS payoff_ratio
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS latest_trade_id BIGINT;
                """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                uq_trade_performance_latest_trade
                ON trade_performance_summary (latest_trade_id)
                WHERE latest_trade_id IS NOT NULL;
                """
            )

            cursor.execute(
                """
                SELECT
                    trade_id,
                    pnl_usd,
                    holding_minutes
                FROM closed_paper_trades
                WHERE pnl_usd IS NOT NULL
                ORDER BY trade_id;
                """
            )

            rows = cursor.fetchall()

            if not rows:
                print("==============================")
                print("TRADE PERFORMANCE ENGINE V2")
                print("==============================")
                print("No closed trades yet.")
                return

            latest_trade_id = int(rows[-1][0])

            cursor.execute(
                """
                SELECT 1
                FROM trade_performance_summary
                WHERE latest_trade_id = %s
                LIMIT 1;
                """,
                (latest_trade_id,),
            )

            if cursor.fetchone() is not None:
                print("==============================")
                print("TRADE PERFORMANCE ENGINE V2")
                print("==============================")
                print(
                    "No new closed trades. "
                    "Performance summary is unchanged."
                )
                return

            pnls = [float(row[1]) for row in rows]

            holdings = [
                float(row[2])
                for row in rows
                if row[2] is not None
            ]

            metrics = calculate_performance(
                pnls=pnls,
                holdings=holdings,
            )

            cursor.execute(
                """
                INSERT INTO trade_performance_summary (
                    trades_closed,
                    winners,
                    losers,
                    breakeven,
                    win_rate_pct,
                    total_pnl_usd,
                    gross_profit_usd,
                    gross_loss_usd,
                    avg_win_usd,
                    avg_loss_usd,
                    profit_factor,
                    payoff_ratio,
                    expectancy_usd,
                    best_trade_usd,
                    worst_trade_usd,
                    avg_holding_minutes,
                    latest_trade_id
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                );
                """,
                (
                    metrics["trades_closed"],
                    metrics["winners"],
                    metrics["losers"],
                    metrics["breakeven"],
                    metrics["win_rate_pct"],
                    metrics["total_pnl_usd"],
                    metrics["gross_profit_usd"],
                    metrics["gross_loss_usd"],
                    metrics["avg_win_usd"],
                    metrics["avg_loss_usd"],
                    metrics["profit_factor"],
                    metrics["payoff_ratio"],
                    metrics["expectancy_usd"],
                    metrics["best_trade_usd"],
                    metrics["worst_trade_usd"],
                    metrics["avg_holding_minutes"],
                    latest_trade_id,
                ),
            )

            profit_factor_text = (
                "INFINITE"
                if metrics["profit_factor"] is None
                else f"{metrics['profit_factor']:.2f}"
            )

            payoff_ratio_text = (
                "N/A"
                if metrics["payoff_ratio"] is None
                else f"{metrics['payoff_ratio']:.2f}"
            )

            print("==============================")
            print("TRADE PERFORMANCE ENGINE V2")
            print("==============================")
            print(
                f"Trades Closed       : "
                f"{metrics['trades_closed']}"
            )
            print(f"Winners             : {metrics['winners']}")
            print(f"Losers              : {metrics['losers']}")
            print(f"Breakeven           : {metrics['breakeven']}")
            print(
                f"Win Rate %          : "
                f"{metrics['win_rate_pct']:.2f}"
            )
            print(
                f"Total PnL USD       : "
                f"{metrics['total_pnl_usd']:.2f}"
            )
            print(
                f"Gross Profit USD    : "
                f"{metrics['gross_profit_usd']:.2f}"
            )
            print(
                f"Gross Loss USD      : "
                f"{metrics['gross_loss_usd']:.2f}"
            )
            print(
                f"Avg Win USD         : "
                f"{metrics['avg_win_usd']:.2f}"
            )
            print(
                f"Avg Loss USD        : "
                f"{metrics['avg_loss_usd']:.2f}"
            )
            print(f"Profit Factor       : {profit_factor_text}")
            print(f"Payoff Ratio        : {payoff_ratio_text}")
            print(
                f"Expectancy USD      : "
                f"{metrics['expectancy_usd']:.2f}"
            )
            print(
                f"Best Trade USD      : "
                f"{metrics['best_trade_usd']:.2f}"
            )
            print(
                f"Worst Trade USD     : "
                f"{metrics['worst_trade_usd']:.2f}"
            )
            print(
                f"Avg Holding Minutes : "
                f"{metrics['avg_holding_minutes']:.2f}"
            )
            print(f"Latest Trade ID     : {latest_trade_id}")


if __name__ == "__main__":
    main()


