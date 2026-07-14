from datetime import datetime, timezone
from typing import Any

import psycopg2

from configs.database import DB_CONFIG


MAX_PRICE_AGE_SECONDS = 300
EXIT_SLIPPAGE_PCT = 0.05
EXIT_TRANSACTION_COST_PCT = 0.02



def fmt(value: Any, decimals: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{decimals}f}"


def is_price_fresh(event_time: datetime) -> bool:
    """Return True when the latest market price is recent enough."""
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)

    age_seconds = (
        datetime.now(timezone.utc) - event_time
    ).total_seconds()

    return 0 <= age_seconds <= MAX_PRICE_AGE_SECONDS


def calculate_exit(
    entry_price: float,
    trigger_price: float,
    quantity: float,
    close_reason: str,
) -> dict[str, float]:
    """Calculate simulated exit price, costs and realized PnL."""
    if close_reason == "CLOSED_STOP_LOSS":
        exit_price = trigger_price * (
            1 - EXIT_SLIPPAGE_PCT / 100.0
        )
    else:
        exit_price = trigger_price * (
            1 - EXIT_SLIPPAGE_PCT / 100.0
        )

    gross_exit_value_usd = quantity * exit_price

    exit_transaction_cost_usd = (
        gross_exit_value_usd
        * EXIT_TRANSACTION_COST_PCT
        / 100.0
    )

    net_exit_value_usd = (
        gross_exit_value_usd - exit_transaction_cost_usd
    )

    initial_value_usd = quantity * entry_price
    gross_pnl_usd = gross_exit_value_usd - initial_value_usd
    net_pnl_usd = net_exit_value_usd - initial_value_usd

    pnl_pct = (
        net_pnl_usd / initial_value_usd * 100.0
        if initial_value_usd > 0
        else 0.0
    )

    return {
        "exit_price": exit_price,
        "gross_exit_value_usd": gross_exit_value_usd,
        "net_exit_value_usd": net_exit_value_usd,
        "exit_transaction_cost_usd": exit_transaction_cost_usd,
        "gross_pnl_usd": gross_pnl_usd,
        "net_pnl_usd": net_pnl_usd,
        "pnl_pct": pnl_pct,
    }


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_paper_trades (
                    trade_id BIGSERIAL PRIMARY KEY,
                    position_id BIGINT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    selected_strategy TEXT,
                    market_regime TEXT,
                    entry_time TIMESTAMPTZ,
                    exit_time TIMESTAMPTZ,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    quantity DOUBLE PRECISION,
                    position_size_usd DOUBLE PRECISION,
                    gross_exit_value_usd DOUBLE PRECISION,
                    net_exit_value_usd DOUBLE PRECISION,
                    exit_transaction_cost_usd DOUBLE PRECISION,
                    stop_loss_price DOUBLE PRECISION,
                    take_profit_price DOUBLE PRECISION,
                    close_reason TEXT,
                    gross_pnl_usd DOUBLE PRECISION,
                    pnl_usd DOUBLE PRECISION,
                    pnl_pct DOUBLE PRECISION,
                    holding_minutes DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE closed_paper_trades
                ADD COLUMN IF NOT EXISTS quantity DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS gross_exit_value_usd
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS net_exit_value_usd
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS exit_transaction_cost_usd
                    DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS gross_pnl_usd
                    DOUBLE PRECISION;
                """
            )


            cursor.execute(
                """
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS quantity DOUBLE PRECISION;
                """
            )

            cursor.execute(
                """
                SELECT
                    position_id,
                    symbol,
                    COALESCE(selected_strategy, 'UNKNOWN'),
                    COALESCE(market_regime, 'UNKNOWN'),
                    entry_time,
                    entry_price,
                    COALESCE(quantity, 0),
                    position_size_usd,
                    stop_loss_price,
                    take_profit_price
                FROM paper_positions
                WHERE position_status = 'OPEN'
                ORDER BY entry_time
                FOR UPDATE SKIP LOCKED;
                """
            )

            open_positions = cursor.fetchall()

            closed = 0
            still_open = 0
            missing_prices = 0
            stale_prices = 0
            invalid_positions = 0

            for (
                position_id,
                symbol,
                selected_strategy,
                market_regime,
                entry_time,
                entry_price,
                quantity,
                position_size_usd,
                stop_loss_price,
                take_profit_price,
            ) in open_positions:
                entry_price = float(entry_price)
                quantity = float(quantity or 0.0)
                position_size_usd = float(position_size_usd)
                stop_loss_price = float(stop_loss_price)
                take_profit_price = float(take_profit_price)

                if quantity <= 0 and entry_price > 0:
                    quantity = position_size_usd / entry_price

                if entry_price <= 0 or quantity <= 0:
                    invalid_positions += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "invalid entry price or quantity."
                    )
                    continue

                cursor.execute(
                    """
                    SELECT
                        price,
                        event_time
                    FROM market_trades
                    WHERE symbol = %s
                    ORDER BY event_time DESC
                    LIMIT 1;
                    """,
                    (symbol,),
                )

                price_row = cursor.fetchone()

                if price_row is None:
                    missing_prices += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "no market price available."
                    )
                    continue

                current_price = float(price_row[0])
                exit_time = price_row[1]

                if not is_price_fresh(exit_time):
                    stale_prices += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "latest market price is stale."
                    )
                    continue

                close_reason = None
                trigger_price = None

                if current_price <= stop_loss_price:
                    close_reason = "CLOSED_STOP_LOSS"
                    trigger_price = stop_loss_price

                elif current_price >= take_profit_price:
                    close_reason = "CLOSED_TAKE_PROFIT"
                    trigger_price = take_profit_price

                if close_reason is None or trigger_price is None:
                    still_open += 1
                    continue

                exit_result = calculate_exit(
                    entry_price=entry_price,
                    trigger_price=trigger_price,
                    quantity=quantity,
                    close_reason=close_reason,
                )

                holding_minutes = max(
                    (exit_time - entry_time).total_seconds() / 60.0,
                    0.0,
                )

                cursor.execute(
                    """
                    INSERT INTO closed_paper_trades (
                        position_id,
                        symbol,
                        selected_strategy,
                        market_regime,
                        entry_time,
                        exit_time,
                        entry_price,
                        exit_price,
                        quantity,
                        position_size_usd,
                        gross_exit_value_usd,
                        net_exit_value_usd,
                        exit_transaction_cost_usd,
                        stop_loss_price,
                        take_profit_price,
                        close_reason,
                        gross_pnl_usd,
                        pnl_usd,
                        pnl_pct,
                        holding_minutes
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (position_id) DO NOTHING;
                    """,
                    (
                        position_id,
                        symbol,
                        selected_strategy,
                        market_regime,
                        entry_time,
                        exit_time,
                        entry_price,
                        exit_result["exit_price"],
                        quantity,
                        position_size_usd,
                        exit_result["gross_exit_value_usd"],
                        exit_result["net_exit_value_usd"],
                        exit_result[
                            "exit_transaction_cost_usd"
                        ],
                        stop_loss_price,
                        take_profit_price,
                        close_reason,
                        exit_result["gross_pnl_usd"],
                        exit_result["net_pnl_usd"],
                        exit_result["pnl_pct"],
                        holding_minutes,
                    ),
                )

                if cursor.rowcount == 0:
                    continue

                cursor.execute(
                    """
                    UPDATE paper_positions
                    SET position_status = %s
                    WHERE position_id = %s
                      AND position_status = 'OPEN';
                    """,
                    (
                        close_reason,
                        position_id,
                    ),
                )

                closed += 1

            print("==============================")
            print("POSITION MONITOR ENGINE V2")
            print("==============================")
            print("Open positions checked:", len(open_positions))
            print("Positions closed:", closed)
            print("Positions still open:", still_open)
            print("Missing prices:", missing_prices)
            print("Stale prices:", stale_prices)
            print("Invalid positions:", invalid_positions)
            print("Exit slippage %:", EXIT_SLIPPAGE_PCT)
            print(
                "Exit transaction cost %:",
                EXIT_TRANSACTION_COST_PCT,
            )

            cursor.execute(
                """
                SELECT
                    position_id,
                    symbol,
                    COALESCE(selected_strategy, 'UNKNOWN'),
                    COALESCE(market_regime, 'UNKNOWN'),
                    entry_price,
                    stop_loss_price,
                    take_profit_price,
                    position_status
                FROM paper_positions
                ORDER BY created_at DESC
                LIMIT 5;
                """
            )

            for row in cursor.fetchall():
                print(
                    f"id={row[0]} | "
                    f"{row[1]} | "
                    f"strategy={row[2]} | "
                    f"regime={row[3]} | "
                    f"entry={fmt(row[4])} | "
                    f"stop={fmt(row[5])} | "
                    f"target={fmt(row[6])} | "
                    f"status={row[7]}"
                )


if __name__ == "__main__":
    main()