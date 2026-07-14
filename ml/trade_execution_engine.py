from datetime import datetime, timezone
from typing import Any

import psycopg2

from configs.database import DB_CONFIG


MAX_PRICE_AGE_SECONDS = 300
DEFAULT_SLIPPAGE_PCT = 0.05
DEFAULT_TRANSACTION_COST_PCT = 0.02


def format_number(
    value: Any,
    decimals: int = 2,
    prefix: str = "",
) -> str:
    """Format optional numeric values without crashing on NULL database fields."""
    if value is None:
        return "N/A"

    return f"{prefix}{float(value):.{decimals}f}"


def calculate_execution(
    market_price: float,
    position_size_usd: float,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    transaction_cost_pct: float = DEFAULT_TRANSACTION_COST_PCT,
) -> dict[str, float]:
    """Calculate simulated BUY execution details."""
    execution_price = market_price * (1 + slippage_pct / 100.0)

    transaction_cost_usd = (
        position_size_usd * transaction_cost_pct / 100.0
    )

    investable_amount_usd = max(
        position_size_usd - transaction_cost_usd,
        0.0,
    )

    quantity = (
        investable_amount_usd / execution_price
        if execution_price > 0
        else 0.0
    )

    return {
        "execution_price": execution_price,
        "transaction_cost_usd": transaction_cost_usd,
        "investable_amount_usd": investable_amount_usd,
        "quantity": quantity,
        "slippage_pct": slippage_pct,
        "transaction_cost_pct": transaction_cost_pct,
    }


def is_price_fresh(event_time: datetime) -> bool:
    """Return True when the market price is recent enough for execution."""
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age_seconds = (now - event_time).total_seconds()

    return 0 <= age_seconds <= MAX_PRICE_AGE_SECONDS


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    selected_strategy TEXT,
                    market_regime TEXT,
                    entry_time TIMESTAMPTZ DEFAULT NOW(),
                    market_price DOUBLE PRECISION,
                    entry_price DOUBLE PRECISION,
                    quantity DOUBLE PRECISION,
                    position_size_usd DOUBLE PRECISION,
                    investable_amount_usd DOUBLE PRECISION,
                    transaction_cost_usd DOUBLE PRECISION,
                    transaction_cost_pct DOUBLE PRECISION,
                    slippage_pct DOUBLE PRECISION,
                    allocation_pct DOUBLE PRECISION,
                    stop_loss_pct DOUBLE PRECISION,
                    take_profit_pct DOUBLE PRECISION,
                    stop_loss_price DOUBLE PRECISION,
                    take_profit_price DOUBLE PRECISION,
                    position_status TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS market_price DOUBLE PRECISION;
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
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS investable_amount_usd
                DOUBLE PRECISION;
                """
            )

            cursor.execute(
                """
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS transaction_cost_usd
                DOUBLE PRECISION;
                """
            )

            cursor.execute(
                """
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS transaction_cost_pct
                DOUBLE PRECISION;
                """
            )

            cursor.execute(
                """
                ALTER TABLE paper_positions
                ADD COLUMN IF NOT EXISTS slippage_pct DOUBLE PRECISION;
                """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                uq_paper_positions_open_symbol
                ON paper_positions (symbol)
                WHERE position_status = 'OPEN';
                """
            )

            cursor.execute(
                """
                SELECT
                    symbol,
                    selected_strategy,
                    allocation_pct,
                    position_size_usd,
                    stop_loss_pct,
                    take_profit_pct
                FROM risk_management_v2_results
                WHERE final_trade_plan = 'TRADE_ALLOWED'
                  AND position_size_usd > 0;
                """
            )

            trade_plans = cursor.fetchall()

            opened = 0
            skipped_existing = 0
            skipped_missing_price = 0
            skipped_stale_price = 0
            skipped_invalid_execution = 0

            for (
                symbol,
                selected_strategy,
                allocation_pct,
                position_size_usd,
                stop_loss_pct,
                take_profit_pct,
            ) in trade_plans:
                selected_strategy = selected_strategy or "UNKNOWN"
                allocation_pct = float(allocation_pct)
                position_size_usd = float(position_size_usd)
                stop_loss_pct = float(stop_loss_pct)
                take_profit_pct = float(take_profit_pct)

                cursor.execute(
                    """
                    SELECT 1
                    FROM paper_positions
                    WHERE symbol = %s
                      AND position_status = 'OPEN'
                    LIMIT 1;
                    """,
                    (symbol,),
                )

                if cursor.fetchone() is not None:
                    skipped_existing += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "an OPEN position already exists."
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
                    skipped_missing_price += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "no market price available."
                    )
                    continue

                market_price = float(price_row[0])
                price_time = price_row[1]

                if not is_price_fresh(price_time):
                    skipped_stale_price += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "latest market price is stale."
                    )
                    continue

                execution = calculate_execution(
                    market_price=market_price,
                    position_size_usd=position_size_usd,
                )

                if (
                    execution["execution_price"] <= 0
                    or execution["quantity"] <= 0
                ):
                    skipped_invalid_execution += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "invalid execution calculation."
                    )
                    continue

                cursor.execute(
                    """
                    SELECT market_regime
                    FROM quant_metrics
                    WHERE symbol = %s
                    ORDER BY metric_time DESC
                    LIMIT 1;
                    """,
                    (symbol,),
                )

                regime_row = cursor.fetchone()

                market_regime = (
                    regime_row[0]
                    if regime_row and regime_row[0]
                    else "UNKNOWN"
                )

                execution_price = execution["execution_price"]

                stop_loss_price = (
                    execution_price
                    * (1 - stop_loss_pct / 100.0)
                )

                take_profit_price = (
                    execution_price
                    * (1 + take_profit_pct / 100.0)
                )

                try:
                    cursor.execute(
                        """
                        INSERT INTO paper_positions (
                            symbol,
                            selected_strategy,
                            market_regime,
                            market_price,
                            entry_price,
                            quantity,
                            position_size_usd,
                            investable_amount_usd,
                            transaction_cost_usd,
                            transaction_cost_pct,
                            slippage_pct,
                            allocation_pct,
                            stop_loss_pct,
                            take_profit_pct,
                            stop_loss_price,
                            take_profit_price,
                            position_status
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s
                        );
                        """,
                        (
                            symbol,
                            selected_strategy,
                            market_regime,
                            market_price,
                            execution_price,
                            execution["quantity"],
                            position_size_usd,
                            execution["investable_amount_usd"],
                            execution["transaction_cost_usd"],
                            execution["transaction_cost_pct"],
                            execution["slippage_pct"],
                            allocation_pct,
                            stop_loss_pct,
                            take_profit_pct,
                            stop_loss_price,
                            take_profit_price,
                            "OPEN",
                        ),
                    )
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    skipped_existing += 1
                    print(
                        f"SKIPPED {symbol}: "
                        "another process opened the position."
                    )
                    continue

                opened += 1

            print("==============================")
            print("TRADE EXECUTION ENGINE V3")
            print("==============================")
            print("Eligible trade plans this cycle:", len(trade_plans))
            print("New positions opened:", opened)
            print("Skipped - existing:", skipped_existing)
            print("Skipped - missing price:", skipped_missing_price)
            print("Skipped - stale price:", skipped_stale_price)
            print(
                "Skipped - invalid execution:",
                skipped_invalid_execution,
            )
            print("Slippage %:", DEFAULT_SLIPPAGE_PCT)
            print(
                "Transaction cost %:",
                DEFAULT_TRANSACTION_COST_PCT,
            )

            cursor.execute(
                """
                SELECT
                    position_id,
                    symbol,
                    COALESCE(selected_strategy, 'UNKNOWN'),
                    COALESCE(market_regime, 'UNKNOWN'),
                    market_price,
                    entry_price,
                    quantity,
                    position_size_usd,
                    transaction_cost_usd,
                    stop_loss_price,
                    take_profit_price,
                    position_status
                FROM paper_positions
                WHERE position_status = 'OPEN'
                ORDER BY entry_time DESC;
                """
            )

            open_positions = cursor.fetchall()
            print("Currently open positions:", len(open_positions))

            for row in open_positions:
                print(
                    f"id={row[0]} | "
                    f"{row[1]} | "
                    f"strategy={row[2]} | "
                    f"regime={row[3]} | "
                    f"market={format_number(row[4], 6)} | "
                    f"entry={format_number(row[5], 6)} | "
                    f"quantity={format_number(row[6], 6)} | "
                    f"position={format_number(row[7], 2, '$')} | "
                    f"cost={format_number(row[8], 2, '$')} | "
                    f"stop={format_number(row[9], 6)} | "
                    f"target={format_number(row[10], 6)} | "
                    f"status={row[11]}"
                )


if __name__ == "__main__":
    main()