
from typing import Any

import psycopg2

from configs.database import DB_CONFIG


STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 6.0

MIN_RISK_REWARD_RATIO = 2.0
MAX_ASSET_ALLOCATION_PCT = 10.0
MAX_POSITION_LOSS_USD = 2_000.0


def evaluate_trade_risk(
    allocation_pct: float,
    position_size_usd: float,
) -> dict[str, Any]:
    """Evaluate one proposed position against explicit risk constraints."""
    max_loss_usd = position_size_usd * STOP_LOSS_PCT / 100.0
    target_profit_usd = position_size_usd * TAKE_PROFIT_PCT / 100.0

    risk_reward_ratio = (
        target_profit_usd / max_loss_usd
        if max_loss_usd > 0
        else 0.0
    )

    failures: list[str] = []

    if allocation_pct > MAX_ASSET_ALLOCATION_PCT:
        failures.append(
            f"allocation exceeds {MAX_ASSET_ALLOCATION_PCT:.1f}%"
        )

    if max_loss_usd > MAX_POSITION_LOSS_USD:
        failures.append(
            f"maximum loss exceeds ${MAX_POSITION_LOSS_USD:,.2f}"
        )

    if risk_reward_ratio < MIN_RISK_REWARD_RATIO:
        failures.append(
            f"risk/reward ratio is below {MIN_RISK_REWARD_RATIO:.1f}"
        )

    if failures:
        final_trade_plan = "NO_TRADE"
        risk_reason = "; ".join(failures)
    else:
        final_trade_plan = "TRADE_ALLOWED"
        risk_reason = "All configured risk constraints passed"

    return {
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "max_loss_usd": max_loss_usd,
        "target_profit_usd": target_profit_usd,
        "risk_reward_ratio": risk_reward_ratio,
        "final_trade_plan": final_trade_plan,
        "risk_reason": risk_reason,
    }


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_management_v2_results (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT,
                    selected_strategy TEXT,
                    final_decision TEXT,
                    allocation_pct DOUBLE PRECISION,
                    position_size_usd DOUBLE PRECISION,
                    stop_loss_pct DOUBLE PRECISION,
                    take_profit_pct DOUBLE PRECISION,
                    max_loss_usd DOUBLE PRECISION,
                    target_profit_usd DOUBLE PRECISION,
                    risk_reward_ratio DOUBLE PRECISION,
                    final_trade_plan TEXT,
                    risk_reason TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            # Keep only the latest risk snapshot.
            # A future version can add run_id for historical runs.
            cursor.execute(
                "TRUNCATE TABLE risk_management_v2_results;"
            )

            cursor.execute(
                """
                SELECT
                    symbol,
                    selected_strategy,
                    final_decision,
                    allocation_pct,
                    position_size_usd
                FROM portfolio_construction_results
                WHERE final_decision = 'BUY'
                  AND position_size_usd > 0;
                """
            )

            rows = cursor.fetchall()
            inserted = 0
            allowed = 0
            rejected = 0

            for (
                symbol,
                selected_strategy,
                final_decision,
                allocation_pct,
                position_size_usd,
            ) in rows:
                allocation_pct = float(allocation_pct)
                position_size_usd = float(position_size_usd)

                risk_result = evaluate_trade_risk(
                    allocation_pct=allocation_pct,
                    position_size_usd=position_size_usd,
                )

                cursor.execute(
                    """
                    INSERT INTO risk_management_v2_results (
                        symbol,
                        selected_strategy,
                        final_decision,
                        allocation_pct,
                        position_size_usd,
                        stop_loss_pct,
                        take_profit_pct,
                        max_loss_usd,
                        target_profit_usd,
                        risk_reward_ratio,
                        final_trade_plan,
                        risk_reason
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    );
                    """,
                    (
                        symbol,
                        selected_strategy,
                        final_decision,
                        allocation_pct,
                        position_size_usd,
                        risk_result["stop_loss_pct"],
                        risk_result["take_profit_pct"],
                        risk_result["max_loss_usd"],
                        risk_result["target_profit_usd"],
                        risk_result["risk_reward_ratio"],
                        risk_result["final_trade_plan"],
                        risk_result["risk_reason"],
                    ),
                )

                inserted += 1

                if risk_result["final_trade_plan"] == "TRADE_ALLOWED":
                    allowed += 1
                else:
                    rejected += 1

            print("==============================")
            print("RISK MANAGEMENT V2")
            print("==============================")
            print("Assets processed:", inserted)
            print("Trades allowed:", allowed)
            print("Trades rejected:", rejected)
            print("Stop loss %:", STOP_LOSS_PCT)
            print("Take profit %:", TAKE_PROFIT_PCT)
            print(
                "Maximum position loss:",
                f"${MAX_POSITION_LOSS_USD:,.2f}",
            )

            cursor.execute(
                """
                SELECT
                    symbol,
                    selected_strategy,
                    final_decision,
                    allocation_pct,
                    position_size_usd,
                    max_loss_usd,
                    target_profit_usd,
                    risk_reward_ratio,
                    final_trade_plan,
                    risk_reason
                FROM risk_management_v2_results
                ORDER BY final_trade_plan, symbol;
                """
            )

            for row in cursor.fetchall():
                print(
                    f"{row[0]} | "
                    f"strategy={row[1]} | "
                    f"decision={row[2]} | "
                    f"allocation={row[3]:.2f}% | "
                    f"position=${row[4]:.2f} | "
                    f"max_loss=${row[5]:.2f} | "
                    f"target=${row[6]:.2f} | "
                    f"RR={row[7]:.2f} | "
                    f"plan={row[8]} | "
                    f"reason={row[9]}"
                )


if __name__ == "__main__":
    main()

