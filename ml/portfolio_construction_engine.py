from typing import Any

import psycopg2

from configs.database import DB_CONFIG


PORTFOLIO_VALUE_USD = 100_000.0
MAX_TOTAL_EXPOSURE_PCT = 30.0
MAX_ASSET_ALLOCATION_PCT = 10.0

BASE_SCORE = 50.0

STRATEGY_SCORES = {
    "MOMENTUM": 25.0,
    "MEAN_REVERSION": 20.0,
}

ML_VOTE_SCORES = {
    "SUPPORT": 25.0,
    "NEUTRAL": 10.0,
    "AGAINST": -20.0,
}

POSITIVE_RETURN_BONUS = 10.0


def calculate_raw_score(
    strategy: str,
    ml_vote: str,
    predicted_return: float,
) -> float:
    """Calculate the portfolio-ranking score for one BUY candidate."""
    score = BASE_SCORE
    score += STRATEGY_SCORES.get(strategy, 0.0)
    score += ML_VOTE_SCORES.get(ml_vote, 0.0)

    if predicted_return > 0:
        score += POSITIVE_RETURN_BONUS

    return max(score, 0.0)


def build_allocations(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Allocate exposure proportionally while respecting portfolio limits."""
    total_score = sum(candidate["raw_score"] for candidate in candidates)

    allocated_candidates: list[dict[str, Any]] = []

    for candidate in candidates:
        if total_score > 0:
            allocation_pct = (
                candidate["raw_score"] / total_score
            ) * MAX_TOTAL_EXPOSURE_PCT
        else:
            allocation_pct = 0.0

        allocation_pct = min(
            allocation_pct,
            MAX_ASSET_ALLOCATION_PCT,
        )

        position_size_usd = (
            PORTFOLIO_VALUE_USD
            * allocation_pct
            / 100.0
        )

        allocated_candidates.append(
            {
                **candidate,
                "allocation_pct": allocation_pct,
                "position_size_usd": position_size_usd,
            }
        )

    total_allocated_pct = sum(
        candidate["allocation_pct"]
        for candidate in allocated_candidates
    )

    cash_remaining_pct = max(
        100.0 - total_allocated_pct,
        0.0,
    )

    return allocated_candidates, cash_remaining_pct


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_construction_results (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT,
                    selected_strategy TEXT,
                    final_decision TEXT,
                    ml_vote TEXT,
                    predicted_return_5m DOUBLE PRECISION,
                    raw_score DOUBLE PRECISION,
                    allocation_pct DOUBLE PRECISION,
                    position_size_usd DOUBLE PRECISION,
                    cash_remaining_pct DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )

            # The table represents the latest portfolio snapshot.
            # A future version can add run_id for historical snapshots.
            cursor.execute(
                "TRUNCATE TABLE portfolio_construction_results;"
            )

            cursor.execute(
                """
                SELECT
                    symbol,
                    selected_strategy,
                    final_decision,
                    ml_vote,
                    predicted_return_5m
                FROM final_strategy_decisions
                WHERE final_decision = 'BUY';
                """
            )

            buy_rows = cursor.fetchall()

            candidates: list[dict[str, Any]] = []

            for (
                symbol,
                strategy,
                decision,
                ml_vote,
                predicted_return,
            ) in buy_rows:
                pred_return = (
                    float(predicted_return)
                    if predicted_return is not None
                    else 0.0
                )

                candidates.append(
                    {
                        "symbol": symbol,
                        "strategy": strategy,
                        "decision": decision,
                        "ml_vote": ml_vote,
                        "pred_return": pred_return,
                        "raw_score": calculate_raw_score(
                            strategy=strategy,
                            ml_vote=ml_vote,
                            predicted_return=pred_return,
                        ),
                    }
                )

            allocated_candidates, cash_remaining_pct = (
                build_allocations(candidates)
            )

            for item in allocated_candidates:
                cursor.execute(
                    """
                    INSERT INTO portfolio_construction_results (
                        symbol,
                        selected_strategy,
                        final_decision,
                        ml_vote,
                        predicted_return_5m,
                        raw_score,
                        allocation_pct,
                        position_size_usd,
                        cash_remaining_pct
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        item["symbol"],
                        item["strategy"],
                        item["decision"],
                        item["ml_vote"],
                        item["pred_return"],
                        item["raw_score"],
                        item["allocation_pct"],
                        item["position_size_usd"],
                        cash_remaining_pct,
                    ),
                )

            print("==============================")
            print("PORTFOLIO CONSTRUCTION ENGINE")
            print("==============================")
            print("BUY assets:", len(allocated_candidates))
            print("Portfolio value:", PORTFOLIO_VALUE_USD)
            print("Max total exposure %:", MAX_TOTAL_EXPOSURE_PCT)
            print(
                "Max asset allocation %:",
                MAX_ASSET_ALLOCATION_PCT,
            )
            print(
                "Cash remaining %:",
                round(cash_remaining_pct, 2),
            )

            cursor.execute(
                """
                SELECT
                    symbol,
                    selected_strategy,
                    final_decision,
                    ml_vote,
                    ROUND(raw_score::numeric, 2),
                    ROUND(allocation_pct::numeric, 2),
                    ROUND(position_size_usd::numeric, 2),
                    ROUND(cash_remaining_pct::numeric, 2)
                FROM portfolio_construction_results
                ORDER BY allocation_pct DESC;
                """
            )

            rows = cursor.fetchall()

            if not rows:
                print(
                    "No BUY assets. "
                    "Portfolio remains 100% cash."
                )

            for row in rows:
                print(
                    f"{row[0]} | "
                    f"strategy={row[1]} | "
                    f"decision={row[2]} | "
                    f"ml={row[3]} | "
                    f"score={row[4]} | "
                    f"allocation={row[5]}% | "
                    f"position=${row[6]} | "
                    f"cash_remaining={row[7]}%"
                )


if __name__ == "__main__":
    main()


