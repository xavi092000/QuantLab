"""Build and display strategy-level attribution metrics for closed paper trades."""

from __future__ import annotations

import sys
from decimal import Decimal
from typing import Optional, Sequence, Tuple

import psycopg2
from psycopg2.extensions import connection as PgConnection

from configs.database import DB_CONFIG


RESULTS_TABLE = "strategy_attribution_results"

REBUILD_ATTRIBUTION_TABLE_SQL = """
    DROP TABLE IF EXISTS strategy_attribution_results;

    CREATE TABLE strategy_attribution_results AS
    SELECT
        selected_strategy,

        COUNT(*) AS total_trades,

        COUNT(*) FILTER (
            WHERE pnl_usd > 0
        ) AS winning_trades,

        COUNT(*) FILTER (
            WHERE pnl_usd < 0
        ) AS losing_trades,

        ROUND(
            AVG(pnl_usd)::numeric,
            2
        ) AS avg_pnl_usd,

        ROUND(
            SUM(pnl_usd)::numeric,
            2
        ) AS total_pnl_usd,

        ROUND(
            AVG(pnl_pct)::numeric,
            2
        ) AS avg_pnl_pct,

        ROUND(
            (
                COUNT(*) FILTER (WHERE pnl_usd > 0)
                * 100.0
                / NULLIF(COUNT(*),0)
            )::numeric,
            2
        ) AS win_rate_pct

    FROM closed_paper_trades
    GROUP BY selected_strategy;
"""

FETCH_ATTRIBUTION_RESULTS_SQL = """
    SELECT
        selected_strategy,
        total_trades,
        winning_trades,
        losing_trades,
        avg_pnl_usd,
        total_pnl_usd,
        avg_pnl_pct,
        win_rate_pct
    FROM strategy_attribution_results
    ORDER BY total_pnl_usd DESC;
"""

AttributionRow = Tuple[
    Optional[str],
    int,
    int,
    int,
    Optional[Decimal],
    Optional[Decimal],
    Optional[Decimal],
    Optional[Decimal],
]


def rebuild_attribution_table(conn: PgConnection) -> None:
    """Recreate the strategy attribution results table in one transaction."""
    with conn.cursor() as cursor:
        cursor.execute(REBUILD_ATTRIBUTION_TABLE_SQL)


def fetch_attribution_results(conn: PgConnection) -> Sequence[AttributionRow]:
    """Fetch strategy attribution rows in reporting order."""
    with conn.cursor() as cursor:
        cursor.execute(FETCH_ATTRIBUTION_RESULTS_SQL)
        return cursor.fetchall()


def print_report(rows: Sequence[AttributionRow]) -> None:
    """Print the attribution report using the legacy console format."""
    print("==============================")
    print("STRATEGY ATTRIBUTION ENGINE")
    print("==============================")

    for row in rows:
        print(
            f"{row[0]} | trades={row[1]} | "
            f"wins={row[2]} | losses={row[3]} | "
            f"avg_pnl=${row[4]} | "
            f"total_pnl=${row[5]} | "
            f"win_rate={row[7]}%"
        )


def run() -> Sequence[AttributionRow]:
    """
    Rebuild strategy attribution results and return rows used for reporting.

    PostgreSQL DDL is transactional, so failures during the rebuild roll back the
    DROP/CREATE operation instead of leaving a partially rebuilt table committed.
    """
    conn: Optional[PgConnection] = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        rebuild_attribution_table(conn)
        conn.commit()

        rows = fetch_attribution_results(conn)
        print_report(rows)
        return rows

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    try:
        run()
    except Exception as exc:
        print(
            f"Strategy attribution engine failed: {exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
