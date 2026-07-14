from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import psycopg2

from configs.database import DB_CONFIG


CREATE_PERFORMANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS strategy_performance (
        strategy_name TEXT PRIMARY KEY,
        trades INTEGER,
        wins INTEGER,
        losses INTEGER,
        win_rate DOUBLE PRECISION,
        total_pnl DOUBLE PRECISION,
        avg_pnl DOUBLE PRECISION,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

NULL_PNL_COUNT_SQL = """
    SELECT COUNT(*)
    FROM closed_paper_trades
    WHERE pnl_usd IS NULL;
"""

AGGREGATE_STRATEGY_PERFORMANCE_SQL = """
    SELECT
        COALESCE(selected_strategy, 'UNKNOWN') AS strategy_name,
        COUNT(*)::INTEGER AS trades,
        COUNT(*) FILTER (WHERE pnl_usd > 0)::INTEGER AS wins,
        COUNT(*) FILTER (WHERE pnl_usd <= 0)::INTEGER AS losses,
        (
            COUNT(*) FILTER (WHERE pnl_usd > 0)::DOUBLE PRECISION
            / NULLIF(COUNT(*)::DOUBLE PRECISION, 0.0)
            * 100.0
        ) AS win_rate,
        SUM(pnl_usd)::DOUBLE PRECISION AS total_pnl,
        AVG(pnl_usd)::DOUBLE PRECISION AS avg_pnl
    FROM closed_paper_trades
    GROUP BY COALESCE(selected_strategy, 'UNKNOWN')
    ORDER BY COALESCE(selected_strategy, 'UNKNOWN');
"""

DELETE_PERFORMANCE_SQL = """
    DELETE FROM strategy_performance;
"""

INSERT_PERFORMANCE_SQL = """
    INSERT INTO strategy_performance (
        strategy_name,
        trades,
        wins,
        losses,
        win_rate,
        total_pnl,
        avg_pnl
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s);
"""

SELECT_PERFORMANCE_SQL = """
    SELECT
        strategy_name,
        trades,
        wins,
        losses,
        win_rate,
        avg_pnl,
        total_pnl
    FROM strategy_performance
    ORDER BY total_pnl DESC;
"""


@dataclass(frozen=True)
class StrategyPerformance:
    strategy_name: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float

    def as_insert_params(self) -> Tuple[str, int, int, int, float, float, float]:
        return (
            self.strategy_name,
            self.trades,
            self.wins,
            self.losses,
            self.win_rate,
            self.total_pnl,
            self.avg_pnl,
        )


def ensure_performance_table(cursor: Any) -> None:
    cursor.execute(CREATE_PERFORMANCE_TABLE_SQL)


def validate_source_trades(cursor: Any) -> None:
    """Fail before refreshing analytics if source PnL values cannot be computed."""
    cursor.execute(NULL_PNL_COUNT_SQL)
    null_pnl_count = cursor.fetchone()[0]

    if null_pnl_count:
        raise ValueError(
            "closed_paper_trades contains rows with NULL pnl_usd; "
            "strategy analytics cannot compute performance safely."
        )


def fetch_strategy_performance(cursor: Any) -> List[StrategyPerformance]:
    cursor.execute(AGGREGATE_STRATEGY_PERFORMANCE_SQL)

    performances: List[StrategyPerformance] = []
    for row in cursor.fetchall():
        performances.append(
            StrategyPerformance(
                strategy_name=row[0],
                trades=int(row[1]),
                wins=int(row[2]),
                losses=int(row[3]),
                win_rate=float(row[4]),
                total_pnl=float(row[5]),
                avg_pnl=float(row[6]),
            )
        )

    return performances


def refresh_strategy_performance(
    cursor: Any,
    performances: Iterable[StrategyPerformance],
) -> None:
    cursor.execute(DELETE_PERFORMANCE_SQL)

    insert_params: Sequence[Tuple[str, int, int, int, float, float, float]] = [
        performance.as_insert_params() for performance in performances
    ]

    if insert_params:
        cursor.executemany(INSERT_PERFORMANCE_SQL, insert_params)


def print_strategy_performance(cursor: Any) -> None:
    print("==============================")
    print("STRATEGY ANALYTICS ENGINE")
    print("==============================")

    cursor.execute(SELECT_PERFORMANCE_SQL)

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


def main() -> None:
    conn = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)

        with conn.cursor() as cursor:
            ensure_performance_table(cursor)
            validate_source_trades(cursor)

            performances = fetch_strategy_performance(cursor)
            refresh_strategy_performance(cursor, performances)

            conn.commit()

            print_strategy_performance(cursor)

    except Exception:
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
