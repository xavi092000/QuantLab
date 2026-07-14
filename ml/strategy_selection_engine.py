from __future__ import annotations

from contextlib import closing
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import psycopg2
from psycopg2.extensions import connection as PsycopgConnection

from configs.database import DB_CONFIG


StatsKey = Tuple[str, str]
StatsRecord = Dict[str, float]
SelectionRow = Tuple[str, str, int, float, float, str]


CREATE_SELECTION_MATRIX_SQL = """
    CREATE TABLE IF NOT EXISTS strategy_selection_matrix (
        market_regime TEXT,
        strategy_name TEXT,
        trades INTEGER,
        wins INTEGER,
        win_rate DOUBLE PRECISION,
        total_pnl DOUBLE PRECISION,
        recommendation TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

CLEAR_SELECTION_MATRIX_SQL = """
    DELETE FROM strategy_selection_matrix;
"""

FETCH_CLOSED_TRADES_SQL = """
    SELECT
        COALESCE(market_regime, 'UNKNOWN'),
        COALESCE(selected_strategy, 'UNKNOWN'),
        pnl_usd
    FROM closed_paper_trades;
"""

INSERT_SELECTION_SQL = """
    INSERT INTO strategy_selection_matrix (
        market_regime,
        strategy_name,
        trades,
        wins,
        win_rate,
        total_pnl,
        recommendation
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s);
"""

FETCH_SELECTION_MATRIX_SQL = """
    SELECT
        market_regime,
        strategy_name,
        trades,
        win_rate,
        total_pnl,
        recommendation
    FROM strategy_selection_matrix
    ORDER BY total_pnl DESC;
"""


MIN_TRADES_FOR_DIRECTIONAL_RECOMMENDATION = 3
MIN_WIN_RATE_TO_FAVOR = 50.0


def _as_float(value: Any, *, field_name: str) -> float:
    """Convert a database numeric value to float with a clear failure message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to convert {field_name} value {value!r} to float.") from exc


def calculate_strategy_stats(
    rows: Iterable[Tuple[str, str, Any]],
) -> Mapping[StatsKey, StatsRecord]:
    """Aggregate trade count, wins, and total PnL by market regime and strategy."""
    stats: MutableMapping[StatsKey, StatsRecord] = {}

    for market_regime, strategy, pnl in rows:
        key = (market_regime, strategy)
        pnl_value = _as_float(pnl, field_name="pnl_usd")

        if key not in stats:
            stats[key] = {
                "trades": 0.0,
                "wins": 0.0,
                "total_pnl": 0.0,
            }

        stats[key]["trades"] += 1.0
        stats[key]["total_pnl"] += pnl_value

        if pnl_value > 0:
            stats[key]["wins"] += 1.0

    return stats


def recommendation_for(trades: int, win_rate: float, total_pnl: float) -> str:
    """Return the strategy recommendation using the existing selection rules."""
    if trades >= MIN_TRADES_FOR_DIRECTIONAL_RECOMMENDATION:
        if total_pnl > 0 and win_rate >= MIN_WIN_RATE_TO_FAVOR:
            return "FAVOR"
        if total_pnl < 0:
            return "AVOID"

    return "NEUTRAL"


def build_selection_rows(stats: Mapping[StatsKey, StatsRecord]) -> List[SelectionRow]:
    """Transform aggregated statistics into rows for strategy_selection_matrix."""
    selection_rows: List[SelectionRow] = []

    for (market_regime, strategy), record in stats.items():
        trades = int(record["trades"])
        wins = int(record["wins"])
        total_pnl = float(record["total_pnl"])
        win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
        recommendation = recommendation_for(trades, win_rate, total_pnl)

        selection_rows.append(
            (
                market_regime,
                strategy,
                trades,
                wins,
                win_rate,
                total_pnl,
                recommendation,
            )
        )

    return selection_rows


def refresh_strategy_selection_matrix(conn: PsycopgConnection) -> None:
    """Rebuild strategy_selection_matrix from closed_paper_trades in one transaction."""
    with conn.cursor() as cursor:
        cursor.execute(CREATE_SELECTION_MATRIX_SQL)
        cursor.execute(CLEAR_SELECTION_MATRIX_SQL)
        cursor.execute(FETCH_CLOSED_TRADES_SQL)

        stats = calculate_strategy_stats(cursor.fetchall())
        selection_rows = build_selection_rows(stats)

        if selection_rows:
            cursor.executemany(INSERT_SELECTION_SQL, selection_rows)


def fetch_strategy_selection_matrix(conn: PsycopgConnection) -> List[Tuple[Any, ...]]:
    """Fetch rows for console reporting."""
    with conn.cursor() as cursor:
        cursor.execute(FETCH_SELECTION_MATRIX_SQL)
        return list(cursor.fetchall())


def print_strategy_selection_matrix(rows: Iterable[Tuple[Any, ...]]) -> None:
    print("==============================")
    print("STRATEGY SELECTION ENGINE")
    print("==============================")

    for row in rows:
        print(
            f"{row[0]} | "
            f"{row[1]} | "
            f"trades={row[2]} | "
            f"win_rate={row[3]:.2f}% | "
            f"total_pnl=${row[4]:.2f} | "
            f"{row[5]}"
        )


def main() -> None:
    with closing(psycopg2.connect(**DB_CONFIG)) as conn:
        try:
            refresh_strategy_selection_matrix(conn)
            conn.commit()
            rows = fetch_strategy_selection_matrix(conn)
        except Exception:
            conn.rollback()
            raise

    print_strategy_selection_matrix(rows)


if __name__ == "__main__":
    main()
