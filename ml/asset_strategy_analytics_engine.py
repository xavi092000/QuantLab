"""Asset-level strategy performance analytics for closed paper trades.

The engine rebuilds ``asset_strategy_performance`` from ``closed_paper_trades``
using the same lifecycle as the original module: create the destination table if
needed, delete existing analytics rows, aggregate closed trades, insert refreshed
analytics, then render a console report.

Source data policy: ``closed_paper_trades.pnl_usd`` is required for every source
row. NULL or non-numeric PnL values abort the refresh and the transaction is
rolled back, preventing a partially refreshed analytics table. This fail-fast
policy avoids silently treating unknown PnL as zero or as a loss.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extensions import connection as PgConnection

from configs.database import DB_CONFIG


logger = logging.getLogger(__name__)

StatsKey = Tuple[Optional[str], str, str]
SourceRow = Tuple[Optional[str], str, str, Any]
ReportRow = Tuple[Optional[str], str, str, int, float, float, float, str]
PerformanceRecord = Tuple[
    Optional[str], str, str, int, int, int, float, float, float, str
]
ConnectionFactory = Callable[[], PgConnection]
OutputFunc = Callable[[str], None]

SOURCE_FETCH_BATCH_SIZE = 1_000
SOURCE_CURSOR_NAME = "asset_strategy_source_cursor"

CREATE_PERFORMANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS asset_strategy_performance (
        symbol TEXT,
        selected_strategy TEXT,
        market_regime TEXT,
        trades INTEGER,
        wins INTEGER,
        losses INTEGER,
        win_rate DOUBLE PRECISION,
        total_pnl DOUBLE PRECISION,
        avg_pnl DOUBLE PRECISION,
        recommendation TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

DELETE_PERFORMANCE_SQL = "DELETE FROM asset_strategy_performance;"

SELECT_CLOSED_TRADES_SQL = """
    SELECT
        symbol,
        COALESCE(selected_strategy, 'UNKNOWN') AS selected_strategy,
        COALESCE(market_regime, 'UNKNOWN') AS market_regime,
        pnl_usd
    FROM closed_paper_trades;
"""

INSERT_PERFORMANCE_SQL = """
    INSERT INTO asset_strategy_performance (
        symbol,
        selected_strategy,
        market_regime,
        trades,
        wins,
        losses,
        win_rate,
        total_pnl,
        avg_pnl,
        recommendation
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

SELECT_REPORT_SQL = """
    SELECT
        symbol,
        selected_strategy,
        market_regime,
        trades,
        win_rate,
        avg_pnl,
        total_pnl,
        recommendation
    FROM asset_strategy_performance
    ORDER BY total_pnl DESC;
"""


class InvalidSourceDataError(ValueError):
    """Raised when closed trade source data cannot support valid analytics."""


@dataclass
class PerformanceStats:
    """Aggregated trade statistics for one symbol/strategy/regime key."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        """Return win rate as a percentage, matching the original report format."""
        return self.wins / self.trades * 100 if self.trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        """Return average PnL per trade."""
        return self.total_pnl / self.trades if self.trades > 0 else 0.0


def _default_connection_factory() -> PgConnection:
    """Create a PostgreSQL connection using the project database configuration."""
    return psycopg2.connect(**DB_CONFIG)


def _coerce_pnl(pnl_usd: Any, key: StatsKey) -> float:
    """Convert source PnL to float or raise a contextual data-quality error.

    NULL and non-numeric values are rejected because including them would change
    the meaning of total PnL, average PnL, win counts, and loss counts. The
    caller runs inside an explicit transaction, so these errors roll back the
    destination table refresh.
    """
    if pnl_usd is None:
        raise InvalidSourceDataError(
            "closed_paper_trades.pnl_usd must not be NULL for asset strategy "
            f"analytics. Row key: symbol={key[0]!r}, "
            f"selected_strategy={key[1]!r}, market_regime={key[2]!r}."
        )

    try:
        return float(pnl_usd)
    except (TypeError, ValueError) as exc:
        raise InvalidSourceDataError(
            "closed_paper_trades.pnl_usd must be numeric for asset strategy "
            f"analytics. Row key: symbol={key[0]!r}, "
            f"selected_strategy={key[1]!r}, market_regime={key[2]!r}, "
            f"pnl_usd={pnl_usd!r}."
        ) from exc


def _accumulate_stats(
    stats: Dict[StatsKey, PerformanceStats], rows: Iterable[SourceRow]
) -> None:
    """Accumulate a batch of source rows into the supplied stats dictionary."""
    for symbol, selected_strategy, market_regime, pnl_usd in rows:
        key = (symbol, selected_strategy, market_regime)
        pnl_value = _coerce_pnl(pnl_usd, key)

        stat = stats.setdefault(key, PerformanceStats())
        stat.trades += 1
        stat.total_pnl += pnl_value

        if pnl_value > 0:
            stat.wins += 1
        else:
            stat.losses += 1


def _load_stats(conn: PgConnection) -> Dict[StatsKey, PerformanceStats]:
    """Stream closed trades from PostgreSQL and build aggregate statistics.

    A server-side cursor avoids loading the entire ``closed_paper_trades`` table
    into memory while preserving the original Python aggregation rules.
    """
    stats: Dict[StatsKey, PerformanceStats] = {}

    with conn.cursor(name=SOURCE_CURSOR_NAME) as cursor:
        cursor.itersize = SOURCE_FETCH_BATCH_SIZE
        cursor.execute(SELECT_CLOSED_TRADES_SQL)

        while True:
            rows: Sequence[SourceRow] = cursor.fetchmany(SOURCE_FETCH_BATCH_SIZE)
            if not rows:
                break
            _accumulate_stats(stats, rows)

    return stats


def _recommendation(stat: PerformanceStats) -> str:
    """Return the strategy recommendation for an aggregate stats bucket."""
    if stat.trades >= 3:
        if stat.total_pnl > 0 and stat.win_rate >= 50:
            return "FAVOR"
        if stat.total_pnl < 0:
            return "AVOID"

    return "NEUTRAL"


def _performance_records(
    stats: Dict[StatsKey, PerformanceStats]
) -> List[PerformanceRecord]:
    """Convert aggregate stats into rows ready for destination insertion."""
    records: List[PerformanceRecord] = []

    for key, stat in stats.items():
        symbol, selected_strategy, market_regime = key
        records.append(
            (
                symbol,
                selected_strategy,
                market_regime,
                stat.trades,
                stat.wins,
                stat.losses,
                stat.win_rate,
                stat.total_pnl,
                stat.avg_pnl,
                _recommendation(stat),
            )
        )

    return records


def _refresh_performance(conn: PgConnection) -> int:
    """Rebuild ``asset_strategy_performance`` in one database transaction."""
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_PERFORMANCE_TABLE_SQL)
            cursor.execute(DELETE_PERFORMANCE_SQL)

        stats = _load_stats(conn)
        records = _performance_records(stats)

        if records:
            with conn.cursor() as cursor:
                cursor.executemany(INSERT_PERFORMANCE_SQL, records)

    logger.info(
        "Asset strategy performance refreshed.",
        extra={"rows_inserted": len(records), "engine": "asset_strategy_analytics"},
    )
    return len(records)


def _fetch_report(conn: PgConnection) -> List[ReportRow]:
    """Fetch report rows after the refresh transaction has committed."""
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(SELECT_REPORT_SQL)
            return cursor.fetchall()


def _emit_report(rows: Iterable[ReportRow], output: OutputFunc) -> None:
    """Render the asset strategy analytics report through an injectable output."""
    output("======================================")
    output("ASSET STRATEGY ANALYTICS ENGINE")
    output("======================================")

    for row in rows:
        symbol, strategy, regime, trades, win_rate, avg_pnl, total_pnl, rec = row
        output(
            f"{symbol} | strategy={strategy} | regime={regime} | "
            f"trades={trades} | win_rate={win_rate:.2f}% | "
            f"avg_pnl=${avg_pnl:.2f} | total_pnl=${total_pnl:.2f} | "
            f"{rec}"
        )


def _close_connection(conn: PgConnection) -> None:
    """Close a PostgreSQL connection, logging cleanup failures structurally."""
    try:
        conn.close()
    except Exception:
        logger.exception(
            "Failed to close PostgreSQL connection.",
            extra={"engine": "asset_strategy_analytics"},
        )
        raise


def main(
    connection_factory: ConnectionFactory = _default_connection_factory,
    output: OutputFunc = print,
) -> None:
    """Run the asset strategy analytics refresh and report.

    ``connection_factory`` and ``output`` are injectable so integration tests can
    provide a controlled database connection and capture report lines without
    monkeypatching ``psycopg2.connect``, ``DB_CONFIG``, or stdout.
    """
    conn: Optional[PgConnection] = None

    try:
        conn = connection_factory()
        _refresh_performance(conn)
        report_rows = _fetch_report(conn)
        _emit_report(report_rows, output)
    except InvalidSourceDataError:
        logger.exception(
            "Invalid source data prevented asset strategy analytics refresh.",
            extra={"engine": "asset_strategy_analytics"},
        )
        raise
    except psycopg2.Error:
        logger.exception(
            "Database error during asset strategy analytics refresh.",
            extra={"engine": "asset_strategy_analytics"},
        )
        raise
    finally:
        if conn is not None:
            active_error = sys.exc_info()[0] is not None
            try:
                _close_connection(conn)
            except Exception:
                if not active_error:
                    raise


if __name__ == "__main__":
    main()
