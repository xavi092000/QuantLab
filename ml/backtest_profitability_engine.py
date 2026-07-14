import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import psycopg2
from psycopg2.extensions import connection as PgConnection, cursor as PgCursor

from configs.database import DB_CONFIG


BUY_RETURN_THRESHOLD = 0.0002
POSITION_SIZE_USD = 10000.0
ENV_BUY_RETURN_THRESHOLD = "QUANTLAB_BUY_RETURN_THRESHOLD"
ENV_POSITION_SIZE_USD = "QUANTLAB_POSITION_SIZE_USD"


@dataclass(frozen=True)
class BacktestConfig:
    """Runtime configuration for the profitability backtest."""

    buy_return_threshold: float = BUY_RETURN_THRESHOLD
    position_size_usd: float = POSITION_SIZE_USD

    def __post_init__(self) -> None:
        """Normalize numeric values and reject unsafe runtime configuration."""
        buy_return_threshold = float(self.buy_return_threshold)
        position_size_usd = float(self.position_size_usd)

        if not math.isfinite(buy_return_threshold):
            raise ValueError("buy_return_threshold must be a finite number.")
        if not math.isfinite(position_size_usd) or position_size_usd <= 0:
            raise ValueError("position_size_usd must be a positive finite number.")

        object.__setattr__(self, "buy_return_threshold", buy_return_threshold)
        object.__setattr__(self, "position_size_usd", position_size_usd)

    @classmethod
    def from_values(
        cls,
        buy_return_threshold: Optional[float] = None,
        position_size_usd: Optional[float] = None,
    ) -> "BacktestConfig":
        """Build a config, falling back to module defaults for omitted values."""
        return cls(
            buy_return_threshold=(
                BUY_RETURN_THRESHOLD
                if buy_return_threshold is None
                else buy_return_threshold
            ),
            position_size_usd=(
                POSITION_SIZE_USD if position_size_usd is None else position_size_usd
            ),
        )


class ProfitabilitySummary(NamedTuple):
    """Aggregate profitability metrics persisted and printed after a run."""

    trades: int
    winners: int
    win_rate_pct: float
    avg_return: float
    total_pnl_usd: float
    worst_trade_usd: float
    best_trade_usd: float


DROP_RESULTS_TABLE_SQL = """
    DROP TABLE IF EXISTS backtest_profitability_results;
"""

CREATE_RESULTS_TABLE_SQL = """
    CREATE TABLE backtest_profitability_results (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT,
        signal_time TIMESTAMPTZ,
        future_return_5m DOUBLE PRECISION,
        position_size_usd DOUBLE PRECISION,
        pnl_usd DOUBLE PRECISION,
        profitable BOOLEAN,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

INSERT_RESULTS_SQL = """
    WITH params AS (
        SELECT %s::DOUBLE PRECISION AS position_size_usd
    ),
    candidate_signals AS (
        SELECT
            symbol,
            signal_time,
            future_return_5m::DOUBLE PRECISION AS future_return_5m
        FROM signal_validation
        WHERE future_return_5m IS NOT NULL
          AND future_return_5m > %s::DOUBLE PRECISION
    )
    INSERT INTO backtest_profitability_results (
        symbol,
        signal_time,
        future_return_5m,
        position_size_usd,
        pnl_usd,
        profitable
    )
    SELECT
        candidate_signals.symbol,
        candidate_signals.signal_time,
        candidate_signals.future_return_5m,
        params.position_size_usd,
        params.position_size_usd * candidate_signals.future_return_5m AS pnl_usd,
        (params.position_size_usd * candidate_signals.future_return_5m) > 0 AS profitable
    FROM candidate_signals
    CROSS JOIN params;
"""

SELECT_SUMMARY_SQL = """
    SELECT
        COUNT(*) AS trades,
        SUM(CASE WHEN profitable THEN 1 ELSE 0 END) AS winners,
        AVG(future_return_5m) AS avg_return,
        SUM(pnl_usd) AS total_pnl,
        MIN(pnl_usd) AS worst_trade,
        MAX(pnl_usd) AS best_trade
    FROM backtest_profitability_results;
"""

CREATE_SUMMARY_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS backtest_profitability_summary (
        id BIGSERIAL PRIMARY KEY,
        buy_return_threshold DOUBLE PRECISION,
        position_size_usd DOUBLE PRECISION,
        trades INTEGER,
        winners INTEGER,
        win_rate_pct DOUBLE PRECISION,
        avg_return DOUBLE PRECISION,
        total_pnl_usd DOUBLE PRECISION,
        worst_trade_usd DOUBLE PRECISION,
        best_trade_usd DOUBLE PRECISION,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

INSERT_SUMMARY_SQL = """
    INSERT INTO backtest_profitability_summary (
        buy_return_threshold,
        position_size_usd,
        trades,
        winners,
        win_rate_pct,
        avg_return,
        total_pnl_usd,
        worst_trade_usd,
        best_trade_usd
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""


def _rebuild_profitability_results(cursor: PgCursor, config: BacktestConfig) -> None:
    """
    Recreate and populate per-signal profitability results.

    This intentionally preserves the original destructive lifecycle: the
    backtest_profitability_results table is dropped and recreated on every run
    so the table reflects only the latest backtest execution.
    """
    cursor.execute(DROP_RESULTS_TABLE_SQL)
    cursor.execute(CREATE_RESULTS_TABLE_SQL)
    cursor.execute(
        INSERT_RESULTS_SQL,
        (config.position_size_usd, config.buy_return_threshold),
    )


def _fetch_profitability_summary(cursor: PgCursor) -> ProfitabilitySummary:
    """Calculate aggregate profitability metrics from rebuilt results."""
    cursor.execute(SELECT_SUMMARY_SQL)
    row = cursor.fetchone()

    if row is None:
        raise RuntimeError("Failed to calculate backtest profitability summary.")

    trades = int(row[0] or 0)
    winners = int(row[1] or 0)
    avg_return = float(row[2] or 0)
    total_pnl = float(row[3] or 0)
    worst_trade = float(row[4] or 0)
    best_trade = float(row[5] or 0)
    win_rate = (winners / trades * 100) if trades > 0 else 0.0

    return ProfitabilitySummary(
        trades=trades,
        winners=winners,
        win_rate_pct=win_rate,
        avg_return=avg_return,
        total_pnl_usd=total_pnl,
        worst_trade_usd=worst_trade,
        best_trade_usd=best_trade,
    )


def _persist_profitability_summary(
    cursor: PgCursor,
    config: BacktestConfig,
    summary: ProfitabilitySummary,
) -> None:
    """Ensure the summary table exists and append the latest summary row."""
    cursor.execute(CREATE_SUMMARY_TABLE_SQL)
    cursor.execute(
        INSERT_SUMMARY_SQL,
        (
            config.buy_return_threshold,
            config.position_size_usd,
            summary.trades,
            summary.winners,
            summary.win_rate_pct,
            summary.avg_return,
            summary.total_pnl_usd,
            summary.worst_trade_usd,
            summary.best_trade_usd,
        ),
    )


def run_backtest(
    config: BacktestConfig,
    db_config: Optional[Mapping[str, Any]] = None,
) -> ProfitabilitySummary:
    """
    Execute the profitability backtest and return its aggregate summary.

    The results rebuild and summary insert run in one PostgreSQL transaction so
    consumers do not observe a freshly rebuilt results table without its matching
    summary row. PostgreSQL transactional DDL preserves rollback safety for the
    DROP/CREATE lifecycle used by this module.
    """
    conn: Optional[PgConnection] = None
    effective_db_config = DB_CONFIG if db_config is None else db_config

    try:
        conn = psycopg2.connect(**dict(effective_db_config))
        with conn:
            with conn.cursor() as cursor:
                _rebuild_profitability_results(cursor, config)
                summary = _fetch_profitability_summary(cursor)
                _persist_profitability_summary(cursor, config, summary)
        return summary
    finally:
        if conn is not None:
            conn.close()


def _print_report(config: BacktestConfig, summary: ProfitabilitySummary) -> None:
    """Render the CLI report for a completed profitability backtest."""
    print("==============================")
    print("BACKTEST PROFITABILITY ENGINE")
    print("==============================")
    print("Threshold      :", config.buy_return_threshold)
    print("Position Size  :", config.position_size_usd)
    print("Trades         :", summary.trades)
    print("Winners        :", summary.winners)
    print("Win Rate %     :", round(summary.win_rate_pct, 2))
    print("Avg Return     :", round(summary.avg_return, 6))
    print("Total PnL USD  :", round(summary.total_pnl_usd, 2))
    print("Worst Trade USD:", round(summary.worst_trade_usd, 2))
    print("Best Trade USD :", round(summary.best_trade_usd, 2))


def _read_optional_env_float(name: str) -> Optional[float]:
    """Read an optional float environment variable for CLI configuration."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid float.") from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser while keeping defaults testable."""
    parser = argparse.ArgumentParser(
        description="Rebuild and summarize backtest profitability results.",
    )
    parser.add_argument(
        "--buy-return-threshold",
        type=float,
        default=None,
        help=(
            "Minimum future_return_5m required for a simulated buy. "
            f"Defaults to {ENV_BUY_RETURN_THRESHOLD} or {BUY_RETURN_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--position-size-usd",
        type=float,
        default=None,
        help=(
            "USD notional used to calculate simulated PnL. "
            f"Defaults to {ENV_POSITION_SIZE_USD} or {POSITION_SIZE_USD}."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> BacktestConfig:
    """Resolve effective config with CLI arguments taking precedence over env."""
    buy_return_threshold = args.buy_return_threshold
    position_size_usd = args.position_size_usd

    if buy_return_threshold is None:
        buy_return_threshold = _read_optional_env_float(ENV_BUY_RETURN_THRESHOLD)
    if position_size_usd is None:
        position_size_usd = _read_optional_env_float(ENV_POSITION_SIZE_USD)

    return BacktestConfig.from_values(
        buy_return_threshold=buy_return_threshold,
        position_size_usd=position_size_usd,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for the backtest profitability engine."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    try:
        summary = run_backtest(config)
    except psycopg2.Error as exc:
        print(f"BACKTEST PROFITABILITY ENGINE DATABASE ERROR: {exc}", file=sys.stderr)
        raise

    _print_report(config, summary)


if __name__ == "__main__":
    main()
