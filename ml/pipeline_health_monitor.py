from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import psycopg2

from configs.database import DB_CONFIG


STALE_LIMIT_MINUTES = 5
MIN_OHLC_ROWS = 100
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

HealthResult = Tuple[
    str,
    int,
    Optional[datetime],
    Optional[float],
    int,
    Optional[datetime],
    Optional[float],
    int,
    int,
    int,
    str,
    str,
]

CREATE_HEALTH_MONITOR_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS pipeline_health_monitor (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT,
        market_trades_rows BIGINT,
        latest_trade_time TIMESTAMPTZ,
        trade_minutes_old DOUBLE PRECISION,
        quant_metrics_rows BIGINT,
        latest_metric_time TIMESTAMPTZ,
        metric_minutes_old DOUBLE PRECISION,
        ohlc_rows BIGINT,
        momentum_rows BIGINT,
        live_signal_rows BIGINT,
        health_status TEXT,
        health_comment TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

MARKET_TRADES_QUERY = """
    SELECT COUNT(*), MAX(event_time)
    FROM market_trades
    WHERE symbol = %s;
"""

QUANT_METRICS_QUERY = """
    SELECT COUNT(*), MAX(metric_time)
    FROM quant_metrics
    WHERE symbol = %s;
"""

OHLC_COUNT_QUERY = """
    SELECT COUNT(*)
    FROM ohlc_1m_bars
    WHERE symbol = %s;
"""

MOMENTUM_COUNT_QUERY = """
    SELECT COUNT(*)
    FROM bar_momentum_features
    WHERE symbol = %s;
"""

LIVE_SIGNAL_COUNT_QUERY = """
    SELECT COUNT(*)
    FROM live_signal_history
    WHERE symbol = %s;
"""

INSERT_HEALTH_ROWS_SQL = """
    INSERT INTO pipeline_health_monitor (
        symbol,
        market_trades_rows,
        latest_trade_time,
        trade_minutes_old,
        quant_metrics_rows,
        latest_metric_time,
        metric_minutes_old,
        ohlc_rows,
        momentum_rows,
        live_signal_rows,
        health_status,
        health_comment
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""


def get_status(minutes_old: Optional[float], row_count: int, min_rows: int) -> str:
    if minutes_old is None:
        return "NO_DATA"

    if minutes_old > STALE_LIMIT_MINUTES:
        return "STALE"

    if row_count < min_rows:
        return "WARMING_UP"

    return "OK"


def _minutes_since(now: datetime, timestamp: Optional[datetime]) -> Optional[float]:
    if timestamp is None:
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return (now - timestamp).total_seconds() / 60


def _fetch_count_and_latest(
    cursor: Any,
    query: str,
    symbol: str,
) -> Tuple[int, Optional[datetime]]:
    cursor.execute(query, (symbol,))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Expected aggregate result for symbol {symbol}, got no row.")

    count, latest_timestamp = row
    return int(count or 0), latest_timestamp


def _fetch_count(cursor: Any, query: str, symbol: str) -> int:
    cursor.execute(query, (symbol,))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Expected count result for symbol {symbol}, got no row.")

    return int(row[0] or 0)


def _build_health_result(cursor: Any, symbol: str, now: datetime) -> HealthResult:
    trade_rows, latest_trade_time = _fetch_count_and_latest(
        cursor,
        MARKET_TRADES_QUERY,
        symbol,
    )
    metric_rows, latest_metric_time = _fetch_count_and_latest(
        cursor,
        QUANT_METRICS_QUERY,
        symbol,
    )
    ohlc_rows = _fetch_count(cursor, OHLC_COUNT_QUERY, symbol)
    momentum_rows = _fetch_count(cursor, MOMENTUM_COUNT_QUERY, symbol)
    live_signal_rows = _fetch_count(cursor, LIVE_SIGNAL_COUNT_QUERY, symbol)

    trade_minutes_old = _minutes_since(now, latest_trade_time)
    metric_minutes_old = _minutes_since(now, latest_metric_time)

    trade_status = get_status(trade_minutes_old, trade_rows, 1)
    metric_status = get_status(metric_minutes_old, metric_rows, 1)

    if trade_status == "STALE":
        health_status = "STALE_TRADES"
        health_comment = "Market trade feed is stale. Restart Binance ingestion."
    elif metric_status == "STALE":
        health_status = "STALE_METRICS"
        health_comment = "Quant metrics are stale. Restart rolling volatility engine."
    elif ohlc_rows < MIN_OHLC_ROWS:
        health_status = "WARMING_UP"
        health_comment = "Symbol is collecting enough OHLC history."
    elif momentum_rows < MIN_OHLC_ROWS:
        health_status = "WARMING_UP"
        health_comment = "Symbol is collecting enough momentum history."
    else:
        health_status = "OK"
        health_comment = "Pipeline is healthy for this symbol."

    return (
        symbol,
        trade_rows,
        latest_trade_time,
        float(trade_minutes_old) if trade_minutes_old is not None else None,
        metric_rows,
        latest_metric_time,
        float(metric_minutes_old) if metric_minutes_old is not None else None,
        ohlc_rows,
        momentum_rows,
        live_signal_rows,
        health_status,
        health_comment,
    )


def _print_report(results: list[HealthResult]) -> None:
    print("==============================")
    print("PIPELINE HEALTH MONITOR")
    print("==============================")

    for row in results:
        print(
            f"{row[0]} | "
            f"trades={row[1]} | "
            f"metrics={row[4]} | "
            f"ohlc={row[7]} | "
            f"momentum={row[8]} | "
            f"status={row[10]} | "
            f"{row[11]}"
        )


def main() -> None:
    results: list[HealthResult] = []
    conn = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_HEALTH_MONITOR_TABLE_SQL)

                now = datetime.now(timezone.utc)
                results = [
                    _build_health_result(cursor, symbol, now)
                    for symbol in SYMBOLS
                ]

                cursor.executemany(INSERT_HEALTH_ROWS_SQL, results)
    finally:
        if conn is not None:
            conn.close()

    _print_report(results)


if __name__ == "__main__":
    main()
