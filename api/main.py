import logging
from contextlib import closing
from typing import Any, Dict, List, Mapping, Optional, Sequence

import psycopg2
from fastapi import FastAPI, HTTPException
from psycopg2.extras import RealDictCursor

from configs.database import DB_CONFIG


logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantLab API",
    description="API for QuantLab paper trading, portfolio, and risk monitoring",
    version="1.0.0",
)


OPEN_POSITIONS_QUERY = """
    SELECT
        position_id,
        symbol,
        selected_strategy,
        allocation_pct,
        entry_price,
        position_size_usd,
        stop_loss_price,
        take_profit_price,
        position_status,
        created_at
    FROM paper_positions
    WHERE position_status = 'OPEN'
    ORDER BY created_at DESC;
"""

LATEST_EQUITY_QUERY = """
    SELECT
        equity_value,
        realized_pnl,
        unrealized_pnl,
        open_positions,
        cash_remaining,
        created_at
    FROM portfolio_equity_curve
    ORDER BY created_at DESC
    LIMIT 1;
"""

LATEST_SIGNALS_QUERY = """
    SELECT
        symbol,
        market_regime,
        selected_strategy,
        adaptive_signal,
        momentum_signal,
        predicted_return_5m,
        ml_vote,
        final_decision,
        decision_reason,
        created_at
    FROM final_strategy_decisions
    ORDER BY created_at DESC;
"""

PERFORMANCE_QUERY = """
    SELECT
        COUNT(*) AS trades,
        COALESCE(SUM(pnl_usd), 0) AS total_pnl,
        COALESCE(AVG(pnl_usd), 0) AS avg_pnl,
        COALESCE(AVG(pnl_pct), 0) AS avg_pct
    FROM closed_paper_trades;
"""

CLOSED_TRADES_QUERY = """
    SELECT
        trade_id,
        symbol,
        selected_strategy,
        entry_price,
        exit_price,
        pnl_usd,
        pnl_pct,
        close_reason,
        holding_minutes
    FROM closed_paper_trades
    ORDER BY exit_time DESC;
"""

OPEN_EXPOSURE_QUERY = """
    SELECT
        COUNT(*) AS open_positions,
        COALESCE(SUM(position_size_usd), 0) AS total_exposure_usd,
        COALESCE(SUM(allocation_pct), 0) AS total_allocation_pct,
        COALESCE(SUM(position_size_usd * 0.02), 0) AS estimated_max_loss_usd
    FROM paper_positions
    WHERE position_status = 'OPEN';
"""

DASHBOARD_PERFORMANCE_QUERY = """
    SELECT
        COUNT(*) AS trades_closed,
        COUNT(*) FILTER (WHERE pnl_usd > 0) AS winners,
        COUNT(*) FILTER (WHERE pnl_usd < 0) AS losers,
        COALESCE(SUM(pnl_usd), 0) AS total_pnl,
        COALESCE(AVG(pnl_usd), 0) AS avg_pnl,
        COALESCE(AVG(pnl_pct), 0) AS avg_pct
    FROM closed_paper_trades;
"""


def get_connection():
    """Create a PostgreSQL connection using the project database configuration."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


def _as_dict(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _as_dict_list(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _database_error() -> HTTPException:
    return HTTPException(status_code=503, detail="Database query failed")


def fetch_all(query: str) -> List[Dict[str, Any]]:
    """Execute a read-only query and return all rows as plain dictionaries."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return _as_dict_list(cursor.fetchall())
    except psycopg2.Error as exc:
        logger.exception("Database query failed while fetching multiple rows.")
        raise _database_error() from exc


def fetch_one(query: str) -> Optional[Dict[str, Any]]:
    """Execute a read-only query and return one row as a plain dictionary."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return _as_dict(cursor.fetchone())
    except psycopg2.Error as exc:
        logger.exception("Database query failed while fetching one row.")
        raise _database_error() from exc


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "quantlab-api"}


@app.get("/positions/open")
def get_open_positions() -> Dict[str, Any]:
    rows = fetch_all(OPEN_POSITIONS_QUERY)
    return {"count": len(rows), "positions": rows}


@app.get("/portfolio/equity/latest")
def get_latest_equity() -> Dict[str, Any]:
    row = fetch_one(LATEST_EQUITY_QUERY)
    return {"latest_equity": row}


@app.get("/signals/latest")
def get_latest_signals() -> Dict[str, Any]:
    rows = fetch_all(LATEST_SIGNALS_QUERY)
    return {"count": len(rows), "signals": rows}


@app.get("/performance")
def get_performance() -> Optional[Dict[str, Any]]:
    return fetch_one(PERFORMANCE_QUERY)


@app.get("/portfolio/closed")
def get_closed_trades() -> Dict[str, Any]:
    rows = fetch_all(CLOSED_TRADES_QUERY)
    return {"count": len(rows), "closed_trades": rows}


@app.get("/risk/open-exposure")
def get_open_exposure() -> Optional[Dict[str, Any]]:
    return fetch_one(OPEN_EXPOSURE_QUERY)


@app.get("/dashboard")
def get_dashboard() -> Dict[str, Any]:
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(LATEST_EQUITY_QUERY)
                equity = _as_dict(cursor.fetchone())

                cursor.execute(OPEN_EXPOSURE_QUERY)
                risk = _as_dict(cursor.fetchone())

                cursor.execute(DASHBOARD_PERFORMANCE_QUERY)
                performance = _as_dict(cursor.fetchone()) or {}
    except psycopg2.Error as exc:
        logger.exception("Database query failed while building dashboard response.")
        raise _database_error() from exc

    trades_closed = performance.get("trades_closed") or 0
    winners = performance.get("winners") or 0

    win_rate_pct = winners / trades_closed * 100 if trades_closed > 0 else 0

    return {
        "system": "QuantLab",
        "status": "live",
        "equity": equity,
        "risk": risk,
        "performance": {**performance, "win_rate_pct": win_rate_pct},
    }
