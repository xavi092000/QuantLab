from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import psycopg2

from configs.database import DB_CONFIG


MIN_TRADES_TO_BLOCK = 3


CREATE_RESULTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS asset_strategy_selector_results (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT,
        selected_strategy TEXT,
        market_regime TEXT,
        final_decision_before TEXT,
        historical_trades INTEGER,
        historical_win_rate DOUBLE PRECISION,
        historical_total_pnl DOUBLE PRECISION,
        historical_recommendation TEXT,
        selector_decision TEXT,
        selector_reason TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

CLEAR_RESULTS_SQL = "DELETE FROM asset_strategy_selector_results;"

FETCH_DECISIONS_SQL = """
    SELECT
        symbol,
        selected_strategy,
        market_regime,
        final_decision
    FROM final_strategy_decisions;
"""

FETCH_PERFORMANCE_SQL = """
    SELECT
        trades,
        win_rate,
        total_pnl,
        recommendation
    FROM asset_strategy_performance
    WHERE symbol = %s
      AND selected_strategy = %s
      AND market_regime = %s;
"""

INSERT_RESULT_SQL = """
    INSERT INTO asset_strategy_selector_results (
        symbol,
        selected_strategy,
        market_regime,
        final_decision_before,
        historical_trades,
        historical_win_rate,
        historical_total_pnl,
        historical_recommendation,
        selector_decision,
        selector_reason
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

FETCH_RESULTS_SQL = """
    SELECT
        symbol,
        selected_strategy,
        market_regime,
        final_decision_before,
        historical_trades,
        historical_recommendation,
        selector_decision,
        selector_reason
    FROM asset_strategy_selector_results
    ORDER BY symbol;
"""


@dataclass(frozen=True)
class DecisionInput:
    symbol: Optional[str]
    selected_strategy: str
    market_regime: str
    final_decision: Optional[str]


@dataclass(frozen=True)
class HistoricalPerformance:
    has_history: bool
    trades: int
    win_rate: float
    total_pnl: float
    recommendation: Optional[str]


@dataclass(frozen=True)
class SelectorOutcome:
    decision: Optional[str]
    reason: str


def _decision_from_row(row: Tuple[Any, ...]) -> DecisionInput:
    return DecisionInput(
        symbol=row[0],
        selected_strategy=row[1] or "UNKNOWN",
        market_regime=row[2] or "UNKNOWN",
        final_decision=row[3],
    )


def _history_from_row(row: Optional[Tuple[Any, ...]]) -> HistoricalPerformance:
    if row is None:
        return HistoricalPerformance(
            has_history=False,
            trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            recommendation="NO_HISTORY",
        )

    return HistoricalPerformance(
        has_history=True,
        trades=int(row[0]),
        win_rate=float(row[1]),
        total_pnl=float(row[2]),
        recommendation=row[3],
    )


def _select_outcome(
    final_decision: Optional[str],
    history: HistoricalPerformance,
) -> SelectorOutcome:
    if not history.has_history:
        return SelectorOutcome(
            decision=final_decision,
            reason="No historical asset-strategy-regime data yet",
        )

    if (
        final_decision == "BUY"
        and history.recommendation == "AVOID"
        and history.trades >= MIN_TRADES_TO_BLOCK
    ):
        return SelectorOutcome(
            decision="NO_TRADE",
            reason="Blocked by poor historical asset-strategy-regime performance",
        )

    return SelectorOutcome(
        decision=final_decision,
        reason="Historical performance does not block decision",
    )


def _fetch_decisions(cursor: Any) -> List[Tuple[Any, ...]]:
    cursor.execute(FETCH_DECISIONS_SQL)
    return cursor.fetchall()


def _fetch_historical_performance(
    cursor: Any,
    decision: DecisionInput,
) -> HistoricalPerformance:
    cursor.execute(
        FETCH_PERFORMANCE_SQL,
        (
            decision.symbol,
            decision.selected_strategy,
            decision.market_regime,
        ),
    )
    return _history_from_row(cursor.fetchone())


def _insert_selector_result(
    cursor: Any,
    decision: DecisionInput,
    history: HistoricalPerformance,
    outcome: SelectorOutcome,
) -> None:
    cursor.execute(
        INSERT_RESULT_SQL,
        (
            decision.symbol,
            decision.selected_strategy,
            decision.market_regime,
            decision.final_decision,
            history.trades,
            history.win_rate,
            history.total_pnl,
            history.recommendation,
            outcome.decision,
            outcome.reason,
        ),
    )


def _process_decision(cursor: Any, row: Tuple[Any, ...]) -> None:
    decision = _decision_from_row(row)
    history = _fetch_historical_performance(cursor, decision)
    outcome = _select_outcome(decision.final_decision, history)
    _insert_selector_result(cursor, decision, history, outcome)


def _print_results(cursor: Any) -> None:
    print("================================")
    print("ASSET STRATEGY SELECTOR")
    print("================================")

    cursor.execute(FETCH_RESULTS_SQL)

    for row in cursor.fetchall():
        print(
            f"{row[0]} | strategy={row[1]} | regime={row[2]} | "
            f"before={row[3]} | hist_trades={row[4]} | "
            f"hist_rec={row[5]} | selector={row[6]} | reason={row[7]}"
        )


def main() -> None:
    with closing(psycopg2.connect(**DB_CONFIG)) as conn:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_RESULTS_TABLE_SQL)
                cursor.execute(CLEAR_RESULTS_SQL)

                decisions = _fetch_decisions(cursor)
                for row in decisions:
                    _process_decision(cursor, row)

        with conn.cursor() as cursor:
            _print_results(cursor)


if __name__ == "__main__":
    main()
