from __future__ import annotations

import sys
from contextlib import closing
from dataclasses import dataclass
from typing import Sequence, Tuple

import psycopg2

from configs.database import DB_CONFIG


CREATE_TABLE_SQL = """
    CREATE TABLE final_strategy_decisions (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT,
        market_regime TEXT,
        selected_strategy TEXT,
        adaptive_signal TEXT,
        momentum_signal TEXT,
        predicted_return_5m DOUBLE PRECISION,
        probability_up DOUBLE PRECISION,
        ml_vote TEXT,
        final_decision TEXT,
        decision_reason TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

SOURCE_SIGNALS_SQL = """
    SELECT
        r.symbol,
        r.market_regime,
        r.selected_strategy,
        COALESCE(a.adaptive_signal, 'NONE') AS adaptive_signal,
        COALESCE(m.momentum_signal, 'NONE') AS momentum_signal,
        COALESCE(l.predicted_return_5m, 0) AS predicted_return_5m,
        COALESCE(l.probability_up, 0.5) AS probability_up
    FROM strategy_router_results r
    LEFT JOIN adaptive_strategy_signals a
        ON r.symbol = a.symbol
    LEFT JOIN momentum_strategy_signals m
        ON r.symbol = m.symbol
    LEFT JOIN live_return_signals l
        ON r.symbol = l.symbol
    WHERE l.created_at = (
        SELECT MAX(created_at)
        FROM live_return_signals
    )
    ORDER BY r.symbol;
"""

INSERT_DECISION_SQL = """
    INSERT INTO final_strategy_decisions (
        symbol,
        market_regime,
        selected_strategy,
        adaptive_signal,
        momentum_signal,
        predicted_return_5m,
        probability_up,
        ml_vote,
        final_decision,
        decision_reason
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

REPORT_SQL = """
    SELECT
        symbol,
        selected_strategy,
        adaptive_signal,
        momentum_signal,
        ROUND(predicted_return_5m::numeric, 8),
        ROUND(probability_up::numeric, 4),
        ml_vote,
        final_decision,
        decision_reason
    FROM final_strategy_decisions
    ORDER BY
        CASE
            WHEN final_decision = 'BUY' THEN 1
            WHEN final_decision = 'WATCH' THEN 2
            WHEN final_decision = 'AVOID' THEN 3
            ELSE 4
        END,
        symbol;
"""


@dataclass(frozen=True)
class SourceSignal:
    symbol: str
    market_regime: str
    selected_strategy: str
    adaptive_signal: str
    momentum_signal: str
    predicted_return_5m: float
    probability_up: float


@dataclass(frozen=True)
class StrategyDecision:
    symbol: str
    market_regime: str
    selected_strategy: str
    adaptive_signal: str
    momentum_signal: str
    predicted_return_5m: float
    probability_up: float
    ml_vote: str
    final_decision: str
    decision_reason: str

    def as_insert_tuple(self) -> Tuple:
        return (
            self.symbol,
            self.market_regime,
            self.selected_strategy,
            self.adaptive_signal,
            self.momentum_signal,
            self.predicted_return_5m,
            self.probability_up,
            self.ml_vote,
            self.final_decision,
            self.decision_reason,
        )


def determine_ml_vote(
    predicted_return_5m: float,
    probability_up: float,
) -> str:
    if probability_up >= 0.60 and predicted_return_5m >= 0.0002:
        return "STRONG_SUPPORT"

    if probability_up >= 0.60 and predicted_return_5m >= -0.0001:
        return "SUPPORT"

    if probability_up >= 0.50:
        return "NEUTRAL"

    return "AGAINST"


def determine_final_decision(
    selected_strategy: str,
    adaptive_signal: str,
    momentum_signal: str,
    ml_vote: str,
) -> Tuple[str, str]:
    supportive_votes = {"STRONG_SUPPORT", "SUPPORT"}
    non_veto_votes = supportive_votes | {"NEUTRAL"}

    if selected_strategy == "MOMENTUM":
        if momentum_signal == "BUY" and ml_vote in supportive_votes:
            return "BUY", "Momentum BUY confirmed by hybrid ML vote"
        if momentum_signal == "BUY" and ml_vote in non_veto_votes:
            return "WATCH", "Momentum BUY detected; ML confidence is moderate"
        if momentum_signal == "WATCH":
            return "WATCH", "Momentum setup is developing"
        return "AVOID", "Momentum strategy not confirmed"

    if selected_strategy == "MEAN_REVERSION":
        if adaptive_signal == "BUY" and ml_vote in supportive_votes:
            return "BUY", "Mean reversion BUY confirmed by hybrid ML vote"
        if adaptive_signal == "BUY" and ml_vote in non_veto_votes:
            return "WATCH", "Mean reversion BUY detected; ML confidence is moderate"
        if adaptive_signal == "WATCH":
            return "WATCH", "Mean reversion setup is close"
        return "AVOID", "Mean reversion strategy not confirmed"

    if selected_strategy == "SHORT_MOMENTUM":
        if adaptive_signal == "BUY" and ml_vote == "AGAINST":
            return (
                "WATCH",
                "Oversold rebound detected inside bearish regime; review manually",
            )
        if adaptive_signal == "WATCH":
            return "WATCH", "Bearish momentum setup is evolving"
        return "AVOID", "Short-momentum strategy not confirmed"

    if selected_strategy == "NO_TRADE":
        return "NO_TRADE", "Router selected NO_TRADE"

    return "AVOID", "No strategy confirmation"


def source_signal_from_row(row: Sequence[object]) -> SourceSignal:
    return SourceSignal(
        symbol=str(row[0]),
        market_regime=str(row[1]),
        selected_strategy=str(row[2]),
        adaptive_signal=str(row[3]),
        momentum_signal=str(row[4]),
        predicted_return_5m=float(row[5]),
        probability_up=float(row[6]),
    )


def build_decision(signal: SourceSignal) -> StrategyDecision:
    ml_vote = determine_ml_vote(
        signal.predicted_return_5m,
        signal.probability_up,
    )

    final_decision, decision_reason = determine_final_decision(
        selected_strategy=signal.selected_strategy,
        adaptive_signal=signal.adaptive_signal,
        momentum_signal=signal.momentum_signal,
        ml_vote=ml_vote,
    )

    return StrategyDecision(
        symbol=signal.symbol,
        market_regime=signal.market_regime,
        selected_strategy=signal.selected_strategy,
        adaptive_signal=signal.adaptive_signal,
        momentum_signal=signal.momentum_signal,
        predicted_return_5m=signal.predicted_return_5m,
        probability_up=signal.probability_up,
        ml_vote=ml_vote,
        final_decision=final_decision,
        decision_reason=decision_reason,
    )


def rebuild_final_strategy_decisions(conn) -> int:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS final_strategy_decisions;")
        cursor.execute(CREATE_TABLE_SQL)
        cursor.execute(SOURCE_SIGNALS_SQL)

        decisions = [
            build_decision(source_signal_from_row(row))
            for row in cursor.fetchall()
        ]

        if decisions:
            cursor.executemany(
                INSERT_DECISION_SQL,
                [decision.as_insert_tuple() for decision in decisions],
            )

    return len(decisions)


def print_decision_report(conn, inserted: int) -> None:
    print("==============================")
    print("FINAL STRATEGY DECISION ENGINE")
    print("==============================")
    print("Rows processed:", inserted)

    with conn.cursor() as cursor:
        cursor.execute(REPORT_SQL)
        rows = cursor.fetchall()

    for row in rows:
        print(
            f"{row[0]} | strategy={row[1]} | adaptive={row[2]} | "
            f"momentum={row[3]} | pred={row[4]} | p_up={row[5]} | "
            f"ml={row[6]} | final={row[7]} | reason={row[8]}"
        )


def main() -> None:
    try:
        with closing(psycopg2.connect(**DB_CONFIG)) as conn:
            with conn:
                inserted = rebuild_final_strategy_decisions(conn)

            print_decision_report(conn, inserted)
    except psycopg2.Error as exc:
        print(f"Final strategy decision engine failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
