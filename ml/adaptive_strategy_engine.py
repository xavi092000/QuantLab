from typing import Optional, Sequence, Tuple

import psycopg2
from psycopg2.extensions import connection as PsycopgConnection

from configs.database import DB_CONFIG


HIGH_RISK_REGIMES = {
    "LIQUIDITY_EVENT",
    "STATISTICAL_ANOMALY",
    "VWAP_DISLOCATION",
}

REGIME_THRESHOLDS = {
    "NORMAL": (-2.5, 40.0),
    "BEARISH_MOMENTUM": (-2.0, 45.0),
    "BULLISH_MOMENTUM": (-3.0, 35.0),
    "VOLATILE_MOMENTUM": (-2.75, 35.0),
}

RECREATE_SIGNALS_TABLE_SQL = """
    DROP TABLE IF EXISTS adaptive_strategy_signals;

    CREATE TABLE adaptive_strategy_signals (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT,
        metric_time TIMESTAMPTZ,
        rsi DOUBLE PRECISION,
        z_score DOUBLE PRECISION,
        market_regime TEXT,
        z_threshold DOUBLE PRECISION,
        rsi_threshold DOUBLE PRECISION,
        adaptive_signal TEXT,
        signal_quality_score DOUBLE PRECISION,
        decision_reason TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

LATEST_METRICS_SQL = """
    SELECT DISTINCT ON (symbol)
        symbol,
        metric_time,
        rsi,
        z_score,
        market_regime
    FROM quant_metrics
    WHERE rsi IS NOT NULL
      AND z_score IS NOT NULL
      AND market_regime IS NOT NULL
    ORDER BY symbol, metric_time DESC;
"""

INSERT_SIGNAL_SQL = """
    INSERT INTO adaptive_strategy_signals (
        symbol,
        metric_time,
        rsi,
        z_score,
        market_regime,
        z_threshold,
        rsi_threshold,
        adaptive_signal,
        signal_quality_score,
        decision_reason
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

REPORT_SIGNALS_SQL = """
    SELECT
        symbol,
        ROUND(rsi::numeric, 2),
        ROUND(z_score::numeric, 3),
        market_regime,
        z_threshold,
        rsi_threshold,
        adaptive_signal,
        ROUND(signal_quality_score::numeric, 2),
        decision_reason
    FROM adaptive_strategy_signals
    ORDER BY signal_quality_score DESC;
"""


def get_thresholds(market_regime: str) -> Tuple[Optional[float], Optional[float]]:
    """Return adaptive z-score and RSI thresholds for a market regime."""
    return REGIME_THRESHOLDS.get(market_regime, (None, None))


def evaluate_adaptive_signal(
    market_regime: str,
    z_score: float,
    rsi: float,
) -> Tuple[Optional[float], Optional[float], str, float, str]:
    """Evaluate the adaptive trading signal for one latest metric row."""
    adaptive_signal = "AVOID"
    signal_quality_score = 0.0
    decision_reason = "Adaptive strategy conditions not met"

    z_threshold, rsi_threshold = get_thresholds(market_regime)

    if market_regime in HIGH_RISK_REGIMES:
        adaptive_signal = "NO_TRADE"
        decision_reason = f"High-risk regime blocks trading: {market_regime}"

    elif z_threshold is None or rsi_threshold is None:
        adaptive_signal = "AVOID"
        decision_reason = f"No adaptive rule defined for regime: {market_regime}"

    else:
        z_condition_met = z_score < z_threshold
        rsi_condition_met = rsi < rsi_threshold

        if z_condition_met:
            signal_quality_score += 45

        if rsi_condition_met:
            signal_quality_score += 45

        if z_condition_met and rsi_condition_met:
            signal_quality_score += 10

        if signal_quality_score >= 90:
            adaptive_signal = "BUY"
            decision_reason = (
                f"Adaptive BUY: z_score {z_score:.3f} < {z_threshold} "
                f"and RSI {rsi:.2f} < {rsi_threshold}"
            )

        elif signal_quality_score >= 45:
            adaptive_signal = "WATCH"
            decision_reason = (
                f"Partial setup: z_score threshold={z_threshold}, "
                f"RSI threshold={rsi_threshold}"
            )

    return (
        z_threshold,
        rsi_threshold,
        adaptive_signal,
        signal_quality_score,
        decision_reason,
    )


def rebuild_adaptive_strategy_signals(conn: PsycopgConnection) -> int:
    """Recreate and populate adaptive_strategy_signals from latest quant_metrics."""
    with conn.cursor() as cursor:
        cursor.execute(RECREATE_SIGNALS_TABLE_SQL)
        cursor.execute(LATEST_METRICS_SQL)
        rows: Sequence[tuple] = cursor.fetchall()

        inserted = 0
        for symbol, metric_time, rsi_value, z_score_value, market_regime in rows:
            rsi = float(rsi_value)
            z_score = float(z_score_value)

            (
                z_threshold,
                rsi_threshold,
                adaptive_signal,
                signal_quality_score,
                decision_reason,
            ) = evaluate_adaptive_signal(market_regime, z_score, rsi)

            cursor.execute(
                INSERT_SIGNAL_SQL,
                (
                    symbol,
                    metric_time,
                    rsi,
                    z_score,
                    market_regime,
                    z_threshold,
                    rsi_threshold,
                    adaptive_signal,
                    signal_quality_score,
                    decision_reason,
                ),
            )
            inserted += 1

    return inserted


def print_report(conn: PsycopgConnection, inserted: int) -> None:
    print("==============================")
    print("ADAPTIVE STRATEGY ENGINE")
    print("==============================")
    print("Rows processed:", inserted)

    with conn.cursor() as cursor:
        cursor.execute(REPORT_SIGNALS_SQL)
        for row in cursor.fetchall():
            print(
                f"{row[0]} | RSI={row[1]} | Z={row[2]} | regime={row[3]} | "
                f"z_thr={row[4]} | rsi_thr={row[5]} | signal={row[6]} | "
                f"quality={row[7]} | reason={row[8]}"
            )


def main() -> None:
    conn: Optional[PsycopgConnection] = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        inserted = rebuild_adaptive_strategy_signals(conn)
        conn.commit()
        print_report(conn, inserted)
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
