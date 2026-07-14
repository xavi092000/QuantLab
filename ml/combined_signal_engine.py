from configs.database import DB_CONFIG
import psycopg2

BUY_PROBABILITY_THRESHOLD = 0.70
WATCH_PROBABILITY_THRESHOLD = 0.50
RETURN_THRESHOLD = 0.0002

HIGH_RISK_REGIMES = [
    "LIQUIDITY_EVENT",
    "STATISTICAL_ANOMALY",
    "VWAP_DISLOCATION",
]


def get_combined_decision(
    predicted_return,
    probability_up,
    market_regime,
    risk_decision,
):
    if risk_decision != "RESEARCH_PAPER_RISK_APPROVED":
        return "NO_TRADE", "Risk engine has not approved paper trading."

    if market_regime in HIGH_RISK_REGIMES:
        return "NO_TRADE", f"High-risk market regime: {market_regime}."

    if probability_up >= BUY_PROBABILITY_THRESHOLD and predicted_return >= RETURN_THRESHOLD:
        return "BUY", "Strong upside probability and expected return exceed thresholds."

    if probability_up >= WATCH_PROBABILITY_THRESHOLD:
        return "WATCH", "Upside probability is present, but BUY thresholds are not met."

    return "AVOID", "Upside probability and expected return are not strong enough."


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT risk_decision
        FROM risk_budgeting_v2
        ORDER BY created_at DESC
        LIMIT 1;
    """)

    risk_row = cursor.fetchone()

    if risk_row is None:
        print("[ERROR] No risk_budgeting_v2 row found.")
        cursor.close()
        conn.close()
        return

    risk_decision = risk_row[0]

    cursor.execute("DROP TABLE IF EXISTS combined_signal_engine;")

    cursor.execute("""
        CREATE TABLE combined_signal_engine (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            market_regime TEXT,
            predicted_return_5m DOUBLE PRECISION,
            probability_up DOUBLE PRECISION,
            probability_down DOUBLE PRECISION,
            return_signal TEXT,
            direction_signal TEXT,
            risk_decision TEXT,
            combined_decision TEXT,
            decision_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            r.symbol,
            r.market_regime,
            r.predicted_return_5m,
            r.research_signal AS return_signal,
            d.probability_up,
            d.probability_down,
            d.direction_signal
        FROM live_return_signals r
        JOIN live_direction_signals d
            ON r.symbol = d.symbol
        ORDER BY d.probability_up DESC;
    """)

    rows = cursor.fetchall()
    output_rows = []

    for row in rows:
        symbol = row[0]
        market_regime = row[1]
        predicted_return = float(row[2])
        return_signal = row[3]
        probability_up = float(row[4])
        probability_down = float(row[5])
        direction_signal = row[6]

        combined_decision, decision_reason = get_combined_decision(
            predicted_return,
            probability_up,
            market_regime,
            risk_decision,
        )

        output_rows.append((
            symbol,
            market_regime,
            predicted_return,
            probability_up,
            probability_down,
            return_signal,
            direction_signal,
            risk_decision,
            combined_decision,
            decision_reason,
        ))

    cursor.executemany("""
        INSERT INTO combined_signal_engine (
            symbol,
            market_regime,
            predicted_return_5m,
            probability_up,
            probability_down,
            return_signal,
            direction_signal,
            risk_decision,
            combined_decision,
            decision_reason
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, output_rows)

    conn.commit()

    print("==============================")
    print("COMBINED SIGNAL ENGINE V2")
    print("==============================")
    print("Rows processed:", len(output_rows))
    print("Risk decision :", risk_decision)
    print("BUY threshold :", BUY_PROBABILITY_THRESHOLD)
    print("Return threshold:", RETURN_THRESHOLD)

    for row in output_rows:
        print(
            f"{row[0]} | "
            f"regime={row[1]} | "
            f"pred_return={row[2]:.6f} | "
            f"prob_up={row[3] * 100:.2f}% | "
            f"decision={row[8]} | "
            f"reason={row[9]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


