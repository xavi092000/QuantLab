from configs.database import DB_CONFIG
import psycopg2

BUY_SCORE_THRESHOLD = 75
WATCH_SCORE_THRESHOLD = 50


def normalize(value, min_value, max_value):
    if value is None:
        return 0.0

    if max_value == min_value:
        return 0.0

    return (value - min_value) / (max_value - min_value)


def regime_score(market_regime):
    if market_regime in ["LIQUIDITY_EVENT", "STATISTICAL_ANOMALY", "VWAP_DISLOCATION"]:
        return 0.0

    if market_regime in ["VOLATILE_MOMENTUM"]:
        return 0.5

    return 1.0


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS opportunity_ranking_engine;")

    cursor.execute("""
        CREATE TABLE opportunity_ranking_engine (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            probability_up DOUBLE PRECISION,
            predicted_return_5m DOUBLE PRECISION,
            liquidity DOUBLE PRECISION,
            market_regime TEXT,
            normalized_probability DOUBLE PRECISION,
            normalized_return DOUBLE PRECISION,
            normalized_liquidity DOUBLE PRECISION,
            normalized_regime DOUBLE PRECISION,
            opportunity_score DOUBLE PRECISION,
            combined_decision TEXT,
            recommendation TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            c.symbol,
            c.probability_up,
            c.predicted_return_5m,
            COALESCE(x.liquidity, 0) AS liquidity,
            c.market_regime,
            c.combined_decision
        FROM combined_signal_engine c
        LEFT JOIN cross_asset_ml_rankings x
            ON c.symbol = x.symbol;
    """)

    rows = cursor.fetchall()

    if not rows:
        print("[ERROR] No rows found from combined_signal_engine.")
        cursor.close()
        conn.close()
        return

    probabilities = [float(r[1]) for r in rows]
    returns = [float(r[2]) for r in rows]
    liquidities = [float(r[3]) for r in rows]

    min_probability = min(probabilities)
    max_probability = max(probabilities)

    min_return = min(returns)
    max_return = max(returns)

    min_liquidity = min(liquidities)
    max_liquidity = max(liquidities)

    inserted = 0

    for row in rows:
        symbol = row[0]
        probability_up = float(row[1])
        predicted_return = float(row[2])
        liquidity = float(row[3])
        market_regime = row[4]
        combined_decision = row[5]

        normalized_probability = normalize(
            probability_up,
            min_probability,
            max_probability
        )

        normalized_return = normalize(
            predicted_return,
            min_return,
            max_return
        )

        normalized_liquidity = normalize(
            liquidity,
            min_liquidity,
            max_liquidity
        )

        normalized_regime = regime_score(market_regime)

        opportunity_score = (
            normalized_probability * 35
            + normalized_return * 35
            + normalized_liquidity * 20
            + normalized_regime * 10
        )

        if combined_decision == "BUY" and opportunity_score >= BUY_SCORE_THRESHOLD:
            recommendation = "BUY"
        elif opportunity_score >= WATCH_SCORE_THRESHOLD:
            recommendation = "WATCH"
        else:
            recommendation = "AVOID"

        cursor.execute("""
            INSERT INTO opportunity_ranking_engine (
                symbol,
                probability_up,
                predicted_return_5m,
                liquidity,
                market_regime,
                normalized_probability,
                normalized_return,
                normalized_liquidity,
                normalized_regime,
                opportunity_score,
                combined_decision,
                recommendation
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            probability_up,
            predicted_return,
            liquidity,
            market_regime,
            normalized_probability,
            normalized_return,
            normalized_liquidity,
            normalized_regime,
            opportunity_score,
            combined_decision,
            recommendation,
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("OPPORTUNITY RANKING ENGINE V2")
    print("==============================")
    print("Assets Ranked:", inserted)

    cursor.execute("""
        SELECT
            symbol,
            ROUND(opportunity_score::numeric, 2),
            combined_decision,
            recommendation
        FROM opportunity_ranking_engine
        ORDER BY opportunity_score DESC;
    """)

    results = cursor.fetchall()

    for r in results:
        print(
            f"{r[0]} | "
            f"score={r[1]} | "
            f"combined={r[2]} | "
            f"recommendation={r[3]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


