from configs.database import DB_CONFIG
import psycopg2


ROUTING_TABLE = {
    "NORMAL": {
        "strategy": "MEAN_REVERSION",
        "reason": (
            "Stable market regime. "
            "Mean reversion preferred."
        ),
    },
    "BULLISH_MOMENTUM": {
        "strategy": "MOMENTUM",
        "reason": "Strong bullish trend detected.",
    },
    "BEARISH_MOMENTUM": {
        "strategy": "SHORT_MOMENTUM",
        "reason": "Strong bearish trend detected.",
    },
    "LIQUIDITY_EVENT": {
        "strategy": "NO_TRADE",
        "reason": "Risk regime blocks trading.",
    },
    "STATISTICAL_ANOMALY": {
        "strategy": "NO_TRADE",
        "reason": "Risk regime blocks trading.",
    },
    "VWAP_DISLOCATION": {
        "strategy": "NO_TRADE",
        "reason": "Risk regime blocks trading.",
    },
}

DEFAULT_ROUTE = {
    "strategy": "NO_TRADE",
    "reason": "Unknown regime.",
}


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS strategy_router_results;")

    cursor.execute(
        """
        CREATE TABLE strategy_router_results (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            market_regime TEXT,
            selected_strategy TEXT,
            routing_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    cursor.execute(
        """
        SELECT DISTINCT ON (symbol)
            symbol,
            market_regime
        FROM quant_metrics
        ORDER BY symbol, metric_time DESC;
        """
    )

    rows = cursor.fetchall()

    for symbol, market_regime in rows:
        route = ROUTING_TABLE.get(market_regime, DEFAULT_ROUTE)

        strategy = route["strategy"]
        reason = route["reason"]

        cursor.execute(
            """
            INSERT INTO strategy_router_results (
                symbol,
                market_regime,
                selected_strategy,
                routing_reason
            )
            VALUES (%s, %s, %s, %s);
            """,
            (
                symbol,
                market_regime,
                strategy,
                reason,
            ),
        )

    conn.commit()

    print("")
    print("==============================")
    print("STRATEGY ROUTER")
    print("==============================")

    cursor.execute(
        """
        SELECT
            symbol,
            market_regime,
            selected_strategy
        FROM strategy_router_results;
        """
    )

    for row in cursor.fetchall():
        print(
            f"{row[0]} | "
            f"{row[1]} | "
            f"{row[2]}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


