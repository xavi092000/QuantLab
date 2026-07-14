from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import joblib

MODEL_PATH = "ml/return_direction_model.pkl"

BUY_PROBABILITY_THRESHOLD = 0.60
WATCH_PROBABILITY_THRESHOLD = 0.50


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    WITH latest_quant AS (
        SELECT DISTINCT ON (symbol)
            symbol,
            metric_time,
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure,
            market_regime
        FROM quant_metrics
        WHERE rsi IS NOT NULL
          AND z_score IS NOT NULL
          AND rolling_volatility IS NOT NULL
          AND liquidity_pressure IS NOT NULL
          AND market_regime IS NOT NULL
        ORDER BY symbol, metric_time DESC
    ),
    latest_momentum AS (
        SELECT DISTINCT ON (symbol)
            symbol,
            bar_time,
            momentum_5m,
            momentum_15m,
            momentum_30m
        FROM bar_momentum_features
        ORDER BY symbol, bar_time DESC
    )
    SELECT
        q.symbol,
        q.metric_time,
        q.rsi,
        q.z_score,
        q.rolling_volatility,
        q.liquidity_pressure,
        q.market_regime,
        COALESCE(m.momentum_5m, 0) AS momentum_5m,
        COALESCE(m.momentum_15m, 0) AS momentum_15m,
        COALESCE(m.momentum_30m, 0) AS momentum_30m
    FROM latest_quant q
    LEFT JOIN latest_momentum m
        ON q.symbol = m.symbol;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No live feature rows found.")
        conn.close()
        return

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    encoder = bundle["market_regime_encoder"]
    features = bundle["features"]

    known_classes = set(encoder.classes_)

    df["market_regime_safe"] = df["market_regime"].apply(
        lambda value: value if value in known_classes else encoder.classes_[0]
    )

    df["market_regime_encoded"] = encoder.transform(
        df["market_regime_safe"].astype(str)
    )

    X = df[features]

    df["probability_down"] = model.predict_proba(X)[:, 0]
    df["probability_up"] = model.predict_proba(X)[:, 1]
    df["predicted_direction"] = model.predict(X)

    def direction_signal(probability_up):
        if probability_up >= BUY_PROBABILITY_THRESHOLD:
            return "BUY_CANDIDATE"
        if probability_up >= WATCH_PROBABILITY_THRESHOLD:
            return "WATCH"
        return "AVOID"

    df["direction_signal"] = df["probability_up"].apply(direction_signal)

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS live_direction_signals;")

    cursor.execute("""
        CREATE TABLE live_direction_signals (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            metric_time TIMESTAMPTZ,
            market_regime TEXT,
            probability_up DOUBLE PRECISION,
            probability_down DOUBLE PRECISION,
            predicted_direction INTEGER,
            direction_signal TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO live_direction_signals (
                symbol,
                metric_time,
                market_regime,
                probability_up,
                probability_down,
                predicted_direction,
                direction_signal
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (
            row["symbol"],
            row["metric_time"],
            row["market_regime"],
            float(row["probability_up"]),
            float(row["probability_down"]),
            int(row["predicted_direction"]),
            row["direction_signal"],
        ))

    conn.commit()

    print("==============================")
    print("LIVE DIRECTION SIGNAL ENGINE")
    print("==============================")
    print(f"Rows processed: {len(df)}")
    print(f"BUY threshold  : {BUY_PROBABILITY_THRESHOLD}")
    print(f"WATCH threshold: {WATCH_PROBABILITY_THRESHOLD}")

    print(df[[
        "symbol",
        "market_regime",
        "probability_up",
        "probability_down",
        "predicted_direction",
        "direction_signal"
    ]].sort_values("probability_up", ascending=False))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


