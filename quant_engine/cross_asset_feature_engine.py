from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        symbol,
        bar_time,
        close_price,
        momentum_5m,
        momentum_15m,
        momentum_30m
    FROM bar_momentum_features
    WHERE momentum_5m IS NOT NULL
      AND momentum_15m IS NOT NULL
      AND momentum_30m IS NOT NULL
    ORDER BY bar_time, symbol;
    """

    print("[INFO] Loading bar momentum features...")
    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No momentum data found.")
        conn.close()
        return

    print(f"[INFO] Rows loaded: {len(df)}")

    market = (
        df.groupby("bar_time")
          .agg(
              market_avg_momentum_5m=("momentum_5m", "mean"),
              market_avg_momentum_15m=("momentum_15m", "mean"),
              market_avg_momentum_30m=("momentum_30m", "mean"),
              market_volatility_5m=("momentum_5m", "std"),
              market_volatility_15m=("momentum_15m", "std"),
              market_volatility_30m=("momentum_30m", "std"),
          )
          .reset_index()
    )

    btc = (
        df[df["symbol"] == "BTCUSDT"]
        [[
            "bar_time",
            "momentum_5m",
            "momentum_15m",
            "momentum_30m",
        ]]
        .rename(columns={
            "momentum_5m": "btc_momentum_5m",
            "momentum_15m": "btc_momentum_15m",
            "momentum_30m": "btc_momentum_30m",
        })
    )

    merged = df.merge(market, on="bar_time", how="left")
    merged = merged.merge(btc, on="bar_time", how="left")

    merged["relative_momentum_5m"] = (
        merged["momentum_5m"] - merged["market_avg_momentum_5m"]
    )

    merged["relative_momentum_15m"] = (
        merged["momentum_15m"] - merged["market_avg_momentum_15m"]
    )

    merged["relative_momentum_30m"] = (
        merged["momentum_30m"] - merged["market_avg_momentum_30m"]
    )

    merged["btc_relative_momentum_5m"] = (
        merged["momentum_5m"] - merged["btc_momentum_5m"]
    )

    merged["btc_relative_momentum_15m"] = (
        merged["momentum_15m"] - merged["btc_momentum_15m"]
    )

    merged["btc_relative_momentum_30m"] = (
        merged["momentum_30m"] - merged["btc_momentum_30m"]
    )

    merged = merged.fillna(0)

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS cross_asset_features;")

    cursor.execute("""
        CREATE TABLE cross_asset_features (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            bar_time TIMESTAMPTZ,
            market_avg_momentum_5m DOUBLE PRECISION,
            market_avg_momentum_15m DOUBLE PRECISION,
            market_avg_momentum_30m DOUBLE PRECISION,
            market_volatility_5m DOUBLE PRECISION,
            market_volatility_15m DOUBLE PRECISION,
            market_volatility_30m DOUBLE PRECISION,
            btc_momentum_5m DOUBLE PRECISION,
            btc_momentum_15m DOUBLE PRECISION,
            btc_momentum_30m DOUBLE PRECISION,
            relative_momentum_5m DOUBLE PRECISION,
            relative_momentum_15m DOUBLE PRECISION,
            relative_momentum_30m DOUBLE PRECISION,
            btc_relative_momentum_5m DOUBLE PRECISION,
            btc_relative_momentum_15m DOUBLE PRECISION,
            btc_relative_momentum_30m DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    rows = []

    for _, row in merged.iterrows():
        rows.append((
            row["symbol"],
            row["bar_time"],
            float(row["market_avg_momentum_5m"]),
            float(row["market_avg_momentum_15m"]),
            float(row["market_avg_momentum_30m"]),
            float(row["market_volatility_5m"]),
            float(row["market_volatility_15m"]),
            float(row["market_volatility_30m"]),
            float(row["btc_momentum_5m"]),
            float(row["btc_momentum_15m"]),
            float(row["btc_momentum_30m"]),
            float(row["relative_momentum_5m"]),
            float(row["relative_momentum_15m"]),
            float(row["relative_momentum_30m"]),
            float(row["btc_relative_momentum_5m"]),
            float(row["btc_relative_momentum_15m"]),
            float(row["btc_relative_momentum_30m"]),
        ))

    cursor.executemany("""
        INSERT INTO cross_asset_features (
            symbol,
            bar_time,
            market_avg_momentum_5m,
            market_avg_momentum_15m,
            market_avg_momentum_30m,
            market_volatility_5m,
            market_volatility_15m,
            market_volatility_30m,
            btc_momentum_5m,
            btc_momentum_15m,
            btc_momentum_30m,
            relative_momentum_5m,
            relative_momentum_15m,
            relative_momentum_30m,
            btc_relative_momentum_5m,
            btc_relative_momentum_15m,
            btc_relative_momentum_30m
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, rows)

    conn.commit()

    print("==============================")
    print("CROSS-ASSET FEATURE ENGINE")
    print("==============================")
    print(f"Rows inserted: {len(rows)}")
    print("Saved table: cross_asset_features")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


