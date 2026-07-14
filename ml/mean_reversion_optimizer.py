from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

POSITION_SIZE = 10000


def evaluate_strategy(df, strategy_name, signal_mask):
    trades = df[signal_mask].copy()

    if len(trades) == 0:
        return None

    returns = trades["actual_return"]
    winners = (returns > 0).sum()
    losers = (returns < 0).sum()

    win_rate = winners / len(trades) * 100
    total_pnl = (returns * POSITION_SIZE).sum()
    avg_return = returns.mean()

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

    cumulative_pnl = (returns * POSITION_SIZE).cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    return {
        "strategy_name": strategy_name,
        "trades": int(len(trades)),
        "winners": int(winners),
        "losers": int(losers),
        "win_rate_pct": float(win_rate),
        "avg_return": float(avg_return),
        "total_pnl_usd": float(total_pnl),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_usd": float(max_drawdown),
    }


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        qm.symbol,
        qm.rsi,
        qm.z_score,
        qm.vwap_deviation,
        qm.rolling_volatility,
        qm.market_regime,
        sv.future_return_5m AS actual_return
    FROM signal_validation sv
    JOIN quant_metrics qm
        ON sv.symbol = qm.symbol
       AND sv.signal_time = qm.metric_time
    WHERE sv.future_return_5m IS NOT NULL
      AND qm.rsi IS NOT NULL
      AND qm.z_score IS NOT NULL
      AND qm.vwap_deviation IS NOT NULL
      AND qm.rolling_volatility IS NOT NULL
      AND qm.market_regime IS NOT NULL;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No strategy research data found.")
        conn.close()
        return

    results = []

    z_thresholds = [-1.0, -1.5, -2.0, -2.5, -3.0]
    rsi_thresholds = [25, 30, 35, 40, 45]
    vwap_thresholds = [-0.0025, -0.005, -0.0075, -0.01]

    for z in z_thresholds:
        for rsi in rsi_thresholds:
            mask = (
                (df["z_score"] < z)
                &
                (df["rsi"] < rsi)
                &
                (~df["market_regime"].isin([
                    "LIQUIDITY_EVENT",
                    "STATISTICAL_ANOMALY",
                    "VWAP_DISLOCATION"
                ]))
            )

            name = f"Mean_Reversion_z{z}_rsi{rsi}"

            result = evaluate_strategy(df, name, mask)

            if result:
                results.append(result)

    for vwap in vwap_thresholds:
        for rsi in rsi_thresholds:
            mask = (
                (df["vwap_deviation"] < vwap)
                &
                (df["rsi"] < rsi)
                &
                (~df["market_regime"].isin([
                    "LIQUIDITY_EVENT",
                    "STATISTICAL_ANOMALY",
                    "VWAP_DISLOCATION"
                ]))
            )

            name = f"VWAP_Reversion_vwap{vwap}_rsi{rsi}"

            result = evaluate_strategy(df, name, mask)

            if result:
                results.append(result)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("[INFO] No strategies produced trades.")
        conn.close()
        return

    cursor = conn.cursor()

    cursor.execute("""
        DROP TABLE IF EXISTS mean_reversion_optimizer_results;
    """)

    cursor.execute("""
        CREATE TABLE mean_reversion_optimizer_results (
            id BIGSERIAL PRIMARY KEY,
            strategy_name TEXT,
            trades INTEGER,
            winners INTEGER,
            losers INTEGER,
            win_rate_pct DOUBLE PRECISION,
            avg_return DOUBLE PRECISION,
            total_pnl_usd DOUBLE PRECISION,
            profit_factor DOUBLE PRECISION,
            sharpe_ratio DOUBLE PRECISION,
            max_drawdown_usd DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    for _, row in results_df.iterrows():
        cursor.execute("""
            INSERT INTO mean_reversion_optimizer_results (
                strategy_name,
                trades,
                winners,
                losers,
                win_rate_pct,
                avg_return,
                total_pnl_usd,
                profit_factor,
                sharpe_ratio,
                max_drawdown_usd
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            row["strategy_name"],
            int(row["trades"]),
            int(row["winners"]),
            int(row["losers"]),
            float(row["win_rate_pct"]),
            float(row["avg_return"]),
            float(row["total_pnl_usd"]),
            float(row["profit_factor"]),
            float(row["sharpe_ratio"]),
            float(row["max_drawdown_usd"]),
        ))

    conn.commit()

    print("==============================")
    print("MEAN REVERSION OPTIMIZER")
    print("==============================")
    print("Rows tested:", len(df))
    print("Strategies tested:", len(results_df))

    print(
        results_df.sort_values(
            ["total_pnl_usd", "profit_factor"],
            ascending=[False, False]
        ).head(10)
    )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


