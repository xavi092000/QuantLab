from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import numpy as np

POSITION_SIZE = 10000
N_WINDOWS = 5


def evaluate_window(window_name, df):
    if df.empty:
        return None

    signal_mask = (
        (df["z_score"] < -2.5)
        &
        (df["rsi"] < 30)
        &
        (~df["market_regime"].isin([
            "LIQUIDITY_EVENT",
            "STATISTICAL_ANOMALY",
            "VWAP_DISLOCATION"
        ]))
    )

    trades = df[signal_mask].copy()

    if trades.empty:
        return {
            "window_name": window_name,
            "rows_tested": len(df),
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return": 0.0,
            "total_pnl_usd": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_usd": 0.0,
        }

    returns = trades["actual_return"]

    winners = (returns > 0).sum()
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
        "window_name": window_name,
        "rows_tested": len(df),
        "trades": int(len(trades)),
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
        qm.metric_time,
        qm.symbol,
        qm.rsi,
        qm.z_score,
        qm.market_regime,
        sv.future_return_5m AS actual_return
    FROM signal_validation sv
    JOIN quant_metrics qm
        ON sv.symbol = qm.symbol
       AND sv.signal_time = qm.metric_time
    WHERE sv.future_return_5m IS NOT NULL
      AND qm.rsi IS NOT NULL
      AND qm.z_score IS NOT NULL
      AND qm.market_regime IS NOT NULL
    ORDER BY qm.metric_time ASC;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No validation data found.")
        conn.close()
        return

    df = df.sort_values("metric_time").reset_index(drop=True)

    window_size = len(df) // N_WINDOWS
    windows = []

    for i in range(N_WINDOWS):
        start = i * window_size

        if i == N_WINDOWS - 1:
            end = len(df)
        else:
            end = (i + 1) * window_size

        windows.append(df.iloc[start:end].copy())

    results = []

    for i, window_df in enumerate(windows, start=1):
        result = evaluate_window(
            f"Window_{i}",
            window_df
        )

        if result:
            results.append(result)

    results_df = pd.DataFrame(results)

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS walk_forward_strategy_validation;")

    cursor.execute("""
        CREATE TABLE walk_forward_strategy_validation (
            id BIGSERIAL PRIMARY KEY,
            window_name TEXT,
            rows_tested INTEGER,
            trades INTEGER,
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
            INSERT INTO walk_forward_strategy_validation (
                window_name,
                rows_tested,
                trades,
                win_rate_pct,
                avg_return,
                total_pnl_usd,
                profit_factor,
                sharpe_ratio,
                max_drawdown_usd
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            row["window_name"],
            int(row["rows_tested"]),
            int(row["trades"]),
            float(row["win_rate_pct"]),
            float(row["avg_return"]),
            float(row["total_pnl_usd"]),
            float(row["profit_factor"]),
            float(row["sharpe_ratio"]),
            float(row["max_drawdown_usd"]),
        ))

    conn.commit()

    print("==============================")
    print("WALK-FORWARD STRATEGY VALIDATION")
    print("==============================")
    print("Rows tested:", len(df))
    print("Windows:", N_WINDOWS)
    print("Strategy: Mean_Reversion_z-2.5_rsi30")
    print("")

    for _, row in results_df.iterrows():
        print(
            f"{row['window_name']} | "
            f"rows={int(row['rows_tested'])} | "
            f"trades={int(row['trades'])} | "
            f"win_rate={row['win_rate_pct']:.2f}% | "
            f"PnL=${row['total_pnl_usd']:.2f} | "
            f"PF={row['profit_factor']:.2f} | "
            f"Sharpe={row['sharpe_ratio']:.4f} | "
            f"MaxDD=${row['max_drawdown_usd']:.2f}"
        )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


