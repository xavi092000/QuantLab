from configs.database import DB_CONFIG
import psycopg2
import pandas as pd

def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        strategy_name,
        total_pnl_usd,
        trades,
        win_rate_pct,
        profit_factor,
        sharpe_ratio,
        max_drawdown_usd
    FROM mean_reversion_optimizer_results;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No attribution data found.")
        conn.close()
        return

    print("")
    print("===================================")
    print("PERFORMANCE ATTRIBUTION ENGINE")
    print("===================================")

    total_positive_pnl = df[df["total_pnl_usd"] > 0]["total_pnl_usd"].sum()
    total_pnl = df["total_pnl_usd"].sum()

    print(f"Strategies analyzed: {len(df)}")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Positive PnL Pool: ${total_positive_pnl:.2f}")
    print("")

    best = df.sort_values("total_pnl_usd", ascending=False).head(10)

    for _, row in best.iterrows():
        contribution = (
            row["total_pnl_usd"] / total_positive_pnl * 100
            if total_positive_pnl != 0 and row["total_pnl_usd"] > 0
            else 0
        )

        print(
            f"{row['strategy_name']} | "
            f"PnL=${row['total_pnl_usd']:.2f} | "
            f"Contribution={contribution:.2f}% | "
            f"Trades={int(row['trades'])} | "
            f"WinRate={row['win_rate_pct']:.2f}% | "
            f"PF={row['profit_factor']:.2f} | "
            f"Sharpe={row['sharpe_ratio']:.4f} | "
            f"MaxDD=${row['max_drawdown_usd']:.2f}"
        )

    conn.close()


if __name__ == "__main__":
    main()


