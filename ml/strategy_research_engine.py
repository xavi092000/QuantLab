from configs.database import DB_CONFIG
import psycopg2
import pandas as pd
import numpy as np

POSITION_SIZE = 10000


def evaluate_strategy(df, strategy_name, signal_mask):

    trades = df[signal_mask].copy()

    if len(trades) == 0:
        return None

    returns = trades["actual_return"]

    winners = (returns > 0).sum()

    win_rate = winners / len(trades) * 100

    total_pnl = (returns * POSITION_SIZE).sum()

    avg_return = returns.mean()

    profit_factor = (
        returns[returns > 0].sum()
        /
        abs(returns[returns < 0].sum())
        if len(returns[returns < 0]) > 0
        else 999
    )

    sharpe = (
        returns.mean()
        /
        returns.std()
        if returns.std() > 0
        else 0
    )

    cumulative = (returns * POSITION_SIZE).cumsum()

    running_max = cumulative.cummax()

    drawdown = cumulative - running_max

    max_drawdown = drawdown.min()

    return (
        strategy_name,
        len(trades),
        win_rate,
        avg_return,
        total_pnl,
        profit_factor,
        sharpe,
        max_drawdown
    )


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
    WHERE sv.future_return_5m IS NOT NULL;
    """


    df = pd.read_sql(query, conn)

    conn.close()

    if len(df) == 0:
        print("No data found")
        return

    results = []

    s1 = (
        (df["rsi"] < 70)
        &
        (~df["market_regime"].isin([
            "STATISTICAL_ANOMALY",
            "LIQUIDITY_EVENT"
        ]))
    )

    s2 = (
        (df["z_score"] < -2)
        &
        (df["rsi"] < 35)
    )

    s3 = (
        (df["vwap_deviation"] < -0.01)
    )

    s4 = (
        (df["rolling_volatility"] > 0.002)
        &
        (df["rsi"] > 50)
    )

    strategies = [
        ("Momentum_Safe_Regime", s1),
        ("Mean_Reversion", s2),
        ("VWAP_Reversion", s3),
        ("Volatility_Breakout", s4),
    ]

    for name, mask in strategies:

        result = evaluate_strategy(
            df,
            name,
            mask
        )

        if result:
            results.append(result)

    results_df = pd.DataFrame(
        results,
        columns=[
            "strategy_name",
            "trades",
            "win_rate_pct",
            "avg_return",
            "total_pnl_usd",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown_usd"
        ]
    )

    print("")
    print("====================================")
    print("STRATEGY RESEARCH ENGINE")
    print("====================================")

    print(
        results_df.sort_values(
            "total_pnl_usd",
            ascending=False
        )
    )


if __name__ == "__main__":
    main()


