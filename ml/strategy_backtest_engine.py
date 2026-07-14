from configs.database import DB_CONFIG
import psycopg2
import math

POSITION_SIZE_USD = 10000.0


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS strategy_backtest_results;")

    cursor.execute("""
        CREATE TABLE strategy_backtest_results (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT,
            signal_time TIMESTAMPTZ,
            rsi DOUBLE PRECISION,
            momentum_5m DOUBLE PRECISION,
            market_regime TEXT,
            future_return_5m DOUBLE PRECISION,
            decision TEXT,
            pnl_usd DOUBLE PRECISION,
            profitable BOOLEAN,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            sv.symbol,
            sv.signal_time,
            qm.rsi,
            COALESCE(mf.momentum_5m, 0) AS momentum_5m,
            qm.market_regime,
            sv.future_return_5m
        FROM signal_validation sv
        JOIN quant_metrics qm
            ON sv.symbol = qm.symbol
           AND sv.signal_time = qm.metric_time
        LEFT JOIN bar_momentum_features mf
            ON qm.symbol = mf.symbol
           AND date_trunc('minute', qm.metric_time) = mf.bar_time
        WHERE sv.future_return_5m IS NOT NULL
          AND qm.rsi IS NOT NULL
          AND qm.market_regime IS NOT NULL;
    """)

    rows = cursor.fetchall()

    inserted = 0

    for row in rows:
        symbol = row[0]
        signal_time = row[1]
        rsi = float(row[2])
        momentum_5m = float(row[3])
        market_regime = row[4]
        future_return = float(row[5])

        decision = "NO_TRADE"

        if (
            momentum_5m > 0
            and rsi < 85
            and market_regime not in [
                "LIQUIDITY_EVENT",
                "STATISTICAL_ANOMALY",
                "VWAP_DISLOCATION"
            ]
        ):
            decision = "BUY"

        if decision == "BUY":
            pnl_usd = POSITION_SIZE_USD * future_return
            profitable = pnl_usd > 0
        else:
            pnl_usd = 0.0
            profitable = False

        cursor.execute("""
            INSERT INTO strategy_backtest_results (
                symbol,
                signal_time,
                rsi,
                momentum_5m,
                market_regime,
                future_return_5m,
                decision,
                pnl_usd,
                profitable
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            symbol,
            signal_time,
            rsi,
            momentum_5m,
            market_regime,
            future_return,
            decision,
            pnl_usd,
            profitable,
        ))

        inserted += 1

    conn.commit()

    cursor.execute("""
        SELECT
            COUNT(*) FILTER (WHERE decision = 'BUY') AS trades,
            COUNT(*) FILTER (WHERE decision = 'BUY' AND profitable = true) AS winners,
            AVG(future_return_5m) FILTER (WHERE decision = 'BUY') AS avg_return,
            SUM(pnl_usd) AS total_pnl,
            MIN(pnl_usd) FILTER (WHERE decision = 'BUY') AS worst_trade,
            MAX(pnl_usd) FILTER (WHERE decision = 'BUY') AS best_trade
        FROM strategy_backtest_results;
    """)

    summary = cursor.fetchone()

    trades = int(summary[0] or 0)
    winners = int(summary[1] or 0)
    avg_return = float(summary[2] or 0)
    total_pnl = float(summary[3] or 0)
    worst_trade = float(summary[4] or 0)
    best_trade = float(summary[5] or 0)

    win_rate = (winners / trades * 100) if trades > 0 else 0

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_backtest_summary (
            id BIGSERIAL PRIMARY KEY,
            rule_name TEXT,
            position_size_usd DOUBLE PRECISION,
            rows_tested INTEGER,
            trades INTEGER,
            winners INTEGER,
            win_rate_pct DOUBLE PRECISION,
            avg_return DOUBLE PRECISION,
            total_pnl_usd DOUBLE PRECISION,
            worst_trade_usd DOUBLE PRECISION,
            best_trade_usd DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        INSERT INTO strategy_backtest_summary (
            rule_name,
            position_size_usd,
            rows_tested,
            trades,
            winners,
            win_rate_pct,
            avg_return,
            total_pnl_usd,
            worst_trade_usd,
            best_trade_usd
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, (
        "momentum_positive_rsi_under_85_safe_regime",
        POSITION_SIZE_USD,
        inserted,
        trades,
        winners,
        win_rate,
        avg_return,
        total_pnl,
        worst_trade,
        best_trade,
    ))

    conn.commit()

    print("==============================")
    print("TRUE STRATEGY BACKTEST ENGINE")
    print("==============================")
    print("Rows tested     :", inserted)
    print("Trades          :", trades)
    print("Winners         :", winners)
    print("Win Rate %      :", round(win_rate, 2))
    print("Avg Return      :", round(avg_return, 6))
    print("Total PnL USD   :", round(total_pnl, 2))
    print("Worst Trade USD :", round(worst_trade, 2))
    print("Best Trade USD  :", round(best_trade, 2))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


