from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PsycopgConnection

from configs.database import DB_CONFIG


MODEL_PATH = "ml/signal_success_model_v2.pkl"

BACKTEST_QUERY = """
SELECT
    qm.rsi,
    qm.z_score,
    qm.rolling_volatility,
    qm.liquidity_pressure,
    qm.market_regime,

    COALESCE(mf.momentum_5m,0) AS momentum_5m,
    COALESCE(mf.momentum_15m,0) AS momentum_15m,
    COALESCE(mf.momentum_30m,0) AS momentum_30m,

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
  AND qm.z_score IS NOT NULL
  AND qm.rolling_volatility IS NOT NULL
  AND qm.liquidity_pressure IS NOT NULL
  AND qm.market_regime IS NOT NULL;
"""

FEATURE_COLUMNS = [
    "rsi",
    "z_score",
    "rolling_volatility",
    "liquidity_pressure",
    "market_regime_encoded",
    "momentum_5m",
    "momentum_15m",
    "momentum_30m",
]


def load_backtest_dataset(conn: PsycopgConnection) -> pd.DataFrame:
    print("[INFO] Loading backtest dataset...")
    df = pd.read_sql_query(BACKTEST_QUERY, conn)
    print(f"[INFO] Rows loaded: {len(df)}")
    return df


def load_model_bundle(model_path: str) -> Tuple[Any, Any]:
    model_bundle: Dict[str, Any] = joblib.load(model_path)

    try:
        return model_bundle["model"], model_bundle["market_regime_encoder"]
    except KeyError as exc:
        raise KeyError(
            f"Model bundle at {model_path!r} is missing required key: {exc.args[0]!r}"
        ) from exc


def add_predictions(df: pd.DataFrame, model: Any, encoder: Any) -> pd.DataFrame:
    df = df.copy()
    df["market_regime_encoded"] = encoder.transform(df["market_regime"].astype(str))
    df["prediction"] = model.predict(df[FEATURE_COLUMNS])
    return df


def calculate_backtest_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float]:
    trades = df[df["prediction"] == 1].copy()

    total_return = float(trades["future_return_5m"].sum())
    avg_return = float(trades["future_return_5m"].mean())

    wins = trades[trades["future_return_5m"] > 0]
    losses = trades[trades["future_return_5m"] < 0]

    if len(losses) == 0:
        profit_factor = 999.0
    else:
        profit_factor = float(
            wins["future_return_5m"].sum()
            / abs(losses["future_return_5m"].sum())
        )

    return trades, avg_return, total_return, profit_factor


def persist_backtest_results(
    conn: PsycopgConnection,
    total_rows: int,
    trades_taken: int,
    avg_return: float,
    total_return: float,
    profit_factor: float,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS backtest_v2_results;")

        cursor.execute(
            """
            CREATE TABLE backtest_v2_results (
                id BIGSERIAL PRIMARY KEY,
                total_rows INTEGER,
                trades_taken INTEGER,
                avg_return_pct DOUBLE PRECISION,
                total_return_pct DOUBLE PRECISION,
                profit_factor DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

        cursor.execute(
            """
            INSERT INTO backtest_v2_results (
                total_rows,
                trades_taken,
                avg_return_pct,
                total_return_pct,
                profit_factor
            )
            VALUES (%s,%s,%s,%s,%s);
            """,
            (
                int(total_rows),
                int(trades_taken),
                float(avg_return),
                float(total_return),
                float(profit_factor),
            ),
        )


def print_backtest_summary(
    total_rows: int,
    trades_taken: int,
    avg_return: float,
    total_return: float,
    profit_factor: float,
) -> None:
    print("==============================")
    print("MODEL V2 BACKTEST")
    print("==============================")
    print("Rows          :", total_rows)
    print("Trades        :", trades_taken)
    print("Avg Return %  :", round(avg_return, 4))
    print("Total Return% :", round(total_return, 4))
    print("Profit Factor :", round(profit_factor, 4))


def main() -> None:
    conn = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)

        df = load_backtest_dataset(conn)

        if df.empty:
            print("[ERROR] No rows loaded.")
            return

        model, encoder = load_model_bundle(MODEL_PATH)
        df = add_predictions(df, model, encoder)

        trades, avg_return, total_return, profit_factor = calculate_backtest_metrics(df)

        if len(trades) == 0:
            print("[INFO] No trades found.")
            return

        persist_backtest_results(
            conn=conn,
            total_rows=len(df),
            trades_taken=len(trades),
            avg_return=avg_return,
            total_return=total_return,
            profit_factor=profit_factor,
        )
        conn.commit()

        print_backtest_summary(
            total_rows=len(df),
            trades_taken=len(trades),
            avg_return=avg_return,
            total_return=total_return,
            profit_factor=profit_factor,
        )

    except Exception as exc:
        if conn is not None and not conn.closed:
            conn.rollback()
        print(f"[ERROR] Backtest failed: {exc}")
        raise

    finally:
        if conn is not None and not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()
