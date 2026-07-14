from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from configs.database import DB_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "artifacts" / "trading_metrics_latest.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "artifacts" / "trading_metrics_breakdown.csv"


@dataclass
class TradingMetrics:
    generated_at_utc: str
    total_closed_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float | None
    gross_profit_usd: float
    gross_loss_usd: float
    net_profit_usd: float
    profit_factor: float | None
    average_trade_pnl_usd: float | None
    median_trade_pnl_usd: float | None
    average_return_pct: float | None
    median_return_pct: float | None
    best_trade_return_pct: float | None
    worst_trade_return_pct: float | None
    cumulative_return_pct: float | None
    max_drawdown_pct: float | None
    sharpe_ratio_trade_based: float | None
    average_holding_minutes: float | None
    longest_holding_minutes: float | None
    open_positions: int
    open_exposure_usd: float
    open_exposure_pct_of_starting_capital: float
    notes: list[str]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def load_closed_trades(conn) -> pd.DataFrame:
    query = """
    SELECT
        trade_id,
        position_id,
        symbol,
        selected_strategy,
        market_regime,
        entry_time,
        exit_time,
        entry_price,
        exit_price,
        position_size_usd,
        quantity,
        pnl_usd,
        pnl_pct,
        gross_pnl_usd,
        gross_exit_value_usd,
        net_exit_value_usd,
        exit_transaction_cost_usd,
        holding_minutes,
        close_reason
    FROM closed_paper_trades
    ORDER BY exit_time, trade_id;
    """
    return pd.read_sql_query(query, conn)


def load_open_position_summary(conn) -> tuple[int, float]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(position_size_usd), 0)
            FROM paper_positions
            WHERE position_status = 'OPEN';
            """
        )
        row = cursor.fetchone()

    if not row:
        return 0, 0.0

    return int(row[0]), float(row[1] or 0.0)


def calculate_max_drawdown_pct(
    pnl_values: pd.Series,
    starting_capital: float,
) -> float | None:
    if pnl_values.empty or starting_capital <= 0:
        return None

    equity = starting_capital + pnl_values.cumsum()
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.replace(0, np.nan)

    clean = drawdown.dropna()
    if clean.empty:
        return 0.0

    return float(abs(clean.min()) * 100)


def calculate_trade_sharpe(returns_pct: pd.Series) -> float | None:
    clean = returns_pct.dropna().astype(float)

    if len(clean) < 2:
        return None

    std = clean.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return None

    return float(clean.mean() / std)


def calculate_cumulative_return_pct(returns_pct: pd.Series) -> float | None:
    clean = returns_pct.dropna().astype(float)

    if clean.empty:
        return None

    compounded = np.prod(1 + clean / 100.0) - 1
    return float(compounded * 100)


def _resolve_pnl_column(df: pd.DataFrame) -> str:
    if "pnl_usd" in df.columns:
        return "pnl_usd"
    if "net_pnl_usd" in df.columns:
        return "net_pnl_usd"
    raise KeyError(
        "Expected either 'pnl_usd' or 'net_pnl_usd' in the trade data."
    )


def build_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dimension",
        "value",
        "trades",
        "wins",
        "losses",
        "win_rate_pct",
        "net_pnl_usd",
        "avg_return_pct",
        "profit_factor",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    pnl_column = _resolve_pnl_column(df)
    frames: list[pd.DataFrame] = []

    for dimension in ("selected_strategy", "market_regime", "symbol"):
        rows = []

        for value, group in df.groupby(dimension, dropna=False):
            pnl = group[pnl_column].fillna(0.0).astype(float)
            returns = group["pnl_pct"].dropna().astype(float)

            gross_profit = float(pnl[pnl > 0].sum())
            gross_loss_abs = abs(float(pnl[pnl < 0].sum()))

            profit_factor = (
                gross_profit / gross_loss_abs
                if gross_loss_abs > 0
                else None
            )

            trades = int(len(group))
            wins = int((pnl > 0).sum())
            losses = int((pnl < 0).sum())

            rows.append(
                {
                    "dimension": dimension,
                    "value": "UNKNOWN" if pd.isna(value) else str(value),
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct": round(wins / trades * 100, 2)
                    if trades
                    else None,
                    "net_pnl_usd": round(float(pnl.sum()), 2),
                    "avg_return_pct": round(float(returns.mean()), 4)
                    if not returns.empty
                    else None,
                    "profit_factor": round(profit_factor, 4)
                    if profit_factor is not None
                    else None,
                }
            )

        frames.append(pd.DataFrame(rows, columns=columns))

    return pd.concat(frames, ignore_index=True)


def calculate_metrics(
    df: pd.DataFrame,
    open_positions: int,
    starting_capital: float,
    open_exposure_usd: float = 0.0,
) -> TradingMetrics:
    notes: list[str] = []

    if df.empty:
        notes.append(
            "No closed paper trades are available yet; performance metrics "
            "will populate after positions are closed."
        )

        return TradingMetrics(
            generated_at_utc=now_utc().isoformat(),
            total_closed_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate_pct=None,
            gross_profit_usd=0.0,
            gross_loss_usd=0.0,
            net_profit_usd=0.0,
            profit_factor=None,
            average_trade_pnl_usd=None,
            median_trade_pnl_usd=None,
            average_return_pct=None,
            median_return_pct=None,
            best_trade_return_pct=None,
            worst_trade_return_pct=None,
            cumulative_return_pct=None,
            max_drawdown_pct=None,
            sharpe_ratio_trade_based=None,
            average_holding_minutes=None,
            longest_holding_minutes=None,
            open_positions=open_positions,
            open_exposure_usd=round(open_exposure_usd, 2),
            open_exposure_pct_of_starting_capital=round(
                open_exposure_usd / starting_capital * 100,
                4,
            ),
            notes=notes,
        )

    working = df.copy()

    pnl_column = _resolve_pnl_column(working)
    if pnl_column == "net_pnl_usd":
        working["pnl_usd"] = working["net_pnl_usd"]

    numeric_columns = (
        "pnl_usd",
        "pnl_pct",
        "gross_pnl_usd",
        "gross_exit_value_usd",
        "net_exit_value_usd",
        "exit_transaction_cost_usd",
        "position_size_usd",
        "holding_minutes",
    )

    for column in numeric_columns:
        if column not in working.columns:
            working[column] = np.nan

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    if "entry_time" not in working.columns and "opened_at" in working.columns:
        working["entry_time"] = working["opened_at"]

    if "exit_time" not in working.columns and "closed_at" in working.columns:
        working["exit_time"] = working["closed_at"]

    if "entry_time" not in working.columns:
        working["entry_time"] = pd.NaT

    if "exit_time" not in working.columns:
        working["exit_time"] = pd.NaT

    working["entry_time"] = pd.to_datetime(
        working["entry_time"],
        utc=True,
        errors="coerce",
    )
    working["exit_time"] = pd.to_datetime(
        working["exit_time"],
        utc=True,
        errors="coerce",
    )

    pnl = working["pnl_usd"].fillna(0.0)
    returns = working["pnl_pct"].dropna()

    total = int(len(working))
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss_signed = float(pnl[pnl < 0].sum())
    gross_loss_abs = abs(gross_loss_signed)

    profit_factor = (
        gross_profit / gross_loss_abs
        if gross_loss_abs > 0
        else None
    )

    holding_minutes = working["holding_minutes"].dropna()

    if holding_minutes.empty:
        holding_minutes = (
            (
                working["exit_time"] - working["entry_time"]
            ).dt.total_seconds()
            / 60.0
        ).dropna()

    sharpe = calculate_trade_sharpe(returns)

    if sharpe is not None:
        notes.append(
            "Sharpe ratio is trade-based and non-annualized."
        )

    if total < 30:
        notes.append(
            "Performance sample contains fewer than 30 closed trades; "
            "metrics are preliminary."
        )

    cumulative_return = calculate_cumulative_return_pct(returns)
    max_drawdown = calculate_max_drawdown_pct(
        pnl_values=pnl,
        starting_capital=starting_capital,
    )

    return TradingMetrics(
        generated_at_utc=now_utc().isoformat(),
        total_closed_trades=total,
        winning_trades=wins,
        losing_trades=losses,
        breakeven_trades=breakeven,
        win_rate_pct=round(wins / total * 100, 2),
        gross_profit_usd=round(gross_profit, 2),
        gross_loss_usd=round(gross_loss_signed, 2),
        net_profit_usd=round(float(pnl.sum()), 2),
        profit_factor=round(profit_factor, 4)
        if profit_factor is not None
        else None,
        average_trade_pnl_usd=round(float(pnl.mean()), 2),
        median_trade_pnl_usd=round(float(pnl.median()), 2),
        average_return_pct=round(float(returns.mean()), 4)
        if not returns.empty
        else None,
        median_return_pct=round(float(returns.median()), 4)
        if not returns.empty
        else None,
        best_trade_return_pct=round(float(returns.max()), 4)
        if not returns.empty
        else None,
        worst_trade_return_pct=round(float(returns.min()), 4)
        if not returns.empty
        else None,
        cumulative_return_pct=round(cumulative_return, 4)
        if cumulative_return is not None
        else None,
        max_drawdown_pct=round(max_drawdown, 4)
        if max_drawdown is not None
        else None,
        sharpe_ratio_trade_based=round(sharpe, 4)
        if sharpe is not None
        else None,
        average_holding_minutes=round(float(holding_minutes.mean()), 2)
        if not holding_minutes.empty
        else None,
        longest_holding_minutes=round(float(holding_minutes.max()), 2)
        if not holding_minutes.empty
        else None,
        open_positions=open_positions,
        open_exposure_usd=round(open_exposure_usd, 2),
        open_exposure_pct_of_starting_capital=round(
            open_exposure_usd / starting_capital * 100,
            4,
        ),
        notes=notes,
    )


def persist_metrics(
    conn,
    metrics: TradingMetrics,
    breakdown: pd.DataFrame,
) -> None:
    payload = json.dumps(asdict(metrics))
    breakdown_payload = breakdown.to_json(orient="records")

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_metrics_history (
                id BIGSERIAL PRIMARY KEY,
                generated_at TIMESTAMPTZ NOT NULL,
                metrics_json JSONB NOT NULL,
                breakdown_json JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )

        cursor.execute(
            """
            INSERT INTO trading_metrics_history (
                generated_at,
                metrics_json,
                breakdown_json
            )
            VALUES (%s, %s::jsonb, %s::jsonb);
            """,
            (
                datetime.fromisoformat(metrics.generated_at_utc),
                payload,
                breakdown_payload,
            ),
        )


def print_metrics(metrics: TradingMetrics) -> None:
    print()
    print("=" * 72)
    print("QUANTLAB VERIFIED TRADING METRICS")
    print("=" * 72)

    for key, value in asdict(metrics).items():
        if key == "notes":
            continue

        label = key.replace("_", " ").title()
        print(f"{label:<44}: {value}")

    if metrics.notes:
        print("-" * 72)
        print("NOTES")

        for note in metrics.notes:
            print(f"- {note}")

    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate reproducible paper-trading metrics."
    )

    parser.add_argument(
        "--starting-capital",
        type=float,
        default=100_000.0,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_PATH,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_PATH,
    )
    parser.add_argument(
        "--store-db",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.starting_capital <= 0:
        raise ValueError("--starting-capital must be greater than zero.")

    with psycopg2.connect(**DB_CONFIG) as conn:
        trades = load_closed_trades(conn)
        open_positions, open_exposure_usd = load_open_position_summary(conn)

        metrics = calculate_metrics(
            trades,
            open_positions=open_positions,
            open_exposure_usd=open_exposure_usd,
            starting_capital=args.starting_capital,
        )

        breakdown = build_breakdown(trades)

        if args.store_db:
            persist_metrics(conn, metrics, breakdown)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)

    args.json_output.write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    breakdown.to_csv(
        args.csv_output,
        index=False,
    )

    print_metrics(metrics)
    print(f"JSON snapshot : {args.json_output}")
    print(f"CSV breakdown : {args.csv_output}")


if __name__ == "__main__":
    main()
