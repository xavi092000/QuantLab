from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

from configs.database import DB_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURN_MODEL_PATH = PROJECT_ROOT / "ml" / "return_prediction_model.pkl"
DIRECTION_MODEL_PATH = PROJECT_ROOT / "ml" / "return_direction_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "research_evaluation"


QUERY = """
SELECT
    sv.signal_time,
    sv.symbol,
    qm.rsi,
    qm.z_score,
    qm.rolling_volatility,
    qm.liquidity_pressure,
    qm.market_regime,
    COALESCE(mf.momentum_5m, 0) AS momentum_5m,
    COALESCE(mf.momentum_15m, 0) AS momentum_15m,
    COALESCE(mf.momentum_30m, 0) AS momentum_30m,
    sv.future_return_5m
FROM signal_validation sv
JOIN quant_metrics qm
    ON sv.symbol = qm.symbol
   AND sv.signal_time = qm.metric_time
LEFT JOIN bar_momentum_features mf
    ON qm.symbol = mf.symbol
   AND date_trunc('minute', qm.metric_time) = mf.bar_time
WHERE sv.future_return_5m IS NOT NULL
ORDER BY sv.signal_time, sv.symbol;
"""


@dataclass
class EvaluationSummary:
    generated_at_utc: str
    methodology: str
    rows_evaluated: int
    trades_generated: int
    starting_capital_usd: float
    ending_capital_usd: float
    net_profit_usd: float
    cumulative_return_pct: float
    win_rate_pct: float | None
    profit_factor: float | None
    average_trade_return_pct: float | None
    median_trade_return_pct: float | None
    best_trade_return_pct: float | None
    worst_trade_return_pct: float | None
    max_drawdown_pct: float | None
    trade_sharpe_non_annualized: float | None
    probability_threshold: float
    predicted_return_threshold: float
    position_size_usd: float
    estimated_round_trip_cost_pct: float
    monte_carlo_simulations: int
    monte_carlo_median_final_equity_usd: float | None
    monte_carlo_p05_final_equity_usd: float | None
    monte_carlo_p95_final_equity_usd: float | None
    monte_carlo_median_max_drawdown_pct: float | None
    notes: list[str]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def prepare_features(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    prepared = df.copy()
    encoder = bundle["market_regime_encoder"]
    known_classes = set(encoder.classes_)

    prepared["market_regime_safe"] = prepared["market_regime"].apply(
        lambda value: value if value in known_classes else encoder.classes_[0]
    )
    prepared["market_regime_encoded"] = encoder.transform(
        prepared["market_regime_safe"].astype(str)
    )
    return prepared[bundle["features"]]


def load_dataset(conn) -> pd.DataFrame:
    df = pd.read_sql_query(QUERY, conn)
    if df.empty:
        raise RuntimeError("No validated historical observations were found.")

    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return df.dropna().sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def score_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if not RETURN_MODEL_PATH.exists():
        raise FileNotFoundError(RETURN_MODEL_PATH)
    if not DIRECTION_MODEL_PATH.exists():
        raise FileNotFoundError(DIRECTION_MODEL_PATH)

    return_bundle = joblib.load(RETURN_MODEL_PATH)
    direction_bundle = joblib.load(DIRECTION_MODEL_PATH)

    scored = df.copy()
    return_x = prepare_features(scored, return_bundle)
    direction_x = prepare_features(scored, direction_bundle)

    scored["predicted_return_5m"] = return_bundle["model"].predict(return_x)
    scored["probability_up"] = direction_bundle["model"].predict_proba(
        direction_x
    )[:, 1]

    return scored


def create_research_trades(
    scored: pd.DataFrame,
    probability_threshold: float,
    return_threshold: float,
    position_size_usd: float,
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    candidates = scored.loc[
        (scored["probability_up"] >= probability_threshold)
        & (scored["predicted_return_5m"] >= return_threshold)
    ].copy()

    if candidates.empty:
        return candidates

    candidates["gross_return_pct"] = candidates["future_return_5m"] * 100.0
    candidates["net_return_pct"] = (
        candidates["gross_return_pct"] - round_trip_cost_pct
    )
    candidates["gross_pnl_usd"] = (
        position_size_usd * candidates["gross_return_pct"] / 100.0
    )
    candidates["estimated_cost_usd"] = (
        position_size_usd * round_trip_cost_pct / 100.0
    )
    candidates["net_pnl_usd"] = (
        position_size_usd * candidates["net_return_pct"] / 100.0
    )
    candidates["trade_outcome"] = np.where(
        candidates["net_pnl_usd"] > 0,
        "WIN",
        np.where(candidates["net_pnl_usd"] < 0, "LOSS", "BREAKEVEN"),
    )

    keep = [
        "signal_time",
        "symbol",
        "market_regime",
        "predicted_return_5m",
        "probability_up",
        "future_return_5m",
        "gross_return_pct",
        "net_return_pct",
        "gross_pnl_usd",
        "estimated_cost_usd",
        "net_pnl_usd",
        "trade_outcome",
    ]
    return candidates[keep].reset_index(drop=True)


def build_equity_curve(
    trades: pd.DataFrame,
    starting_capital: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            {
                "step": [0],
                "signal_time": [pd.NaT],
                "equity_usd": [starting_capital],
                "running_peak_usd": [starting_capital],
                "drawdown_pct": [0.0],
            }
        )

    equity = starting_capital + trades["net_pnl_usd"].cumsum()
    running_peak = equity.cummax()
    drawdown_pct = (equity - running_peak) / running_peak * 100.0

    return pd.DataFrame(
        {
            "step": np.arange(1, len(trades) + 1),
            "signal_time": trades["signal_time"],
            "equity_usd": equity,
            "running_peak_usd": running_peak,
            "drawdown_pct": drawdown_pct,
        }
    )


def profit_factor(pnl: pd.Series) -> float | None:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def trade_sharpe(returns_pct: pd.Series) -> float | None:
    if len(returns_pct) < 2:
        return None
    std = float(returns_pct.std(ddof=1))
    if std == 0 or not math.isfinite(std):
        return None
    return float(returns_pct.mean() / std)


def run_monte_carlo(
    trades: pd.DataFrame,
    starting_capital: float,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    if trades.empty or simulations <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    trade_returns = trades["net_return_pct"].to_numpy(dtype=float) / 100.0
    n_trades = len(trade_returns)
    rows = []

    for simulation_id in range(simulations):
        sampled = rng.choice(trade_returns, size=n_trades, replace=True)
        equity = starting_capital * np.cumprod(1.0 + sampled)
        running_peak = np.maximum.accumulate(equity)
        drawdown = (equity - running_peak) / running_peak

        rows.append(
            {
                "simulation_id": simulation_id + 1,
                "final_equity_usd": float(equity[-1]),
                "return_pct": float((equity[-1] / starting_capital - 1) * 100),
                "max_drawdown_pct": float(abs(drawdown.min()) * 100),
            }
        )

    return pd.DataFrame(rows)


def safe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def summarize(
    rows_evaluated: int,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    args: argparse.Namespace,
) -> EvaluationSummary:
    notes = [
        "This is an offline historical research simulation, not live or real-money performance.",
        "Trades use observed future 5-minute returns and estimated round-trip costs.",
        "The same dataset may have contributed to model development; results are not a substitute for a fully isolated out-of-sample study.",
    ]

    if trades.empty:
        notes.append("No trades met the configured model thresholds.")

    if len(trades) < 30:
        notes.append("Fewer than 30 simulated trades: performance statistics are preliminary.")

    ending_capital = (
        float(equity_curve["equity_usd"].iloc[-1])
        if not equity_curve.empty
        else args.starting_capital
    )

    if trades.empty:
        return EvaluationSummary(
            generated_at_utc=now_utc().isoformat(),
            methodology="Historical 5-minute hybrid-model research simulation",
            rows_evaluated=rows_evaluated,
            trades_generated=0,
            starting_capital_usd=args.starting_capital,
            ending_capital_usd=args.starting_capital,
            net_profit_usd=0.0,
            cumulative_return_pct=0.0,
            win_rate_pct=None,
            profit_factor=None,
            average_trade_return_pct=None,
            median_trade_return_pct=None,
            best_trade_return_pct=None,
            worst_trade_return_pct=None,
            max_drawdown_pct=None,
            trade_sharpe_non_annualized=None,
            probability_threshold=args.probability_threshold,
            predicted_return_threshold=args.return_threshold,
            position_size_usd=args.position_size,
            estimated_round_trip_cost_pct=args.round_trip_cost_pct,
            monte_carlo_simulations=args.monte_carlo,
            monte_carlo_median_final_equity_usd=None,
            monte_carlo_p05_final_equity_usd=None,
            monte_carlo_p95_final_equity_usd=None,
            monte_carlo_median_max_drawdown_pct=None,
            notes=notes,
        )

    pnl = trades["net_pnl_usd"]
    returns = trades["net_return_pct"]
    wins = int((pnl > 0).sum())

    return EvaluationSummary(
        generated_at_utc=now_utc().isoformat(),
        methodology="Historical 5-minute hybrid-model research simulation",
        rows_evaluated=rows_evaluated,
        trades_generated=len(trades),
        starting_capital_usd=args.starting_capital,
        ending_capital_usd=round(ending_capital, 2),
        net_profit_usd=round(float(pnl.sum()), 2),
        cumulative_return_pct=round(
            (ending_capital / args.starting_capital - 1) * 100,
            4,
        ),
        win_rate_pct=round(wins / len(trades) * 100, 2),
        profit_factor=safe_round(profit_factor(pnl)),
        average_trade_return_pct=safe_round(float(returns.mean())),
        median_trade_return_pct=safe_round(float(returns.median())),
        best_trade_return_pct=safe_round(float(returns.max())),
        worst_trade_return_pct=safe_round(float(returns.min())),
        max_drawdown_pct=safe_round(
            abs(float(equity_curve["drawdown_pct"].min()))
        ),
        trade_sharpe_non_annualized=safe_round(trade_sharpe(returns)),
        probability_threshold=args.probability_threshold,
        predicted_return_threshold=args.return_threshold,
        position_size_usd=args.position_size,
        estimated_round_trip_cost_pct=args.round_trip_cost_pct,
        monte_carlo_simulations=args.monte_carlo,
        monte_carlo_median_final_equity_usd=safe_round(
            float(monte_carlo["final_equity_usd"].median()), 2
        ) if not monte_carlo.empty else None,
        monte_carlo_p05_final_equity_usd=safe_round(
            float(monte_carlo["final_equity_usd"].quantile(0.05)), 2
        ) if not monte_carlo.empty else None,
        monte_carlo_p95_final_equity_usd=safe_round(
            float(monte_carlo["final_equity_usd"].quantile(0.95)), 2
        ) if not monte_carlo.empty else None,
        monte_carlo_median_max_drawdown_pct=safe_round(
            float(monte_carlo["max_drawdown_pct"].median())
        ) if not monte_carlo.empty else None,
        notes=notes,
    )


def save_plots(
    equity_curve: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve["step"], equity_curve["equity_usd"])
    plt.title("QuantLab Historical Research Equity Curve")
    plt.xlabel("Simulated Trade")
    plt.ylabel("Equity (USD)")
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve["step"], equity_curve["drawdown_pct"])
    plt.title("QuantLab Historical Research Drawdown")
    plt.xlabel("Simulated Trade")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown_curve.png", dpi=160)
    plt.close()

    if not monte_carlo.empty:
        plt.figure(figsize=(10, 5))
        plt.hist(monte_carlo["final_equity_usd"], bins=40)
        plt.title("Monte Carlo Final Equity Distribution")
        plt.xlabel("Final Equity (USD)")
        plt.ylabel("Simulation Count")
        plt.tight_layout()
        plt.savefig(output_dir / "monte_carlo_final_equity.png", dpi=160)
        plt.close()


def persist_summary(
    conn,
    summary: EvaluationSummary,
) -> None:
    payload = json.dumps(asdict(summary))

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS research_evaluation_history (
                id BIGSERIAL PRIMARY KEY,
                generated_at TIMESTAMPTZ NOT NULL,
                summary_json JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cursor.execute(
            """
            INSERT INTO research_evaluation_history (
                generated_at,
                summary_json
            )
            VALUES (%s, %s::jsonb);
            """,
            (
                datetime.fromisoformat(summary.generated_at_utc),
                payload,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a historical hybrid-model research simulation, build an "
            "equity curve, calculate performance metrics, and run Monte Carlo."
        )
    )
    parser.add_argument("--starting-capital", type=float, default=100_000.0)
    parser.add_argument("--position-size", type=float, default=1_000.0)
    parser.add_argument("--probability-threshold", type=float, default=0.60)
    parser.add_argument("--return-threshold", type=float, default=0.0002)
    parser.add_argument(
        "--round-trip-cost-pct",
        type=float,
        default=0.14,
        help="Estimated total entry + exit transaction and slippage cost.",
    )
    parser.add_argument("--monte-carlo", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store-db", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.starting_capital <= 0:
        raise ValueError("--starting-capital must be positive.")
    if args.position_size <= 0:
        raise ValueError("--position-size must be positive.")
    if not 0 <= args.probability_threshold <= 1:
        raise ValueError("--probability-threshold must be between 0 and 1.")
    if args.monte_carlo < 0:
        raise ValueError("--monte-carlo cannot be negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with psycopg2.connect(**DB_CONFIG) as conn:
        dataset = load_dataset(conn)
        scored = score_dataset(dataset)
        trades = create_research_trades(
            scored=scored,
            probability_threshold=args.probability_threshold,
            return_threshold=args.return_threshold,
            position_size_usd=args.position_size,
            round_trip_cost_pct=args.round_trip_cost_pct,
        )
        equity_curve = build_equity_curve(
            trades,
            starting_capital=args.starting_capital,
        )
        monte_carlo = run_monte_carlo(
            trades,
            starting_capital=args.starting_capital,
            simulations=args.monte_carlo,
            seed=args.seed,
        )
        summary = summarize(
            rows_evaluated=len(scored),
            trades=trades,
            equity_curve=equity_curve,
            monte_carlo=monte_carlo,
            args=args,
        )

        if args.store_db:
            persist_summary(conn, summary)

    trades.to_csv(args.output_dir / "simulated_trades.csv", index=False)
    equity_curve.to_csv(args.output_dir / "equity_curve.csv", index=False)
    monte_carlo.to_csv(args.output_dir / "monte_carlo_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    save_plots(
        equity_curve=equity_curve,
        monte_carlo=monte_carlo,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 76)
    print("QUANTLAB HISTORICAL RESEARCH EVALUATION")
    print("=" * 76)
    for key, value in asdict(summary).items():
        if key == "notes":
            continue
        print(f"{key.replace('_', ' ').title():<46}: {value}")
    print("-" * 76)
    print("NOTES")
    for note in summary.notes:
        print(f"- {note}")
    print("=" * 76)
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
