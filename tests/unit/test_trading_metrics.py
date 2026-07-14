from __future__ import annotations

import pandas as pd

from tools.trading_metrics import (
    build_breakdown,
    calculate_cumulative_return_pct,
    calculate_max_drawdown_pct,
    calculate_metrics,
    calculate_trade_sharpe,
)


def test_cumulative_return_is_compounded() -> None:
    returns = pd.Series([10.0, -10.0])
    result = calculate_cumulative_return_pct(returns)

    assert round(result, 2) == -1.0


def test_drawdown_uses_running_equity_peak() -> None:
    pnl = pd.Series([100.0, -50.0, -100.0, 25.0])
    result = calculate_max_drawdown_pct(
        pnl_values=pnl,
        starting_capital=1000.0,
    )

    assert round(result, 2) == 13.64


def test_trade_sharpe_requires_variability() -> None:
    assert calculate_trade_sharpe(pd.Series([1.0])) is None
    assert calculate_trade_sharpe(pd.Series([1.0, 1.0])) is None


def test_metrics_for_empty_trade_history() -> None:
    metrics = calculate_metrics(
        pd.DataFrame(),
        open_positions=0,
        starting_capital=100_000.0,
    )

    assert metrics.total_closed_trades == 0
    assert metrics.win_rate_pct is None
    assert metrics.profit_factor is None


def test_metrics_for_known_trade_sample() -> None:
    df = pd.DataFrame(
        {
            "net_pnl_usd": [100.0, -50.0, 25.0],
            "pnl_pct": [1.0, -0.5, 0.25],
            "opened_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
            ],
            "closed_at": [
                "2026-01-01T00:30:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T02:15:00Z",
            ],
            "selected_strategy": [
                "MOMENTUM",
                "MOMENTUM",
                "MEAN_REVERSION",
            ],
            "market_regime": ["NORMAL", "NORMAL", "VOLATILE"],
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        }
    )

    metrics = calculate_metrics(
        df,
        open_positions=1,
        starting_capital=100_000.0,
    )

    assert metrics.total_closed_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.win_rate_pct == 66.67
    assert metrics.net_profit_usd == 75.0
    assert metrics.profit_factor == 2.5
    assert metrics.open_positions == 1


def test_breakdown_contains_all_dimensions() -> None:
    df = pd.DataFrame(
        {
            "net_pnl_usd": [100.0, -50.0],
            "pnl_pct": [1.0, -0.5],
            "selected_strategy": ["MOMENTUM", "MOMENTUM"],
            "market_regime": ["NORMAL", "VOLATILE"],
            "symbol": ["BTCUSDT", "ETHUSDT"],
        }
    )

    breakdown = build_breakdown(df)

    assert set(breakdown["dimension"]) == {
        "selected_strategy",
        "market_regime",
        "symbol",
    }
