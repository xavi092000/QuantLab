from datetime import datetime, timedelta, timezone

import pytest

from ml.position_monitor_engine import calculate_exit, is_price_fresh


def test_recent_price_is_fresh() -> None:
    event_time = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert is_price_fresh(event_time) is True


def test_old_price_is_stale() -> None:
    event_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    assert is_price_fresh(event_time) is False


def test_exit_calculation_applies_slippage_and_costs() -> None:
    result = calculate_exit(
        entry_price=100.0,
        trigger_price=98.0,
        quantity=10.0,
        close_reason="CLOSED_STOP_LOSS",
    )

    assert result["exit_price"] < 98.0
    assert result["exit_transaction_cost_usd"] > 0
    assert result["net_exit_value_usd"] < result["gross_exit_value_usd"]
    assert result["net_pnl_usd"] < 0
    assert result["pnl_pct"] < 0


def test_zero_initial_value_does_not_divide_by_zero() -> None:
    result = calculate_exit(
        entry_price=0.0,
        trigger_price=1.0,
        quantity=0.0,
        close_reason="CLOSED_STOP_LOSS",
    )

    assert result["pnl_pct"] == 0.0
