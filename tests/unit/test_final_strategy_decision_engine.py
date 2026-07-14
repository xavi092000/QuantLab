import pytest

from ml.final_strategy_decision_engine import (
    determine_final_decision,
    determine_ml_vote,
)


@pytest.mark.parametrize(
    ("predicted_return", "probability_up", "expected"),
    [
        (0.0004, 0.70, "STRONG_SUPPORT"),
        (0.0, 0.70, "SUPPORT"),
        (0.0001, 0.55, "NEUTRAL"),
        (-0.0005, 0.40, "AGAINST"),
    ],
)
def test_ml_vote_levels(
    predicted_return: float,
    probability_up: float,
    expected: str,
) -> None:
    assert determine_ml_vote(predicted_return, probability_up) == expected


def test_momentum_buy_with_support_becomes_buy() -> None:
    decision, reason = determine_final_decision(
        selected_strategy="MOMENTUM",
        adaptive_signal="AVOID",
        momentum_signal="BUY",
        ml_vote="SUPPORT",
    )

    assert decision == "BUY"
    assert "confirmed" in reason


def test_mean_reversion_buy_with_support_becomes_buy() -> None:
    decision, _ = determine_final_decision(
        selected_strategy="MEAN_REVERSION",
        adaptive_signal="BUY",
        momentum_signal="NONE",
        ml_vote="STRONG_SUPPORT",
    )

    assert decision == "BUY"


def test_no_trade_router_is_respected() -> None:
    decision, reason = determine_final_decision(
        selected_strategy="NO_TRADE",
        adaptive_signal="BUY",
        momentum_signal="BUY",
        ml_vote="STRONG_SUPPORT",
    )

    assert decision == "NO_TRADE"
    assert reason == "Router selected NO_TRADE"


def test_short_momentum_rebound_is_watch_not_buy() -> None:
    decision, _ = determine_final_decision(
        selected_strategy="SHORT_MOMENTUM",
        adaptive_signal="BUY",
        momentum_signal="NONE",
        ml_vote="AGAINST",
    )

    assert decision == "WATCH"
