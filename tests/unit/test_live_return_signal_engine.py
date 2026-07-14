from ml.live_return_signal_engine import hybrid_research_signal


def test_buy_candidate_requires_probability_and_return_confirmation() -> None:
    signal, reason = hybrid_research_signal(
        predicted_return=0.0004,
        probability_up=0.70,
        return_threshold=0.0002,
    )

    assert signal == "BUY_CANDIDATE"
    assert "both support upside" in reason


def test_direction_support_with_weak_return_returns_watch() -> None:
    signal, _ = hybrid_research_signal(
        predicted_return=0.0,
        probability_up=0.65,
        return_threshold=0.0002,
    )

    assert signal == "WATCH"


def test_positive_return_without_direction_confidence_returns_watch() -> None:
    signal, _ = hybrid_research_signal(
        predicted_return=0.0003,
        probability_up=0.55,
        return_threshold=0.0002,
    )

    assert signal == "WATCH"


def test_missing_confirmation_returns_avoid() -> None:
    signal, _ = hybrid_research_signal(
        predicted_return=-0.0005,
        probability_up=0.40,
        return_threshold=0.0002,
    )

    assert signal == "AVOID"
