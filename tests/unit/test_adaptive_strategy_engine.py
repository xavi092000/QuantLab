from ml.adaptive_strategy_engine import evaluate_adaptive_signal


def test_high_risk_regime_blocks_trading() -> None:
    _, _, signal, quality, reason = evaluate_adaptive_signal(
        "STATISTICAL_ANOMALY",
        z_score=-4.0,
        rsi=10.0,
    )

    assert signal == "NO_TRADE"
    assert quality == 0.0
    assert "High-risk regime" in reason


def test_mean_reversion_buy_requires_both_conditions() -> None:
    z_threshold, rsi_threshold, signal, quality, _ = evaluate_adaptive_signal(
        "NORMAL",
        z_score=-3.0,
        rsi=30.0,
    )

    assert z_threshold == -2.5
    assert rsi_threshold == 40.0
    assert signal == "BUY"
    assert quality == 100.0


def test_partial_setup_returns_watch() -> None:
    _, _, signal, quality, _ = evaluate_adaptive_signal(
        "NORMAL",
        z_score=-3.0,
        rsi=55.0,
    )

    assert signal == "WATCH"
    assert quality == 45.0


def test_unknown_regime_returns_avoid() -> None:
    z_threshold, rsi_threshold, signal, quality, reason = (
        evaluate_adaptive_signal(
            "UNSUPPORTED_REGIME",
            z_score=-5.0,
            rsi=5.0,
        )
    )

    assert z_threshold is None
    assert rsi_threshold is None
    assert signal == "AVOID"
    assert quality == 0.0
    assert "No adaptive rule defined" in reason
