SELECT
    f.symbol,
    f.selected_strategy,
    f.final_decision,
    f.predicted_return_5m,
    f.ml_vote,

    p.allocation_pct,
    p.position_size_usd,

    r.market_regime,

    a.selector_decision,
    a.historical_recommendation

FROM final_strategy_decisions f

LEFT JOIN portfolio_construction_results p
ON f.symbol = p.symbol

LEFT JOIN quant_metrics r
ON f.symbol = r.symbol

LEFT JOIN asset_strategy_selector_results a
ON f.symbol = a.symbol

ORDER BY f.symbol;