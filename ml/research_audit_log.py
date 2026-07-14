from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS research_audit_log;")

    cursor.execute("""
        CREATE TABLE research_audit_log (
            id BIGSERIAL PRIMARY KEY,
            experiment_name TEXT,
            experiment_type TEXT,
            key_result TEXT,
            verdict TEXT,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    experiments = [
        (
            "Baseline Strategy Backtest",
            "Backtest",
            "Profit Factor 0.866; Kelly negative",
            "REJECTED",
            "Baseline signal strategy did not show positive expectancy."
        ),
        (
            "Momentum V2 Classifier",
            "Machine Learning",
            "Accuracy improved from 69.25% to 71.77%",
            "IMPROVED",
            "OHLC momentum features improved predictive accuracy."
        ),
        (
            "Feature Importance V2",
            "Explainability",
            "Momentum features contributed 37.40% of model importance",
            "VALIDATED",
            "Momentum became a major driver of model decisions."
        ),
        (
            "Naive Return Regression Backtest",
            "Regression Backtest",
            "Very high Profit Factor observed",
            "INVALIDATED",
            "Result likely affected by data leakage / in-sample evaluation."
        ),
        (
            "Out-of-Sample Return Model",
            "Out-of-Sample Validation",
            "Profit Factor 0.4263; Total Return -0.9249",
            "REJECTED",
            "Single holdout test showed poor generalization."
        ),
        (
            "Walk-Forward Return Model",
            "Walk-Forward Validation",
            "27 windows tested; total return +0.3593",
            "PROMISING",
            "Rolling validation showed some positive signal but instability."
        ),
        (
            "Threshold Sweep",
            "Strategy Optimization",
            "Best threshold 0.0002; 16/27 profitable windows; return +0.4723",
            "APPROVED_FOR_RESEARCH",
            "Predicted-return threshold improved walk-forward return."
        ),
        (
            "Production Strategy Config",
            "Strategy Governance",
            "Status RESEARCH_APPROVED; threshold 0.0002",
            "REGISTERED",
            "Strategy approved for research only, not live trading."
        ),
        (
            "Risk Budgeting",
            "Risk Management",
            "Risk Decision NO_RISK_ALLOCATED",
            "BLOCKED",
            "Kelly remained negative, so live capital allocation is blocked."
        ),
    ]

    cursor.executemany("""
        INSERT INTO research_audit_log (
            experiment_name,
            experiment_type,
            key_result,
            verdict,
            notes
        )
        VALUES (%s, %s, %s, %s, %s);
    """, experiments)

    conn.commit()

    print("==============================")
    print("RESEARCH AUDIT LOG CREATED")
    print("==============================")
    print(f"Experiments logged: {len(experiments)}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


