from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS quantlab_final_summary;")

cursor.execute("""
CREATE TABLE quantlab_final_summary (
    metric TEXT,
    value TEXT,
    category TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
SELECT
    strategy_status,
    model_type,
    feature_set,
    return_threshold,
    profitable_windows,
    windows_tested,
    total_return_pct,
    avg_trades_per_window
FROM production_strategy_config
ORDER BY created_at DESC
LIMIT 1;
""")

strategy = cursor.fetchone()

cursor.execute("""
SELECT accuracy_pct
FROM model_v2_metrics
ORDER BY created_at DESC
LIMIT 1;
""")

model = cursor.fetchone()

cursor.execute("""
SELECT risk_decision
FROM risk_budgeting_engine
ORDER BY created_at DESC
LIMIT 1;
""")

risk = cursor.fetchone()

rows = [
    ("Strategy Status", strategy[0], "Strategy"),
    ("Model Type", strategy[1], "Model"),
    ("Feature Set", strategy[2], "Model"),
    ("Model V2 Accuracy", f"{float(model[0]):.2f}%", "Model"),
    ("Return Threshold", str(strategy[3]), "Strategy"),
    ("Profitable Windows", f"{strategy[4]}/{strategy[5]}", "Validation"),
    ("Walk-Forward Total Return", f"{float(strategy[6]):.4f}", "Validation"),
    ("Avg Trades Per Window", f"{float(strategy[7]):.2f}", "Validation"),
    ("Risk Decision", risk[0], "Risk"),
]

cursor.executemany("""
INSERT INTO quantlab_final_summary (
    metric,
    value,
    category
)
VALUES (%s,%s,%s);
""", rows)

conn.commit()

print("==============================")
print("QUANTLAB FINAL SUMMARY CREATED")
print("==============================")
print("Rows:", len(rows))

cursor.close()
conn.close()


