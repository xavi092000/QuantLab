from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS production_strategy_config;")

cursor.execute("""
CREATE TABLE production_strategy_config (
    id BIGSERIAL PRIMARY KEY,
    strategy_name TEXT,
    model_type TEXT,
    feature_set TEXT,
    return_threshold DOUBLE PRECISION,
    train_method TEXT,
    validation_method TEXT,
    approved_profit_factor DOUBLE PRECISION,
    profitable_windows INTEGER,
    windows_tested INTEGER,
    total_return_pct DOUBLE PRECISION,
    avg_trades_per_window DOUBLE PRECISION,
    strategy_status TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
SELECT
    avg_profit_factor,
    profitable_windows,
    windows_tested,
    total_return,
    avg_trades_per_window
FROM threshold_sweep_results
WHERE threshold = 0.0002
LIMIT 1;
""")

row = cursor.fetchone()

if row is None:
    print("[ERROR] No threshold 0.0002 found.")
    conn.close()
    raise SystemExit

avg_profit_factor = float(row[0])
profitable_windows = int(row[1])
windows_tested = int(row[2])
total_return = float(row[3])
avg_trades = float(row[4])

cursor.execute("""
INSERT INTO production_strategy_config (
    strategy_name,
    model_type,
    feature_set,
    return_threshold,
    train_method,
    validation_method,
    approved_profit_factor,
    profitable_windows,
    windows_tested,
    total_return_pct,
    avg_trades_per_window,
    strategy_status,
    notes
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
""", (
    "QuantLab Momentum Return Strategy",
    "RandomForestRegressor",
    "OHLC Momentum V2",
    0.0002,
    "Walk-forward regression",
    "Out-of-sample rolling windows",
    avg_profit_factor,
    profitable_windows,
    windows_tested,
    total_return,
    avg_trades,
    "RESEARCH_APPROVED",
    "Threshold selected from walk-forward threshold sweep. Not production trading advice."
))

conn.commit()

print("==============================")
print("PRODUCTION STRATEGY CONFIG")
print("==============================")
print("Strategy : QuantLab Momentum Return Strategy")
print("Threshold:", 0.0002)
print("Status   : RESEARCH_APPROVED")
print("PF       :", round(avg_profit_factor, 4))
print("Windows  :", f"{profitable_windows}/{windows_tested}")
print("Return   :", round(total_return, 4))

cursor.close()
conn.close()


