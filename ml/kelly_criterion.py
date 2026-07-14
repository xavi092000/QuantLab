from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS kelly_criterion_results;
""")

cursor.execute("""
CREATE TABLE kelly_criterion_results (
    id BIGSERIAL PRIMARY KEY,
    win_rate_pct DOUBLE PRECISION,
    avg_win_pct DOUBLE PRECISION,
    avg_loss_pct DOUBLE PRECISION,
    reward_risk_ratio DOUBLE PRECISION,
    full_kelly_pct DOUBLE PRECISION,
    half_kelly_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

# Gagnants
cursor.execute("""
SELECT AVG(pnl_pct)
FROM trade_simulation_results
WHERE pnl_pct > 0;
""")

avg_win = float(cursor.fetchone()[0])

# Perdants
cursor.execute("""
SELECT ABS(AVG(pnl_pct))
FROM trade_simulation_results
WHERE pnl_pct < 0;
""")

avg_loss = float(cursor.fetchone()[0])

# Win Rate
cursor.execute("""
SELECT
    100.0 *
    SUM(
        CASE
            WHEN pnl_pct > 0 THEN 1
            ELSE 0
        END
    ) / COUNT(*)
FROM trade_simulation_results;
""")

win_rate_pct = float(cursor.fetchone()[0])

p = win_rate_pct / 100.0

reward_risk_ratio = avg_win / avg_loss

b = reward_risk_ratio

full_kelly = ((b * p) - (1 - p)) / b

full_kelly_pct = full_kelly * 100

half_kelly_pct = full_kelly_pct / 2

cursor.execute("""
INSERT INTO kelly_criterion_results (
    win_rate_pct,
    avg_win_pct,
    avg_loss_pct,
    reward_risk_ratio,
    full_kelly_pct,
    half_kelly_pct
)
VALUES (%s,%s,%s,%s,%s,%s)
""",
(
    win_rate_pct,
    avg_win,
    avg_loss,
    reward_risk_ratio,
    full_kelly_pct,
    half_kelly_pct
))

conn.commit()

print("==========================")
print("KELLY CRITERION COMPLETE")
print("==========================")
print("Win Rate       :", round(win_rate_pct, 2))
print("Reward/Risk    :", round(reward_risk_ratio, 2))
print("Full Kelly %   :", round(full_kelly_pct, 2))
print("Half Kelly %   :", round(half_kelly_pct, 2))

cursor.close()
conn.close()


