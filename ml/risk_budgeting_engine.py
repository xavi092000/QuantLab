from configs.database import DB_CONFIG
import psycopg2

MAX_PORTFOLIO_RISK_PCT = 10.0
MAX_SINGLE_POSITION_PCT = 5.0

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS risk_budgeting_engine;
""")

cursor.execute("""
CREATE TABLE risk_budgeting_engine (
    id BIGSERIAL PRIMARY KEY,
    full_kelly_pct DOUBLE PRECISION,
    half_kelly_pct DOUBLE PRECISION,
    max_portfolio_risk_pct DOUBLE PRECISION,
    max_single_position_pct DOUBLE PRECISION,
    approved_risk_pct DOUBLE PRECISION,
    risk_decision TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

cursor.execute("""
SELECT
    full_kelly_pct,
    half_kelly_pct
FROM kelly_criterion_results
ORDER BY created_at DESC
LIMIT 1;
""")

row = cursor.fetchone()

if row is None:
    print("[ERROR] No Kelly result found.")
    conn.close()
    raise SystemExit

full_kelly = float(row[0])
half_kelly = float(row[1])

if half_kelly <= 0:
    approved_risk = 0.0
    decision = "NO_RISK_ALLOCATED"

elif half_kelly > MAX_SINGLE_POSITION_PCT:
    approved_risk = MAX_SINGLE_POSITION_PCT
    decision = "CAPPED_BY_SINGLE_POSITION_LIMIT"

elif half_kelly > MAX_PORTFOLIO_RISK_PCT:
    approved_risk = MAX_PORTFOLIO_RISK_PCT
    decision = "CAPPED_BY_PORTFOLIO_RISK_LIMIT"

else:
    approved_risk = half_kelly
    decision = "RISK_APPROVED"

cursor.execute("""
INSERT INTO risk_budgeting_engine (
    full_kelly_pct,
    half_kelly_pct,
    max_portfolio_risk_pct,
    max_single_position_pct,
    approved_risk_pct,
    risk_decision
)
VALUES (%s,%s,%s,%s,%s,%s);
""", (
    full_kelly,
    half_kelly,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_SINGLE_POSITION_PCT,
    approved_risk,
    decision,
))

conn.commit()

print("==============================")
print("RISK BUDGETING ENGINE COMPLETE")
print("==============================")
print("Full Kelly       :", round(full_kelly, 2))
print("Half Kelly       :", round(half_kelly, 2))
print("Approved Risk %  :", round(approved_risk, 2))
print("Decision         :", decision)

cursor.close()
conn.close()


