from configs.database import DB_CONFIG
import psycopg2

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
SELECT health_status
FROM model_health_monitor
LIMIT 1;
""")

row = cursor.fetchone()

if not row:
    print("[ERROR] No model health record found.")
    conn.close()
    quit()

health_status = row[0]

cursor.execute("""
CREATE TABLE IF NOT EXISTS retraining_jobs (
    job_id SERIAL PRIMARY KEY,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    health_status TEXT,
    job_status TEXT,
    notes TEXT
);
""")

if health_status == "RETRAIN_REQUIRED":

    cursor.execute("""
    INSERT INTO retraining_jobs (
        health_status,
        job_status,
        notes
    )
    VALUES (
        %s,
        %s,
        %s
    );
    """,
    (
        health_status,
        "PENDING",
        "Automatic retraining requested by Model Health Monitor"
    ))

    conn.commit()

    print("==============================")
    print("RETRAINING JOB CREATED")
    print("==============================")
    print("Status: PENDING")

else:

    print("==============================")
    print("NO RETRAINING REQUIRED")
    print("==============================")
    print("Current Status:", health_status)

cursor.close()
conn.close()


