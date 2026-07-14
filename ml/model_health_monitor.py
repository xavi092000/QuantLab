from configs.database import DB_CONFIG
import psycopg2

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_health_monitor (
            id BIGSERIAL PRIMARY KEY,
            health_status TEXT,
            production_model_name TEXT,
            production_model_version TEXT,
            production_accuracy_pct DOUBLE PRECISION,
            drift_critical_count INTEGER,
            drift_warning_count INTEGER,
            pipeline_ok_count INTEGER,
            pipeline_warning_count INTEGER,
            health_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    cursor.execute("""
        SELECT
            model_name,
            model_version,
            accuracy_pct
        FROM model_registry
        WHERE model_status = 'PRODUCTION'
        ORDER BY registered_at DESC
        LIMIT 1;
    """)

    model_row = cursor.fetchone()

    if model_row is None:
        production_model_name = "NO_PRODUCTION_MODEL"
        production_model_version = "N/A"
        production_accuracy_pct = 0.0
    else:
        production_model_name = model_row[0]
        production_model_version = model_row[1]
        production_accuracy_pct = float(model_row[2])

    cursor.execute("""
        SELECT
            SUM(CASE WHEN drift_status = 'CRITICAL_DRIFT' THEN 1 ELSE 0 END),
            SUM(CASE WHEN drift_status = 'WARNING_DRIFT' THEN 1 ELSE 0 END)
        FROM (
            SELECT DISTINCT ON (feature_name)
                feature_name,
                drift_status,
                created_at
            FROM model_drift_monitor
            ORDER BY feature_name, created_at DESC
        ) x;
    """)

    drift_row = cursor.fetchone()
    drift_critical_count = int(drift_row[0] or 0)
    drift_warning_count = int(drift_row[1] or 0)

    cursor.execute("""
        SELECT
            SUM(CASE WHEN health_status = 'OK' THEN 1 ELSE 0 END),
            SUM(CASE WHEN health_status != 'OK' THEN 1 ELSE 0 END)
        FROM pipeline_health_latest;
    """)

    pipeline_row = cursor.fetchone()
    pipeline_ok_count = int(pipeline_row[0] or 0)
    pipeline_warning_count = int(pipeline_row[1] or 0)

    if drift_critical_count > 0:
        health_status = "RETRAIN_REQUIRED"
        health_reason = "At least one feature shows critical drift."

    elif drift_warning_count > 0:
        health_status = "WARNING"
        health_reason = "At least one feature shows warning-level drift."

    elif pipeline_warning_count > 0:
        health_status = "WARNING"
        health_reason = "One or more pipeline components are warming up or unhealthy."

    else:
        health_status = "HEALTHY"
        health_reason = "Model and pipeline are within expected operating conditions."

    cursor.execute("""
        INSERT INTO model_health_monitor (
            health_status,
            production_model_name,
            production_model_version,
            production_accuracy_pct,
            drift_critical_count,
            drift_warning_count,
            pipeline_ok_count,
            pipeline_warning_count,
            health_reason
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """, (
        health_status,
        production_model_name,
        production_model_version,
        production_accuracy_pct,
        drift_critical_count,
        drift_warning_count,
        pipeline_ok_count,
        pipeline_warning_count,
        health_reason,
    ))

    conn.commit()

    print("==============================")
    print("MODEL HEALTH MONITOR")
    print("==============================")
    print("Status             :", health_status)
    print("Reason             :", health_reason)
    print("Production Model   :", production_model_name)
    print("Version            :", production_model_version)
    print("Accuracy %         :", round(production_accuracy_pct, 2))
    print("Critical Drift     :", drift_critical_count)
    print("Warning Drift      :", drift_warning_count)
    print("Pipeline OK        :", pipeline_ok_count)
    print("Pipeline Warnings  :", pipeline_warning_count)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


