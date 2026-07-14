import psycopg2

from configs.database import DB_CONFIG


DEFAULT_MODEL_NAME = "signal_success_model"
MODEL_TYPE = "RandomForestClassifier"
FEATURE_SET_NAME = "Quant Metrics + Momentum Features"
VALIDATION_METHOD = "Retrained model validation"
APPROVED_BY = "QuantLab automated registry"


def main() -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id BIGSERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_type TEXT,
                    feature_set_name TEXT,
                    accuracy_pct DOUBLE PRECISION,
                    precision_pct DOUBLE PRECISION,
                    recall_pct DOUBLE PRECISION,
                    f1_score_pct DOUBLE PRECISION,
                    training_rows INTEGER,
                    validation_method TEXT,
                    model_path TEXT NOT NULL,
                    model_status TEXT NOT NULL,
                    promotion_reason TEXT,
                    approved_by TEXT,
                    rollback_ready BOOLEAN DEFAULT TRUE,
                    drift_status TEXT DEFAULT 'NOT_EVALUATED',
                    registered_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (model_name, model_version)
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE model_registry
                ADD COLUMN IF NOT EXISTS model_type TEXT,
                ADD COLUMN IF NOT EXISTS feature_set_name TEXT,
                ADD COLUMN IF NOT EXISTS precision_pct DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS recall_pct DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS f1_score_pct DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS training_rows INTEGER,
                ADD COLUMN IF NOT EXISTS validation_method TEXT,
                ADD COLUMN IF NOT EXISTS promotion_reason TEXT,
                ADD COLUMN IF NOT EXISTS approved_by TEXT,
                ADD COLUMN IF NOT EXISTS rollback_ready BOOLEAN
                    DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS drift_status TEXT
                    DEFAULT 'NOT_EVALUATED';
                """
            )

            # Lock registry updates so two concurrent processes cannot
            # generate the same model version.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(20260713);"
            )

            # Validate that a candidate exists before changing the
            # current production model.
            cursor.execute(
                """
                SELECT
                    model_name,
                    accuracy_pct,
                    model_path
                FROM retrained_model_metrics
                WHERE model_path IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )

            latest_model = cursor.fetchone()

            if latest_model is None:
                print("[SKIPPED] No valid retrained model found.")
                return

            model_name = latest_model[0] or DEFAULT_MODEL_NAME
            accuracy_pct = (
                float(latest_model[1])
                if latest_model[1] is not None
                else None
            )
            model_path = latest_model[2]

            # Prevent the same artifact from being registered twice.
            cursor.execute(
                """
                SELECT
                    model_version,
                    model_status
                FROM model_registry
                WHERE model_name = %s
                  AND model_path = %s
                ORDER BY registered_at DESC
                LIMIT 1;
                """,
                (model_name, model_path),
            )

            existing_model = cursor.fetchone()

            if existing_model is not None:
                print("==============================")
                print("MODEL ALREADY REGISTERED")
                print("==============================")
                print("Model   :", model_name)
                print("Version :", existing_model[0])
                print("Status  :", existing_model[1])
                return

            cursor.execute(
                """
                SELECT COALESCE(MAX(model_id), 0)
                FROM model_registry
                WHERE model_name = %s;
                """,
                (model_name,),
            )

            version_number = cursor.fetchone()[0] + 1
            model_version = f"v{version_number}"

            # Only archive the current production model after a valid
            # candidate has been found and prepared for registration.
            cursor.execute(
                """
                UPDATE model_registry
                SET
                    model_status = 'ARCHIVED',
                    rollback_ready = TRUE
                WHERE model_name = %s
                  AND model_status = 'PRODUCTION';
                """,
                (model_name,),
            )

            promotion_reason = (
                "Latest validated retrained model promoted "
                "to production."
            )

            cursor.execute(
                """
                INSERT INTO model_registry (
                    model_name,
                    model_version,
                    model_type,
                    feature_set_name,
                    accuracy_pct,
                    precision_pct,
                    recall_pct,
                    f1_score_pct,
                    training_rows,
                    validation_method,
                    model_path,
                    model_status,
                    promotion_reason,
                    approved_by,
                    rollback_ready,
                    drift_status
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                );
                """,
                (
                    model_name,
                    model_version,
                    MODEL_TYPE,
                    FEATURE_SET_NAME,
                    accuracy_pct,
                    None,
                    None,
                    None,
                    None,
                    VALIDATION_METHOD,
                    model_path,
                    "PRODUCTION",
                    promotion_reason,
                    APPROVED_BY,
                    True,
                    "NOT_EVALUATED",
                ),
            )

            print("==============================")
            print("MODEL REGISTRY UPDATED")
            print("==============================")
            print("Model   :", model_name)
            print("Version :", model_version)
            print("Type    :", MODEL_TYPE)
            print("Feature :", FEATURE_SET_NAME)
            print(
                "Accuracy:",
                round(accuracy_pct, 2)
                if accuracy_pct is not None
                else "NOT_AVAILABLE",
            )
            print("Status  : PRODUCTION")
            print("Drift   : NOT_EVALUATED")
            print("Artifact:", model_path)


if __name__ == "__main__":
    main()