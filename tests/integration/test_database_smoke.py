from __future__ import annotations

import os

import psycopg2
import pytest

from configs.database import DB_CONFIG


pytestmark = pytest.mark.integration


@pytest.fixture
def db_connection():
    if not os.getenv("PGPASSWORD"):
        pytest.skip("PGPASSWORD is required for integration tests.")

    connection = psycopg2.connect(**DB_CONFIG)
    try:
        yield connection
    finally:
        connection.close()


def test_database_connection(db_connection) -> None:
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT 1;")
        assert cursor.fetchone() == (1,)


def test_core_tables_exist(db_connection) -> None:
    required_tables = {
        "market_trades",
        "quant_metrics",
        "live_return_signals",
        "final_strategy_decisions",
        "paper_positions",
        "closed_paper_trades",
    }

    with db_connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
            '''
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

    missing = required_tables - existing_tables
    assert not missing, f"Missing core tables: {sorted(missing)}"
