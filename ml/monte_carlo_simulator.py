from configs.database import DB_CONFIG


import psycopg2
import pandas as pd
import numpy as np

NUM_SIMULATIONS = 1000
NUM_TRADES = 340
INITIAL_CAPITAL = 10000


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT pnl_pct
    FROM trade_simulation_results
    WHERE pnl_pct IS NOT NULL;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print("[ERROR] No trade returns found.")
        conn.close()
        return

    results = []

    for sim in range(1, NUM_SIMULATIONS + 1):
        sampled_returns = np.random.choice(
            df["pnl_pct"],
            size=NUM_TRADES,
            replace=True
        )

        capital = INITIAL_CAPITAL

        for r in sampled_returns:
            capital *= (1 + float(r) / 100)

        results.append((sim, capital))

    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS monte_carlo_results;")

    cursor.execute("""
        CREATE TABLE monte_carlo_results (
            simulation_id INTEGER PRIMARY KEY,
            final_capital DOUBLE PRECISION
        );
    """)

    cursor.executemany("""
        INSERT INTO monte_carlo_results (
            simulation_id,
            final_capital
        )
        VALUES (%s, %s);
    """, results)

    conn.commit()

    capitals = [row[1] for row in results]

    print("=================================")
    print("MONTE CARLO COMPLETE")
    print("=================================")
    print(f"Simulations : {NUM_SIMULATIONS}")
    print(f"Worst 5%    : {np.percentile(capitals, 5):.2f}")
    print(f"Median      : {np.percentile(capitals, 50):.2f}")
    print(f"Best 95%    : {np.percentile(capitals, 95):.2f}")
    print("Saved table : monte_carlo_results")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


