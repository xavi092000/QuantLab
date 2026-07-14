from configs.database import DB_CONFIG
import time
import psycopg2


THRESHOLD = 0.0002  # 0.02%
POLL_SECONDS = 30


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_unvalidated_signals(cursor):
    cursor.execute(
        """
        SELECT
            qm.metric_time,
            qm.symbol,
            qm.market_regime,
            qm.alert_level,
            qm.price
        FROM quant_metrics qm
        LEFT JOIN signal_validation sv
            ON qm.metric_time = sv.signal_time
            AND qm.symbol = sv.symbol
        WHERE sv.id IS NULL
          AND qm.metric_time <= NOW() - INTERVAL '5 minutes'
          AND qm.market_regime IN (
              'STATISTICAL_ANOMALY',
              'LIQUIDITY_EVENT',
              'VOLATILE_MOMENTUM',
              'BULLISH_MOMENTUM',
              'BEARISH_MOMENTUM',
              'VWAP_DISLOCATION'
          )
        ORDER BY qm.metric_time ASC
        LIMIT 50;
        """
    )
    return cursor.fetchall()


def fetch_future_price(cursor, symbol, signal_time):
    cursor.execute(
        """
        SELECT price
        FROM market_trades
        WHERE symbol = %s
          AND event_time >= %s + INTERVAL '5 minutes'
        ORDER BY event_time ASC
        LIMIT 1;
        """,
        (symbol, signal_time),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return float(row[0])


def validate_signal_direction(market_regime, future_return_5m):
    if market_regime == "BULLISH_MOMENTUM":
        return future_return_5m >= THRESHOLD

    if market_regime == "BEARISH_MOMENTUM":
        return future_return_5m <= -THRESHOLD

    if market_regime in (
        "VOLATILE_MOMENTUM",
        "LIQUIDITY_EVENT",
        "STATISTICAL_ANOMALY",
        "VWAP_DISLOCATION",
    ):
        return abs(future_return_5m) >= THRESHOLD

    return False


def insert_validation(
    cursor,
    signal_time,
    symbol,
    market_regime,
    alert_level,
    current_price,
    future_price_5m,
    future_return_5m,
    signal_success,
):
    cursor.execute(
        """
        INSERT INTO signal_validation (
            signal_time,
            symbol,
            market_regime,
            alert_level,
            current_price,
            future_price_5m,
            future_return_5m,
            signal_success,
            threshold
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            signal_time,
            symbol,
            market_regime,
            alert_level,
            current_price,
            future_price_5m,
            future_return_5m,
            signal_success,
            THRESHOLD,
        ),
    )


def main():
    conn = get_connection()
    cursor = conn.cursor()

    print("[INFO] Forward Validation Engine started: Directional Validation")

    while True:
        try:
            signals = fetch_unvalidated_signals(cursor)

            if not signals:
                print("[INFO] No mature signals to validate yet")
                time.sleep(POLL_SECONDS)
                continue

            for signal in signals:
                signal_time, symbol, market_regime, alert_level, current_price = signal

                current_price = float(current_price)
                future_price = fetch_future_price(cursor, symbol, signal_time)

                if future_price is None:
                    continue

                future_return_5m = (future_price - current_price) / current_price
                signal_success = validate_signal_direction(
                    market_regime,
                    future_return_5m,
                )

                insert_validation(
                    cursor,
                    signal_time,
                    symbol,
                    market_regime,
                    alert_level,
                    current_price,
                    future_price,
                    future_return_5m,
                    signal_success,
                )

                print(
                    f"[VALIDATION] "
                    f"time={signal_time} "
                    f"regime={market_regime} "
                    f"alert={alert_level} "
                    f"current={current_price:.2f} "
                    f"future_5m={future_price:.2f} "
                    f"return_5m={future_return_5m:.4%} "
                    f"directional_success={signal_success}"
                )

            conn.commit()

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()


