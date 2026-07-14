from configs.database import DB_CONFIG
import psycopg2
import math
from datetime import datetime

WINDOW_SIZE = 30
MIN_TRADES_PER_SYMBOL = 30


def calculate_rsi(prices):
    gains = []
    losses = []

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def classify_regime(rsi, z_score, rolling_volatility, liquidity_pressure):
    if abs(z_score) >= 3:
        return "STATISTICAL_ANOMALY"

    if liquidity_pressure >= 3:
        return "LIQUIDITY_EVENT"

    if rolling_volatility >= 0.002:
        return "VOLATILE_MOMENTUM"

    if rsi >= 70:
        return "BULLISH_MOMENTUM"

    if rsi <= 30:
        return "BEARISH_MOMENTUM"

    return "NORMAL"


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quant_metrics (
            id BIGSERIAL PRIMARY KEY,
            metric_time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            price DOUBLE PRECISION,
            log_return DOUBLE PRECISION,
            rolling_volatility DOUBLE PRECISION,
            window_size INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            z_score DOUBLE PRECISION,
            rsi DOUBLE PRECISION,
            vwap DOUBLE PRECISION,
            vwap_deviation DOUBLE PRECISION,
            volume_spike_ratio DOUBLE PRECISION,
            rolling_mean_distance DOUBLE PRECISION,
            momentum_slope DOUBLE PRECISION,
            liquidity_pressure DOUBLE PRECISION,
            market_regime TEXT,
            signal_coherence_score DOUBLE PRECISION,
            signal_coherence_label TEXT,
            alert_level TEXT
        );
    """)

    cursor.execute("""
        SELECT DISTINCT symbol
        FROM market_trades
        ORDER BY symbol;
    """)

    symbols = [row[0] for row in cursor.fetchall()]
    inserted = 0

    for symbol in symbols:
        cursor.execute("""
            SELECT
                event_time,
                price,
                quantity
            FROM market_trades
            WHERE symbol = %s
            ORDER BY event_time DESC
            LIMIT %s;
        """, (symbol, WINDOW_SIZE))

        rows = cursor.fetchall()

        if len(rows) < MIN_TRADES_PER_SYMBOL:
            continue

        rows = list(reversed(rows))

        times = [r[0] for r in rows]
        prices = [float(r[1]) for r in rows]
        quantities = [float(r[2]) for r in rows]

        metric_time = times[-1]
        current_price = prices[-1]
        previous_price = prices[-2]

        if previous_price <= 0:
            log_return = 0.0
        else:
            log_return = math.log(current_price / previous_price)

        mean_price = sum(prices) / len(prices)

        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_price = math.sqrt(variance)

        z_score = 0.0 if std_price == 0 else (current_price - mean_price) / std_price

        rolling_volatility = 0.0
        log_returns = []

        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))

        if log_returns:
            mean_lr = sum(log_returns) / len(log_returns)
            variance_lr = sum((x - mean_lr) ** 2 for x in log_returns) / len(log_returns)
            rolling_volatility = math.sqrt(variance_lr)

        total_volume = sum(quantities)
        vwap = (
            sum(prices[i] * quantities[i] for i in range(len(prices))) / total_volume
            if total_volume > 0
            else current_price
        )

        vwap_deviation = (
            (current_price - vwap) / vwap
            if vwap != 0
            else 0.0
        )

        avg_volume = total_volume / len(quantities)
        current_volume = quantities[-1]

        volume_spike_ratio = (
            current_volume / avg_volume
            if avg_volume > 0
            else 0.0
        )

        rolling_mean_distance = (
            (current_price - mean_price) / mean_price
            if mean_price != 0
            else 0.0
        )

        momentum_slope = (
            (prices[-1] - prices[0]) / prices[0]
            if prices[0] != 0
            else 0.0
        )

        liquidity_pressure = volume_spike_ratio * abs(vwap_deviation)

        rsi = calculate_rsi(prices)

        market_regime = classify_regime(
            rsi,
            z_score,
            rolling_volatility,
            liquidity_pressure
        )

        signal_coherence_score = 0.0

        if momentum_slope > 0:
            signal_coherence_score += 25

        if rsi > 50:
            signal_coherence_score += 25

        if vwap_deviation > 0:
            signal_coherence_score += 25

        if rolling_volatility < 0.002:
            signal_coherence_score += 25

        if signal_coherence_score >= 75:
            signal_coherence_label = "HIGH"
        elif signal_coherence_score >= 50:
            signal_coherence_label = "MEDIUM"
        else:
            signal_coherence_label = "LOW"

        if market_regime in ["STATISTICAL_ANOMALY", "LIQUIDITY_EVENT"]:
            alert_level = "HIGH"
        elif market_regime == "VOLATILE_MOMENTUM":
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"

        cursor.execute("""
            INSERT INTO quant_metrics (
                metric_time,
                symbol,
                price,
                log_return,
                rolling_volatility,
                window_size,
                z_score,
                rsi,
                vwap,
                vwap_deviation,
                volume_spike_ratio,
                rolling_mean_distance,
                momentum_slope,
                liquidity_pressure,
                market_regime,
                signal_coherence_score,
                signal_coherence_label,
                alert_level
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
        """, (
            metric_time,
            symbol,
            current_price,
            log_return,
            rolling_volatility,
            WINDOW_SIZE,
            z_score,
            rsi,
            vwap,
            vwap_deviation,
            volume_spike_ratio,
            rolling_mean_distance,
            momentum_slope,
            liquidity_pressure,
            market_regime,
            signal_coherence_score,
            signal_coherence_label,
            alert_level
        ))

        inserted += 1

    conn.commit()

    print("==============================")
    print("QUANT METRICS ENGINE")
    print("==============================")
    print("Run time      :", datetime.now())
    print("Symbols found :", len(symbols))
    print("Rows inserted :", inserted)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()


