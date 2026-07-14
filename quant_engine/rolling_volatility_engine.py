from configs.database import DB_CONFIG

import math
import time
from statistics import mean, stdev

import psycopg2


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

WINDOW_SIZE = 100
RSI_PERIOD = 14
POLL_SECONDS = 3

last_market_regime_by_symbol = {symbol: None for symbol in SYMBOLS}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_recent_trades(cursor, symbol, limit):
    cursor.execute(
        """
        SELECT event_time, symbol, price, quantity
        FROM market_trades
        WHERE symbol = %s
        ORDER BY event_time DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    return list(reversed(cursor.fetchall()))


def compute_log_returns(prices):
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i - 1]))
    return returns


def compute_z_score(values):
    if len(values) < 2:
        return 0.0
    deviation = stdev(values)
    if deviation == 0:
        return 0.0
    return (values[-1] - mean(values)) / deviation


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent_deltas = deltas[-period:]

    gains = [delta for delta in recent_deltas if delta > 0]
    losses = [-delta for delta in recent_deltas if delta < 0]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def compute_vwap(prices, quantities):
    total_volume = sum(quantities)
    if total_volume == 0:
        return None
    return sum(p * q for p, q in zip(prices, quantities)) / total_volume


def compute_vwap_deviation(price, vwap):
    if vwap is None or vwap == 0:
        return None
    return (price - vwap) / vwap


def compute_volume_spike_ratio(quantities):
    if len(quantities) < 2:
        return 0.0
    average_volume = mean(quantities[:-1])
    if average_volume == 0:
        return 0.0
    return quantities[-1] / average_volume


def compute_rolling_mean_distance(prices):
    avg_price = mean(prices)
    if avg_price == 0:
        return 0.0
    return (prices[-1] - avg_price) / avg_price


def compute_momentum_slope(prices):
    if len(prices) < 2:
        return 0.0
    first_price = prices[0]
    last_price = prices[-1]
    if first_price == 0:
        return 0.0
    return (last_price - first_price) / first_price


def compute_liquidity_pressure(quantities):
    if len(quantities) < 2:
        return 0.0
    recent_volume = sum(quantities[-5:])
    average_volume = mean(quantities)
    if average_volume == 0:
        return 0.0
    return recent_volume / (average_volume * 5)


def classify_market_regime(
    rolling_volatility,
    z_score,
    rsi,
    volume_spike_ratio,
    vwap_deviation,
):
    if abs(z_score) >= 3:
        return "STATISTICAL_ANOMALY"

    if volume_spike_ratio >= 3:
        return "LIQUIDITY_EVENT"

    if rolling_volatility >= 0.00001 and rsi is not None and (rsi >= 70 or rsi <= 30):
        return "VOLATILE_MOMENTUM"

    if rsi is not None and rsi >= 70:
        return "BULLISH_MOMENTUM"

    if rsi is not None and rsi <= 30:
        return "BEARISH_MOMENTUM"

    if vwap_deviation is not None and abs(vwap_deviation) >= 0.0005:
        return "VWAP_DISLOCATION"

    if rolling_volatility < 0.000003:
        return "CALM"

    return "NORMAL"


def compute_signal_coherence_score(
    rolling_volatility,
    z_score,
    rsi,
    vwap_deviation,
    volume_spike_ratio,
    momentum_slope,
):
    score = 0

    if abs(z_score) >= 2:
        score += 20

    if rsi is not None and (rsi >= 70 or rsi <= 30):
        score += 20

    if vwap_deviation is not None and abs(vwap_deviation) >= 0.00005:
        score += 20

    if volume_spike_ratio >= 2:
        score += 20

    if abs(momentum_slope) >= 0.00001:
        score += 20

    if rolling_volatility < 0.000003 and score >= 60:
        score -= 20

    return max(0, min(score, 100))


def classify_signal_coherence(score):
    if score >= 75:
        return "HIGH_COHERENCE"
    if score >= 50:
        return "MODERATE_COHERENCE"
    if score >= 25:
        return "LOW_COHERENCE"
    return "NO_COHERENCE"


def classify_alert_level(signal_coherence_score, z_score, liquidity_pressure, rsi):
    if signal_coherence_score >= 80:
        return "CRITICAL"

    if abs(z_score) >= 5:
        return "EXTREME"

    if liquidity_pressure >= 7:
        return "STRESS"

    if rsi is not None and rsi >= 95:
        return "OVERHEATED"

    return "NORMAL"


def insert_metric(
    cursor,
    metric_time,
    symbol,
    price,
    log_return,
    rolling_volatility,
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
    alert_level,
):
    cursor.execute(
        """
        INSERT INTO quant_metrics (
            metric_time,
            symbol,
            price,
            log_return,
            rolling_volatility,
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
            alert_level,
            window_size
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            metric_time,
            symbol,
            price,
            log_return,
            rolling_volatility,
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
            alert_level,
            WINDOW_SIZE,
        ),
    )


def insert_regime_transition(
    cursor,
    regime_time,
    symbol,
    market_regime,
    rolling_volatility,
    z_score,
    rsi,
    volume_spike_ratio,
    liquidity_pressure,
):
    cursor.execute(
        """
        INSERT INTO market_regimes (
            regime_time,
            symbol,
            market_regime,
            rolling_volatility,
            z_score,
            rsi,
            volume_spike_ratio,
            liquidity_pressure
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            regime_time,
            symbol,
            market_regime,
            rolling_volatility,
            z_score,
            rsi,
            volume_spike_ratio,
            liquidity_pressure,
        ),
    )


def process_symbol(cursor, symbol):
    rows = fetch_recent_trades(cursor, symbol, WINDOW_SIZE + 1)

    if len(rows) < WINDOW_SIZE + 1:
        print(f"[INFO] Waiting for more trades for {symbol}...")
        return

    event_times = [row[0] for row in rows]
    symbols = [row[1] for row in rows]
    prices = [float(row[2]) for row in rows]
    quantities = [float(row[3]) for row in rows]

    returns = compute_log_returns(prices)

    if len(returns) < 2:
        print(f"[INFO] Not enough returns yet for {symbol}...")
        return

    rolling_volatility = stdev(returns)
    latest_log_return = returns[-1]
    z_score = compute_z_score(returns)
    rsi = compute_rsi(prices, RSI_PERIOD)
    vwap = compute_vwap(prices, quantities)
    latest_price = prices[-1]
    vwap_deviation = compute_vwap_deviation(latest_price, vwap)

    volume_spike_ratio = compute_volume_spike_ratio(quantities)
    rolling_mean_distance = compute_rolling_mean_distance(prices)
    momentum_slope = compute_momentum_slope(prices)
    liquidity_pressure = compute_liquidity_pressure(quantities)

    latest_time = event_times[-1]
    latest_symbol = symbols[-1]

    market_regime = classify_market_regime(
        rolling_volatility,
        z_score,
        rsi,
        volume_spike_ratio,
        vwap_deviation,
    )

    signal_coherence_score = compute_signal_coherence_score(
        rolling_volatility,
        z_score,
        rsi,
        vwap_deviation,
        volume_spike_ratio,
        momentum_slope,
    )

    signal_coherence_label = classify_signal_coherence(signal_coherence_score)

    alert_level = classify_alert_level(
        signal_coherence_score,
        z_score,
        liquidity_pressure,
        rsi,
    )

    insert_metric(
        cursor,
        latest_time,
        latest_symbol,
        latest_price,
        latest_log_return,
        rolling_volatility,
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
        alert_level,
    )

    previous_regime = last_market_regime_by_symbol.get(symbol)

    if market_regime != previous_regime:
        insert_regime_transition(
            cursor,
            latest_time,
            latest_symbol,
            market_regime,
            rolling_volatility,
            z_score,
            rsi,
            volume_spike_ratio,
            liquidity_pressure,
        )

        print(
            f"[REGIME] symbol={symbol} "
            f"from={previous_regime} "
            f"to={market_regime}"
        )

        last_market_regime_by_symbol[symbol] = market_regime

    print(
        f"[QUANT] "
        f"symbol={latest_symbol} "
        f"price={latest_price:.4f} "
        f"vol={rolling_volatility:.8f} "
        f"z={z_score:.2f} "
        f"rsi={rsi:.2f} "
        f"vwap_dev={vwap_deviation:.6f} "
        f"vol_spike={volume_spike_ratio:.2f} "
        f"mean_dist={rolling_mean_distance:.6f} "
        f"mom_slope={momentum_slope:.6f} "
        f"liq_pressure={liquidity_pressure:.2f} "
        f"regime={market_regime} "
        f"coherence={signal_coherence_score:.0f} "
        f"coherence_label={signal_coherence_label} "
        f"alert={alert_level}"
    )


def main():
    conn = get_connection()
    cursor = conn.cursor()

    print("[INFO] Multi-Asset Quant Engine started")

    while True:
        try:
            for symbol in SYMBOLS:
                process_symbol(cursor, symbol)

            conn.commit()
            time.sleep(POLL_SECONDS)

        except Exception as error:
            print(f"[ERROR] {error}")
            conn.rollback()
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()




