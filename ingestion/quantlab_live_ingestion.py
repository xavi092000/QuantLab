import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import psycopg2
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI, WebSocketException

from configs.database import DB_CONFIG


SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt"]
STREAMS = "/".join(f"{symbol}@trade" for symbol in SYMBOLS)
BINANCE_WS_URL = f"wss://stream.binance.com:9443/stream?streams={STREAMS}"

# Preserve the original effective database target for this live-ingestion script.
DATABASE_NAME = "quantlab"
DATABASE_PORT = "5432"

STALE_FEED_SECONDS = 5
RECONNECT_DELAY_SECONDS = 5
REQUIRED_DB_CONFIG_KEYS = ("host", "user", "password")
RECONNECTABLE_WEBSOCKET_ERRORS = (
    ConnectionClosed,
    InvalidHandshake,
    InvalidURI,
    WebSocketException,
    OSError,
)
CONNECTION_LEVEL_DB_ERRORS = (
    psycopg2.OperationalError,
    psycopg2.InterfaceError,
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

message_count = 0
start_time = time.time()
last_message_time = time.time()


class MalformedTradeMessage(ValueError):
    """Raised when a Binance trade message is malformed enough to reconnect."""


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    trade_id: int
    price: float
    quantity: float
    event_time: datetime
    ingest_time: datetime
    latency_ms: float


class DatabaseSession:
    """Small PostgreSQL session wrapper with lazy connection and rollback handling."""

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connection_factory = connection_factory
        self._connection: Optional[Any] = None

    def ensure_connection(self) -> Any:
        if self._connection is not None and self._connection.closed == 0:
            return self._connection

        self.close()
        self._connection = self._connection_factory()
        logger.info("postgres_connection_established")
        return self._connection

    def close(self) -> None:
        if self._connection is None:
            return

        connection = self._connection
        self._connection = None

        try:
            connection.close()
            logger.info("postgres_connection_closed")
        except psycopg2.Error as close_error:
            logger.warning(
                "postgres_connection_close_failed error_type=%s error=%s",
                type(close_error).__name__,
                close_error,
            )

    def insert_trade(self, trade: TradeRecord) -> None:
        """Insert and commit one trade, preserving the original per-trade policy."""
        connection = self.ensure_connection()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO market_trades (
                    symbol,
                    trade_id,
                    price,
                    quantity,
                    event_time,
                    ingest_time,
                    latency_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    trade.symbol,
                    trade.trade_id,
                    trade.price,
                    trade.quantity,
                    trade.event_time,
                    trade.ingest_time,
                    trade.latency_ms,
                ),
            )

        connection.commit()

    def recover_after_error(self, db_error: psycopg2.Error) -> None:
        """Rollback failed work and reset the connection when it may be unusable."""
        rollback_failed = False

        if self._connection is not None and self._connection.closed == 0:
            try:
                self._connection.rollback()
            except psycopg2.Error as rollback_error:
                rollback_failed = True
                logger.warning(
                    "postgres_rollback_failed error_type=%s error=%s",
                    type(rollback_error).__name__,
                    rollback_error,
                )

        connection_closed = self._connection is not None and self._connection.closed != 0
        connection_level_error = isinstance(db_error, CONNECTION_LEVEL_DB_ERRORS)

        if rollback_failed or connection_closed or connection_level_error:
            self.close()


def validate_database_config(config: dict[str, Any]) -> None:
    """
    Validate only DB_CONFIG values used by this module.

    The original script hardcoded database='quantlab' and port='5432'. This
    module intentionally preserves that effective target instead of requiring
    DB_CONFIG['database'] or DB_CONFIG['port'].
    """
    missing_keys = [key for key in REQUIRED_DB_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise KeyError(f"Missing database configuration keys: {', '.join(missing_keys)}")

    if not isinstance(config["host"], str) or not config["host"].strip():
        raise ValueError("Invalid database configuration: host must be a non-empty string")

    if not isinstance(config["user"], str) or not config["user"].strip():
        raise ValueError("Invalid database configuration: user must be a non-empty string")

    if config["password"] is None:
        raise ValueError("Invalid database configuration: password must not be None")


def create_db_connection() -> Any:
    validate_database_config(DB_CONFIG)
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        database=DATABASE_NAME,
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        port=DATABASE_PORT,
    )


def parse_trade_message(raw_message: str | bytes, ingest_time: datetime) -> Optional[TradeRecord]:
    """
    Parse one Binance combined-stream trade message.

    Missing 'data' is treated as a non-fatal warning, matching the original
    continue behavior. Malformed JSON or malformed trade payloads raise
    MalformedTradeMessage so the supervisor reconnects, matching the original
    broad outer exception/reconnect behavior for parsing/runtime failures.
    """
    try:
        message = json.loads(raw_message)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as json_error:
        raise MalformedTradeMessage("Invalid JSON trade message") from json_error

    if not isinstance(message, dict):
        raise MalformedTradeMessage("Trade message is not a JSON object")

    trade = message.get("data")
    if trade is None:
        logger.warning("message_without_trade_data_received")
        return None

    if not isinstance(trade, dict):
        raise MalformedTradeMessage("Trade data is not a JSON object")

    try:
        event_time_ms = int(trade["T"])
        event_time = datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc)
        latency_ms = abs((ingest_time - event_time).total_seconds() * 1000)

        return TradeRecord(
            symbol=str(trade["s"]),
            trade_id=int(trade["t"]),
            price=float(trade["p"]),
            quantity=float(trade["q"]),
            event_time=event_time,
            ingest_time=ingest_time,
            latency_ms=latency_ms,
        )
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as parse_error:
        raise MalformedTradeMessage("Malformed trade payload") from parse_error


def record_message(clock: Callable[[], float] = time.time) -> float:
    global message_count
    global last_message_time

    now = clock()
    message_count += 1
    last_message_time = now

    elapsed = max(now - start_time, 0.000001)
    return round(message_count / elapsed, 2)


def persist_trade_safely(db_session: DatabaseSession, trade: TradeRecord) -> None:
    try:
        db_session.insert_trade(trade)
    except psycopg2.Error as db_error:
        db_session.recover_after_error(db_error)
        logger.exception(
            "db_error symbol=%s trade_id=%s error_type=%s",
            trade.symbol,
            trade.trade_id,
            type(db_error).__name__,
        )


def log_trade(trade: TradeRecord, throughput: float) -> None:
    logger.info(
        "trade symbol=%s price=%.4f qty=%.6f latency_ms=%.2f throughput=%.2f/sec",
        trade.symbol,
        trade.price,
        trade.quantity,
        trade.latency_ms,
        throughput,
    )


def log_stale_feed(clock: Callable[[], float] = time.time) -> None:
    idle_seconds = clock() - last_message_time
    logger.warning("stale_feed_detected idle_seconds=%.2f", idle_seconds)


async def receive_trade_messages(
    websocket: Any,
    db_session: DatabaseSession,
    clock: Callable[[], float] = time.time,
) -> None:
    """
    Receive, parse, persist, and log trades for one open WebSocket session.

    Stale-feed detection is intentionally timeout-based. The original script
    attempted a stale-feed warning after receiving each message, but it updated
    last_message_time immediately before that check, making the warning
    effectively unreachable during an actual message stall.
    """
    while True:
        try:
            raw_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=STALE_FEED_SECONDS,
            )
        except asyncio.TimeoutError:
            log_stale_feed(clock)
            continue

        ingest_time = datetime.now(timezone.utc)
        trade = parse_trade_message(raw_message, ingest_time)

        if trade is None:
            continue

        throughput = record_message(clock)
        persist_trade_safely(db_session, trade)
        log_trade(trade, throughput)


async def run_websocket_session(
    db_session: DatabaseSession,
    websocket_connector: Callable[..., Any] = websockets.connect,
    clock: Callable[[], float] = time.time,
) -> None:
    logger.info("connecting_to_binance_websocket streams=%s", ",".join(SYMBOLS))

    async with websocket_connector(BINANCE_WS_URL) as websocket:
        logger.info("binance_multi_asset_feed_connected")
        await receive_trade_messages(websocket, db_session, clock)


async def connect_to_binance(
    websocket_connector: Callable[..., Any] = websockets.connect,
    db_connection_factory: Callable[[], Any] = create_db_connection,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> None:
    global last_message_time

    validate_database_config(DB_CONFIG)
    last_message_time = clock()
    reconnect_count = 0
    db_session = DatabaseSession(db_connection_factory)

    try:
        while True:
            try:
                await run_websocket_session(
                    db_session=db_session,
                    websocket_connector=websocket_connector,
                    clock=clock,
                )
            except asyncio.CancelledError:
                logger.info("binance_ingestion_cancelled")
                raise
            except RECONNECTABLE_WEBSOCKET_ERRORS as websocket_error:
                reconnect_count += 1
                logger.error(
                    "websocket_connection_failed reconnect_count=%d error_type=%s error=%s",
                    reconnect_count,
                    type(websocket_error).__name__,
                    websocket_error,
                )
                await sleep(RECONNECT_DELAY_SECONDS)
            except Exception as runtime_error:
                reconnect_count += 1
                db_session.close()
                logger.exception(
                    "ingestion_runtime_error reconnect_count=%d error_type=%s",
                    reconnect_count,
                    type(runtime_error).__name__,
                )
                await sleep(RECONNECT_DELAY_SECONDS)
    finally:
        db_session.close()


if __name__ == "__main__":
    asyncio.run(connect_to_binance())
