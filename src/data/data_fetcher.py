"""DataFetcher: ingestión de datos de mercado en tiempo real vía WebSocket.

Usa `ccxt` (que integra ccxt.pro >= v4) para suscribirse al libro de órdenes
por WebSocket con reconexión automática, backoff exponencial y monitorización
de latencia. Está diseñado para minimizar la latencia y sobrevivir a caídas de
red sin intervención manual.

Patrón de uso:

    async with DataFetcher(settings) as fetcher:
        await fetcher.stream_order_book(
            ["BTC/USDT", "ETH/USDT"],
            on_update=my_callback,
        )
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import ccxt.pro as ccxtpro

from config.settings import Settings
from src.core.logger import get_logger
from src.core.models import OrderBook, PriceLevel

log = get_logger("data_fetcher")

# Callback que recibe cada snapshot del order book. Puede ser sync o async.
OrderBookCallback = Callable[[OrderBook], Awaitable[None] | None]


class DataFetcher:
    """Cliente WebSocket para streaming de order books de baja latencia."""

    def __init__(
        self,
        settings: Settings,
        depth: int = 20,
        max_reconnect_delay: float = 30.0,
        staleness_threshold_ms: float = 2_000.0,
    ) -> None:
        """
        Parámetros:
            depth:                 niveles del libro a mantener (top-N).
            max_reconnect_delay:   techo del backoff exponencial (segundos).
            staleness_threshold_ms: si un snapshot supera esta antigüedad,
                                    se registra una alerta de latencia.
        """
        self._settings = settings
        self._depth = depth
        self._max_reconnect_delay = max_reconnect_delay
        self._staleness_threshold_ms = staleness_threshold_ms

        self._exchange: ccxtpro.Exchange | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------ #
    #  Ciclo de vida / context manager
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "DataFetcher":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Instancia el cliente del exchange con la config de credenciales."""
        exchange_class = getattr(ccxtpro, self._settings.exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": self._settings.exchange_api_key,
                "secret": self._settings.exchange_api_secret,
                "enableRateLimit": True,
                "newUpdates": True,  # entrega solo deltas nuevos (menor latencia)
                "options": {"defaultType": "spot"},
            }
        )
        if self._settings.use_testnet:
            self._exchange.set_sandbox_mode(True)

        self._running = True
        log.info(
            "exchange_conectado",
            exchange=self._settings.exchange_id,
            testnet=self._settings.use_testnet,
        )

    async def close(self) -> None:
        """Cierra tareas y la conexión WebSocket de forma ordenada."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        # Espera a que todas las tareas terminen la cancelación.
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._exchange is not None:
            await self._exchange.close()
            log.info("exchange_desconectado")

    # ------------------------------------------------------------------ #
    #  Streaming de order books
    # ------------------------------------------------------------------ #
    async def stream_order_book(
        self,
        symbols: list[str],
        on_update: OrderBookCallback,
    ) -> None:
        """Lanza una tarea de streaming por símbolo y espera a todas.

        Cada símbolo se vigila en su propia corrutina con reconexión
        independiente, de modo que un fallo en un par no tumba a los demás.
        """
        if self._exchange is None:
            raise RuntimeError("DataFetcher no conectado: llama a connect() primero")

        self._tasks = [
            asyncio.create_task(
                self._watch_symbol(symbol, on_update),
                name=f"ob-{symbol}",
            )
            for symbol in symbols
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _watch_symbol(
        self,
        symbol: str,
        on_update: OrderBookCallback,
    ) -> None:
        """Bucle de suscripción con reconexión y backoff exponencial."""
        assert self._exchange is not None
        reconnect_delay = 1.0

        while self._running:
            try:
                raw = await self._exchange.watch_order_book(symbol, limit=self._depth)
                reconnect_delay = 1.0  # reset del backoff tras un tick correcto

                order_book = self._parse_order_book(symbol, raw)
                self._check_staleness(order_book)
                await self._dispatch(on_update, order_book)

            except asyncio.CancelledError:
                log.info("stream_cancelado", symbol=symbol)
                raise

            except ccxtpro.NetworkError as exc:
                # Errores de red son transitorios: reintentar con backoff.
                log.warning(
                    "error_red_reconectando",
                    symbol=symbol,
                    error=str(exc),
                    delay_s=reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_delay)

            except ccxtpro.ExchangeError as exc:
                # Error lógico del exchange (símbolo inválido, permisos...).
                log.error("error_exchange", symbol=symbol, error=str(exc))
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_delay)

            except Exception as exc:  # noqa: BLE001 - red de seguridad final
                log.error(
                    "error_inesperado_stream",
                    symbol=symbol,
                    error=str(exc),
                    exc_info=True,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_delay)

    # ------------------------------------------------------------------ #
    #  Utilidades internas
    # ------------------------------------------------------------------ #
    def _parse_order_book(self, symbol: str, raw: dict[str, Any]) -> OrderBook:
        """Convierte el dict crudo de ccxt en nuestro DTO tipado `OrderBook`."""
        bids = [PriceLevel(price=p, amount=a) for p, a in raw.get("bids", [])[: self._depth]]
        asks = [PriceLevel(price=p, amount=a) for p, a in raw.get("asks", [])[: self._depth]]

        # ccxt entrega timestamp en ms (epoch). Convertimos a segundos.
        ts = raw.get("timestamp")
        timestamp = ts / 1_000.0 if ts else None

        book = OrderBook(symbol=symbol, bids=bids, asks=asks)
        if timestamp is not None:
            book.timestamp = timestamp
        return book

    def _check_staleness(self, book: OrderBook) -> None:
        """Alerta si el snapshot llega con demasiada latencia."""
        if book.age_ms > self._staleness_threshold_ms:
            log.warning(
                "order_book_obsoleto",
                symbol=book.symbol,
                age_ms=round(book.age_ms, 1),
                umbral_ms=self._staleness_threshold_ms,
            )

    @staticmethod
    async def _dispatch(callback: OrderBookCallback, book: OrderBook) -> None:
        """Invoca el callback, soportando funciones sync o async."""
        result = callback(book)
        if asyncio.iscoroutine(result):
            await result
