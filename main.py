"""Punto de entrada del bot.

Orquesta el pipeline completo: DataFetcher (WebSocket) -> StrategyEngine
(señales de pairs trading) -> ExecutionEngine (órdenes de mercado), con el
RiskManager vigilando exposición y Kill Switch en todo momento.

Ejecutar:
    python main.py
"""
from __future__ import annotations

import asyncio
import contextlib
import signal
import sys

import ccxt.async_support as ccxt_async

from config.settings import Settings, get_settings
from src.core.logger import configure_logging, get_logger
from src.core.models import OrderBook
from src.data.data_fetcher import DataFetcher
from src.execution.execution_engine import ExecutionEngine
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.signals import PairSignal
from src.strategy.strategy_engine import PairConfig, StrategyEngine

# Windows: el proactor loop por defecto es suficiente. En Linux/Mac usar uvloop.
if sys.platform != "win32":
    with contextlib.suppress(ImportError):
        import uvloop

        uvloop.install()

log = get_logger("main")

SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _build_rest_exchange(settings: Settings) -> ccxt_async.Exchange:
    """Crea el cliente REST async de ccxt para la ejecución de órdenes."""
    exchange_class = getattr(ccxt_async, settings.exchange_id)
    exchange = exchange_class(
        {
            "apiKey": settings.exchange_api_key,
            "secret": settings.exchange_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if settings.use_testnet:
        exchange.set_sandbox_mode(True)
    return exchange


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    limits = RiskLimits(
        account_equity_usd=settings.account_equity_usd,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        kelly_fraction=settings.kelly_fraction,
        max_gross_exposure_pct=settings.max_gross_exposure_pct,
        max_position_pct=settings.max_position_pct,
    )
    risk = RiskManager(limits)

    strategy = StrategyEngine(
        PairConfig(symbol_a=SYMBOLS[0], symbol_b=SYMBOLS[1])
    )

    rest_exchange = _build_rest_exchange(settings)
    execution = ExecutionEngine(rest_exchange, risk)

    stop_event = asyncio.Event()

    async def _on_order_book(book: OrderBook) -> None:
        signal: PairSignal | None = await strategy.on_order_book(book)
        if signal is not None and signal.is_actionable:
            await execution.execute_signal(signal)

        # El Kill Switch dispara la liquidación de emergencia y detiene el bot.
        if risk.is_halted:
            await execution.liquidate_all_and_halt()
            stop_event.set()

    try:
        async with DataFetcher(settings) as fetcher:
            stream_task = asyncio.create_task(
                fetcher.stream_order_book(SYMBOLS, on_update=_on_order_book)
            )
            # Espera a señal de parada (Ctrl+C) o Kill Switch.
            await stop_event.wait()
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
    finally:
        await rest_exchange.close()


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(run())

    # Manejo de Ctrl+C multiplataforma.
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, main_task.cancel)

    try:
        loop.run_until_complete(main_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("apagado_solicitado")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
