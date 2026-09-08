"""ExecutionEngine: corazón transaccional del bot (ccxt.async_support).

Convierte las `PairSignal` del StrategyEngine en órdenes reales de mercado,
ejecutando las DOS patas del par de forma cuasi-atómica para minimizar el
*leg risk* (riesgo de quedarse con una sola pata y, por tanto, exposición
direccional). Integra el RiskManager para el sizing y el control de límites,
y expone un procedimiento de emergencia `liquidate_all_and_halt()` para el
Kill Switch.

Reglas de esta iteración:
    - Solo órdenes a MERCADO (garantizan ejecución de ambas patas).
    - Enrutamiento inteligente con órdenes limit -> iteración futura.

Convención del spread = A − β·B:
    - SHORT_SPREAD -> VENDER A, COMPRAR B.
    - LONG_SPREAD  -> COMPRAR A, VENDER B.
    - EXIT         -> cerrar ambas patas a mercado.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import ccxt  # excepciones tipadas (compartidas con async_support)
import ccxt.async_support as ccxt_async

from src.core.logger import get_logger
from src.core.models import Position, Side
from src.risk.risk_manager import RiskManager
from src.strategy.signals import PairSignal, SignalType

log = get_logger("execution_engine")


@dataclass(slots=True)
class ExecutionConfig:
    """Parámetros de ejecución y sizing.

    `win_prob` y `win_loss_ratio` alimentan el Kelly del RiskManager. En
    producción deben provenir de las estadísticas del backtest (paso 4); aquí
    se usan valores conservadores por defecto.
    """

    win_prob: float = 0.55
    win_loss_ratio: float = 1.5
    max_retries: int = 4
    base_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    order_timeout_s: float = 10.0


@dataclass(slots=True)
class OrderResult:
    """Resultado normalizado de una orden ejecutada."""

    symbol: str
    side: Side
    amount: float
    filled: float
    avg_price: float
    order_id: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.filled > 0


# Excepciones transitorias que justifican reintento con backoff.
_RETRYABLE = (
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
    ccxt.RequestTimeout,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
)


class LegExecutionError(Exception):
    """Fallo definitivo al ejecutar una pata tras agotar los reintentos."""


class ExecutionEngine:
    """Motor de ejecución de órdenes a mercado para pairs trading."""

    def __init__(
        self,
        exchange: ccxt_async.Exchange,
        risk_manager: RiskManager,
        config: ExecutionConfig | None = None,
    ) -> None:
        # Inyección de dependencias: exchange async y gestor de riesgo.
        self._exchange = exchange
        self._risk = risk_manager
        self._cfg = config or ExecutionConfig()

        # Posiciones abiertas gestionadas por este motor (símbolo -> Position).
        self._positions: dict[str, Position] = {}
        # Serializa la ejecución: nunca dos señales del mismo par en paralelo.
        self._exec_lock = asyncio.Lock()
        self._running = True

    # ------------------------------------------------------------------ #
    #  API pública
    # ------------------------------------------------------------------ #
    async def execute_signal(self, signal: PairSignal) -> None:
        """Punto de entrada: traduce una señal en órdenes de mercado.

        Aplica todos los controles de riesgo previos y garantiza que, ante un
        fallo de una pata, el motor queda neutral (sin exposición direccional).
        """
        if not self._running or self._risk.is_halted:
            log.warning("ejecucion_bloqueada", halted=self._risk.is_halted)
            return

        async with self._exec_lock:
            if signal.signal is SignalType.EXIT:
                await self._close_spread(signal)
                return

            if signal.signal in (SignalType.SHORT_SPREAD, SignalType.LONG_SPREAD):
                await self._open_spread(signal)

    async def liquidate_all_and_halt(self) -> None:
        """Procedimiento de EMERGENCIA (Kill Switch).

        1. Detiene el loop de ejecución.
        2. Cancela todas las órdenes abiertas.
        3. Cierra TODAS las posiciones a mercado sin importar el spread.
        """
        self._running = False
        log.critical("LIQUIDACION_EMERGENCIA_INICIADA")

        await self._cancel_all_orders()

        # Cierre de todas las patas a mercado en paralelo.
        symbols = list(self._positions.keys())
        results = await asyncio.gather(
            *(self._close_position(sym) for sym in symbols),
            return_exceptions=True,
        )
        for sym, res in zip(symbols, results, strict=True):
            if isinstance(res, Exception):
                log.critical("fallo_cierre_emergencia", symbol=sym, error=str(res))

        self._positions.clear()
        log.critical("LIQUIDACION_EMERGENCIA_COMPLETA")

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    # ------------------------------------------------------------------ #
    #  Apertura cuasi-atómica de ambas patas
    # ------------------------------------------------------------------ #
    async def _open_spread(self, signal: PairSignal) -> None:
        """Abre las dos patas simultáneamente mitigando el leg risk."""
        if signal.signal is SignalType.SHORT_SPREAD:
            side_a, side_b = Side.SELL, Side.BUY
        else:  # LONG_SPREAD
            side_a, side_b = Side.BUY, Side.SELL

        # 1) Sizing a través del RiskManager (basado en equity y volatilidad).
        qty_a = self._size_leg(signal.price_a)
        # Beta-neutral: β unidades de B por cada unidad de A (spread = A − β·B).
        qty_b = qty_a * abs(signal.hedge_ratio)
        if qty_a <= 0 or qty_b <= 0:
            log.info("sizing_cero_sin_operacion", qty_a=qty_a, qty_b=qty_b)
            return

        # 2) Control de riesgo previo (exposición y estado) para ambas patas.
        prospective = [
            Position(signal.symbol_a, side_a, qty_a, signal.price_a, signal.price_a),
            Position(signal.symbol_b, side_b, qty_b, signal.price_b, signal.price_b),
        ]
        current = self.open_positions
        if not all(self._risk.can_open(p, current) for p in prospective):
            log.warning("riesgo_rechaza_apertura", signal=signal.signal.value)
            return

        # 3) Ejecución simultánea de ambas patas (cuasi-atómica).
        log.info(
            "abriendo_spread",
            signal=signal.signal.value,
            qty_a=round(qty_a, 8),
            qty_b=round(qty_b, 8),
        )
        res_a, res_b = await asyncio.gather(
            self._safe_market_order(signal.symbol_a, side_a, qty_a),
            self._safe_market_order(signal.symbol_b, side_b, qty_b),
            return_exceptions=True,
        )

        # 4) Gestión de fallos de pata -> quedar neutral.
        await self._reconcile_legs(
            (signal.symbol_a, side_a, res_a),
            (signal.symbol_b, side_b, res_b),
        )

    async def _reconcile_legs(
        self,
        leg_a: tuple[str, Side, OrderResult | BaseException],
        leg_b: tuple[str, Side, OrderResult | BaseException],
    ) -> None:
        """Si una pata falla y la otra no, deshace la ejecutada a mercado."""
        sym_a, side_a, res_a = leg_a
        sym_b, side_b, res_b = leg_b
        ok_a = isinstance(res_a, OrderResult) and res_a.is_filled
        ok_b = isinstance(res_b, OrderResult) and res_b.is_filled

        if ok_a and ok_b:
            assert isinstance(res_a, OrderResult) and isinstance(res_b, OrderResult)
            self._register_fill(res_a, side_a)
            self._register_fill(res_b, side_b)
            log.info("spread_abierto_ok", leg_a=sym_a, leg_b=sym_b)
            return

        # Ambas fallaron: neutral por definición, solo registrar.
        if not ok_a and not ok_b:
            log.error("ambas_patas_fallaron", leg_a=sym_a, leg_b=sym_b)
            return

        # Fallo asimétrico: cerrar a mercado la pata que sí entró.
        filled_leg = res_a if ok_a else res_b
        filled_side = side_a if ok_a else side_b
        failed_sym = sym_b if ok_a else sym_a
        assert isinstance(filled_leg, OrderResult)
        failed_err = res_b if ok_a else res_a
        log.error(
            "leg_risk_detectado_deshaciendo",
            pata_ejecutada=filled_leg.symbol,
            pata_fallida=failed_sym,
            error=str(failed_err),
        )
        await self._unwind_filled_leg(filled_leg, filled_side)

    async def _unwind_filled_leg(self, leg: OrderResult, side: Side) -> None:
        """Cierra a mercado la pata ejecutada para volver a exposición neutra."""
        opposite = Side.BUY if side is Side.SELL else Side.SELL
        try:
            await self._safe_market_order(leg.symbol, opposite, leg.filled)
            log.info("pata_deshecha_neutral", symbol=leg.symbol, amount=leg.filled)
        except Exception as exc:  # noqa: BLE001 - último recurso: alertar y parar
            log.critical(
                "fallo_deshacer_pata_EXPOSICION_ABIERTA",
                symbol=leg.symbol,
                amount=leg.filled,
                error=str(exc),
            )
            self._risk.trip_kill_switch(reason=f"leg_unwind_failed:{leg.symbol}")

    # ------------------------------------------------------------------ #
    #  Cierre de posiciones
    # ------------------------------------------------------------------ #
    async def _close_spread(self, signal: PairSignal) -> None:
        """Cierra ambas patas del par a mercado (señal EXIT)."""
        symbols = [signal.symbol_a, signal.symbol_b]
        log.info("cerrando_spread", symbols=symbols)
        await asyncio.gather(
            *(self._close_position(sym) for sym in symbols),
            return_exceptions=True,
        )

    async def _close_position(self, symbol: str) -> None:
        """Cierra una posición concreta a mercado con la cantidad registrada."""
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return
        opposite = Side.BUY if pos.side is Side.SELL else Side.SELL
        result = await self._safe_market_order(symbol, opposite, pos.quantity)
        if result.is_filled:
            self._positions.pop(symbol, None)
            log.info("posicion_cerrada", symbol=symbol, amount=result.filled)

    # ------------------------------------------------------------------ #
    #  Envío de órdenes con reintentos y backoff exponencial
    # ------------------------------------------------------------------ #
    async def _safe_market_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
    ) -> OrderResult:
        """Envía una orden de mercado con reintentos ante errores transitorios.

        Reintenta con backoff exponencial en Rate Limits (HTTP 429), timeouts y
        errores de red. Ante `InsufficientFunds` u otros errores lógicos, aborta
        de inmediato (no tiene sentido reintentar).
        """
        amount = self._normalize_amount(symbol, amount)
        if amount <= 0:
            raise LegExecutionError(f"cantidad no válida para {symbol}: {amount}")

        backoff = self._cfg.base_backoff_s
        last_exc: Exception | None = None

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                raw = await asyncio.wait_for(
                    self._exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side=side.value,
                        amount=amount,
                    ),
                    timeout=self._cfg.order_timeout_s,
                )
                return self._parse_order(symbol, side, amount, raw)

            except ccxt.InsufficientFunds as exc:
                log.error("fondos_insuficientes", symbol=symbol, error=str(exc))
                raise LegExecutionError(str(exc)) from exc

            except _RETRYABLE as exc:
                last_exc = exc
                log.warning(
                    "orden_reintento",
                    symbol=symbol,
                    intento=attempt,
                    max=self._cfg.max_retries,
                    error=type(exc).__name__,
                    backoff_s=round(backoff, 2),
                )
                if attempt < self._cfg.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.max_backoff_s)

            except asyncio.TimeoutError as exc:
                last_exc = exc
                log.warning("orden_timeout", symbol=symbol, intento=attempt)
                if attempt < self._cfg.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.max_backoff_s)

            except ccxt.ExchangeError as exc:
                log.error("error_exchange_orden", symbol=symbol, error=str(exc))
                raise LegExecutionError(str(exc)) from exc

        raise LegExecutionError(
            f"orden fallida tras {self._cfg.max_retries} intentos: {symbol} ({last_exc})"
        )

    async def _cancel_all_orders(self) -> None:
        """Cancela todas las órdenes abiertas de los símbolos gestionados."""
        for symbol in list(self._positions.keys()):
            try:
                await self._exchange.cancel_all_orders(symbol)
                log.info("ordenes_canceladas", symbol=symbol)
            except ccxt.NotSupported:
                # Fallback: cancelar una a una si el exchange no soporta bulk.
                await self._cancel_open_orders_individually(symbol)
            except Exception as exc:  # noqa: BLE001 - no debe abortar la liquidación
                log.error("fallo_cancelar_ordenes", symbol=symbol, error=str(exc))

    async def _cancel_open_orders_individually(self, symbol: str) -> None:
        try:
            open_orders = await self._exchange.fetch_open_orders(symbol)
            for order in open_orders:
                await self._exchange.cancel_order(order["id"], symbol)
        except Exception as exc:  # noqa: BLE001
            log.error("fallo_cancelar_individual", symbol=symbol, error=str(exc))

    # ------------------------------------------------------------------ #
    #  Sizing, parsing y registro de fills
    # ------------------------------------------------------------------ #
    def _size_leg(self, price: float) -> float:
        """Pide al RiskManager el tamaño de la pata A (unidades base)."""
        return self._risk.position_size(
            win_prob=self._cfg.win_prob,
            win_loss_ratio=self._cfg.win_loss_ratio,
            price=price,
        )

    def _normalize_amount(self, symbol: str, amount: float) -> float:
        """Ajusta la cantidad a la precisión del mercado (evita rechazos)."""
        try:
            return float(self._exchange.amount_to_precision(symbol, amount))
        except (ccxt.BadSymbol, KeyError, ValueError):
            return float(amount)

    def _parse_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
        raw: dict[str, Any],
    ) -> OrderResult:
        """Normaliza la respuesta cruda de ccxt a un `OrderResult`."""
        filled = float(raw.get("filled") or 0.0)
        avg = raw.get("average") or raw.get("price") or 0.0
        return OrderResult(
            symbol=symbol,
            side=side,
            amount=amount,
            filled=filled if filled > 0 else amount,
            avg_price=float(avg),
            order_id=str(raw.get("id", "")),
            raw=raw,
        )

    def _register_fill(self, result: OrderResult, side: Side) -> None:
        """Actualiza el estado interno de posiciones tras un fill."""
        self._positions[result.symbol] = Position(
            symbol=result.symbol,
            side=side,
            quantity=result.filled,
            entry_price=result.avg_price,
            current_price=result.avg_price,
        )

    async def close_all(self) -> None:
        """Cierra todas las posiciones (invocado por el Kill Switch)."""
        raise NotImplementedError("ExecutionEngine se implementa en un paso posterior")
