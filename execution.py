"""execution.py — Motor dual (spot + perp) y gestor de margen (el salvavidas).

Contiene los dos módulos finales para operar Funding Rate Arbitrage delta-neutral:

    • DualMarketExecutionEngine: abre/cierra la posición cash-and-carry lanzando
      las DOS patas simultáneamente (comprar SPOT + vender PERP) con asyncio.gather.
      Como el Liquidity Gate ya validó el slippage, usa Market Orders para
      garantizar el fill de ambas patas y la neutralidad delta.

    • MarginManager: tarea de fondo que vigila el Margin Ratio del perpetuo corto.
      Si el subyacente sube y el uso de margen se acerca a la liquidación, mueve
      USDT del wallet SPOT al de FUTUROS (ccxt.transfer) para alejar el precio de
      liquidación y preservar la neutralidad.

Reutiliza el patrón robusto de reintentos/backoff del ExecutionEngine original.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import ccxt  # excepciones tipadas
import ccxt.async_support as ccxt_async

from src.core.logger import get_logger
from src.core.models import Side
from src.risk.risk_manager import RiskManager

log = get_logger("dual_execution")

_RETRYABLE = (
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
    ccxt.RequestTimeout,
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
)


# --------------------------------------------------------------------------- #
#  Modelos
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ExecutableSignal:
    """Señal 🟢 EXECUTABLE validada por radar + liquidity gate."""

    perp_symbol: str          # p.ej. "BTC/USDT:USDT"
    spot_symbol: str          # p.ej. "BTC/USDT"
    notional_usd: float       # tamaño objetivo por pata
    funding_apr: float
    breakeven_days: float


@dataclass(slots=True)
class CarryPosition:
    """Posición cash-and-carry abierta (delta-neutral)."""

    perp_symbol: str
    spot_symbol: str
    quantity: float           # unidades base (iguales en ambas patas)
    spot_entry: float
    perp_entry: float
    is_open: bool = True


@dataclass(slots=True)
class ExecutionConfig:
    max_retries: int = 4
    base_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    order_timeout_s: float = 10.0


@dataclass(slots=True)
class MarginConfig:
    """Parámetros del salvavidas de margen."""

    check_interval_s: float = 15.0     # frecuencia del loop de vigilancia
    danger_ratio: float = 0.80         # uso de margen que dispara el top-up
    target_ratio: float = 0.60         # ratio seguro objetivo tras recapitalizar
    min_spot_reserve_usd: float = 100.0  # colchón que NO se transfiere
    max_topups: int = 20               # límite de seguridad por posición


# --------------------------------------------------------------------------- #
#  DualMarketExecutionEngine
# --------------------------------------------------------------------------- #
class DualMarketExecutionEngine:
    """Ejecuta y cierra la posición delta-neutral spot+perp de forma atómica."""

    def __init__(
        self,
        spot_exchange: ccxt_async.Exchange,
        perp_exchange: ccxt_async.Exchange,
        risk_manager: RiskManager,
        config: ExecutionConfig | None = None,
    ) -> None:
        # Inyección de dependencias: dos exchanges (spot/derivados) + riesgo.
        self._spot = spot_exchange
        self._perp = perp_exchange
        self._risk = risk_manager
        self._cfg = config or ExecutionConfig()
        self._position: CarryPosition | None = None

    @property
    def position(self) -> CarryPosition | None:
        return self._position

    async def open_position(self, signal: ExecutableSignal) -> CarryPosition | None:
        """Abre la posición: COMPRAR spot + VENDER perp simultáneamente."""
        if self._risk.is_halted:
            log.warning("apertura_bloqueada_killswitch")
            return None
        if self._position is not None and self._position.is_open:
            log.warning("posicion_ya_abierta", perp=self._position.perp_symbol)
            return None

        # Precio de referencia y cantidad (misma qty base en ambas patas).
        ref_price = await self._ref_price(signal.perp_symbol)
        if ref_price <= 0:
            return None
        qty = self._normalize(self._spot, signal.spot_symbol, signal.notional_usd / ref_price)
        if qty <= 0:
            log.warning("cantidad_invalida", qty=qty)
            return None

        log.info("abriendo_carry", perp=signal.perp_symbol, qty=qty, apr=signal.funding_apr)

        # Ejecución cuasi-atómica de ambas patas.
        spot_res, perp_res = await asyncio.gather(
            self._safe_market_order(self._spot, signal.spot_symbol, Side.BUY, qty),
            self._safe_market_order(self._perp, signal.perp_symbol, Side.SELL, qty),
            return_exceptions=True,
        )

        ok_spot = not isinstance(spot_res, BaseException)
        ok_perp = not isinstance(perp_res, BaseException)

        # Gestión de fallo de pata -> volver a neutral (delta) inmediatamente.
        if ok_spot and ok_perp:
            self._position = CarryPosition(
                perp_symbol=signal.perp_symbol,
                spot_symbol=signal.spot_symbol,
                quantity=qty,
                spot_entry=self._fill_price(spot_res, ref_price),
                perp_entry=self._fill_price(perp_res, ref_price),
            )
            log.info("carry_abierto", perp=signal.perp_symbol, qty=qty)
            return self._position

        await self._unwind_partial(signal, ok_spot, ok_perp, qty, spot_res, perp_res)
        return None

    async def close_position(self) -> None:
        """Cierra la posición: VENDER spot + COMPRAR perp simultáneamente."""
        pos = self._position
        if pos is None or not pos.is_open:
            return
        log.info("cerrando_carry", perp=pos.perp_symbol, qty=pos.quantity)
        await asyncio.gather(
            self._safe_market_order(self._spot, pos.spot_symbol, Side.SELL, pos.quantity),
            self._safe_market_order(self._perp, pos.perp_symbol, Side.BUY, pos.quantity),
            return_exceptions=True,
        )
        pos.is_open = False
        log.info("carry_cerrado", perp=pos.perp_symbol)

    async def _unwind_partial(
        self,
        signal: ExecutableSignal,
        ok_spot: bool,
        ok_perp: bool,
        qty: float,
        spot_res: Any,
        perp_res: Any,
    ) -> None:
        """Deshace la pata que sí entró si la otra falló (evita delta abierto)."""
        if not ok_spot and not ok_perp:
            log.error("ambas_patas_fallaron", spot=str(spot_res), perp=str(perp_res))
            return
        if ok_spot and not ok_perp:
            log.error("leg_risk_perp_fallo_deshaciendo_spot", error=str(perp_res))
            try:
                await self._safe_market_order(self._spot, signal.spot_symbol, Side.SELL, qty)
            except Exception as exc:  # noqa: BLE001 - último recurso
                log.critical("fallo_deshacer_spot_EXPOSICION", error=str(exc))
                self._risk.trip_kill_switch(reason="carry_unwind_spot_failed")
        elif ok_perp and not ok_spot:
            log.error("leg_risk_spot_fallo_deshaciendo_perp", error=str(spot_res))
            try:
                await self._safe_market_order(self._perp, signal.perp_symbol, Side.BUY, qty)
            except Exception as exc:  # noqa: BLE001
                log.critical("fallo_deshacer_perp_EXPOSICION", error=str(exc))
                self._risk.trip_kill_switch(reason="carry_unwind_perp_failed")

    # ------------------------------------------------------------------ #
    #  Envío de órdenes con reintentos + backoff
    # ------------------------------------------------------------------ #
    async def _safe_market_order(
        self,
        exchange: ccxt_async.Exchange,
        symbol: str,
        side: Side,
        amount: float,
    ) -> dict[str, Any]:
        backoff = self._cfg.base_backoff_s
        last_exc: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    exchange.create_order(symbol, "market", side.value, amount),
                    timeout=self._cfg.order_timeout_s,
                )
            except ccxt.InsufficientFunds as exc:
                log.error("fondos_insuficientes", symbol=symbol, error=str(exc))
                raise
            except (_RETRYABLE, asyncio.TimeoutError) as exc:
                last_exc = exc
                log.warning("orden_reintento", symbol=symbol, intento=attempt,
                            error=type(exc).__name__, backoff_s=round(backoff, 2))
                if attempt < self._cfg.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.max_backoff_s)
            except ccxt.ExchangeError as exc:
                log.error("error_exchange_orden", symbol=symbol, error=str(exc))
                raise
        raise RuntimeError(f"orden fallida {symbol} tras {self._cfg.max_retries} intentos: {last_exc}")

    # ------------------------------------------------------------------ #
    #  Utilidades
    # ------------------------------------------------------------------ #
    async def _ref_price(self, perp_symbol: str) -> float:
        try:
            ticker = await self._perp.fetch_ticker(perp_symbol)
            return float(ticker.get("last") or ticker.get("close") or 0.0)
        except ccxt.BaseError as exc:
            log.error("fallo_ref_price", symbol=perp_symbol, error=str(exc))
            return 0.0

    def _normalize(self, exchange: ccxt_async.Exchange, symbol: str, amount: float) -> float:
        try:
            return float(exchange.amount_to_precision(symbol, amount))
        except (ccxt.BadSymbol, KeyError, ValueError):
            return float(amount)

    @staticmethod
    def _fill_price(result: Any, fallback: float) -> float:
        if isinstance(result, dict):
            return float(result.get("average") or result.get("price") or fallback)
        return fallback


# --------------------------------------------------------------------------- #
#  MarginManager — el salvavidas anti-liquidación
# --------------------------------------------------------------------------- #
class MarginManager:
    """Vigila el margin ratio del perp corto y recapitaliza antes de liquidar."""

    def __init__(
        self,
        spot_exchange: ccxt_async.Exchange,
        perp_exchange: ccxt_async.Exchange,
        risk_manager: RiskManager,
        config: MarginConfig | None = None,
    ) -> None:
        self._spot = spot_exchange
        self._perp = perp_exchange
        self._risk = risk_manager
        self._cfg = config or MarginConfig()
        self._running = False
        self._topups = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Lanza el loop de vigilancia como tarea de fondo."""
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop(), name="margin-manager")
            log.info("margin_manager_iniciado", danger_ratio=self._cfg.danger_ratio)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Bucle: mide el margin ratio y recapitaliza si entra en zona de peligro."""
        while self._running:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - el vigilante nunca debe morir
                log.error("margin_monitor_error", error=str(exc), exc_info=True)
            await asyncio.sleep(self._cfg.check_interval_s)

    async def check_once(self) -> float | None:
        """Ejecuta UN ciclo de vigilancia: mide y, si hay peligro, recapitaliza.

        Devuelve el margin ratio observado (antes del top-up), o None si no se
        pudo leer. Aislado del loop para poder testearlo de forma determinista.
        """
        data = await self._read_margin()
        if data is None:
            return None
        ratio, maint, margin_balance = data
        log.info("margin_ratio", ratio=round(ratio, 4))
        if ratio >= self._cfg.danger_ratio:
            await self._top_up_margin(maint, margin_balance, ratio)
        return ratio

    async def _read_margin(self) -> tuple[float, float, float] | None:
        """Lee (ratio, maintMargin, marginBalance) de la cuenta de futuros.

        ratio = maintMargin / marginBalance; -> 1.0 significa liquidación.
        """
        try:
            balance = await self._perp.fetch_balance()
        except ccxt.BaseError as exc:
            log.warning("fallo_fetch_balance_futuros", error=str(exc))
            return None

        info = balance.get("info", {}) or {}
        maint = _to_float(info.get("totalMaintMargin"))
        margin_balance = _to_float(info.get("totalMarginBalance"))
        if margin_balance and margin_balance > 0:
            return maint / margin_balance, maint, margin_balance

        # Fallback: si el exchange no expone maint, aproxima con used/total.
        usdt = balance.get("USDT", {}) or {}
        used, total = _to_float(usdt.get("used")), _to_float(usdt.get("total"))
        if total and total > 0:
            return used / total, used, total
        return None

    async def get_margin_ratio(self) -> float | None:
        """Margin ratio actual de la cuenta de futuros (o None)."""
        data = await self._read_margin()
        return data[0] if data is not None else None

    async def _top_up_margin(self, maint: float, margin_balance: float, ratio: float) -> None:
        """Transfiere el importe EXACTO de SPOT a FUTUROS para volver al target.

        Como el margin ratio = maint / marginBalance, para llevarlo a
        `target_ratio` hay que añadir X al marginBalance tal que:

            maint / (marginBalance + X) = target_ratio
            =>  X = maint / target_ratio − marginBalance
        """
        if self._topups >= self._cfg.max_topups:
            log.critical("max_topups_alcanzado_KILLSWITCH", topups=self._topups)
            self._risk.trip_kill_switch(reason="margin_topups_exhausted")
            return

        required = maint / self._cfg.target_ratio - margin_balance
        if required <= 0:
            return

        available = await self._available_spot()
        amount = round(min(required, available), 2)
        if amount <= 0:
            log.critical("sin_reserva_spot_para_topup_KILLSWITCH", ratio=round(ratio, 4))
            self._risk.trip_kill_switch(reason="no_spot_reserve_for_margin")
            return

        try:
            # ccxt unificado: mover USDT del wallet spot al de futuros lineales.
            await self._perp.transfer("USDT", amount, "spot", "future")
            self._topups += 1
            log.warning(
                "margin_topup_ejecutado",
                amount_usd=amount,
                ratio_previo=round(ratio, 4),
                target=self._cfg.target_ratio,
                topup_num=self._topups,
            )
        except ccxt.BaseError as exc:
            log.error("fallo_transfer_margen", amount=amount, error=str(exc))

    async def _available_spot(self) -> float:
        """USDT libre en spot descontando la reserva mínima intocable."""
        try:
            spot_balance = await self._spot.fetch_balance()
        except ccxt.BaseError:
            return 0.0
        free = _to_float((spot_balance.get("USDT", {}) or {}).get("free"))
        return max(0.0, free - self._cfg.min_spot_reserve_usd)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
