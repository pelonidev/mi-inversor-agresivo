"""liquidity_gate.py — Filtro de ejecutabilidad (el espejismo de la liquidez).

Somete cada señal 🟢 GO del radar a un escrutinio profundo antes de considerarla
operable delta-neutral:

    1.  Spot-Perp match: debe existir el par SPOT exacto en USDT.
    2.  Anti-honeypot: volumen 24h de AMBAS patas > MIN_VOLUME_USD.
    3.  Slippage real barriendo el order book L2 con una orden de test_order_size.
    4.  Break-even real = taker (×2) + slippage spot + slippage perp.

Solo sobrevive como 🟢 EXECUTABLE si el break-even real < MAX_BREAKEVEN_DAYS.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

import ccxt.async_support as ccxt_async

# --- Parámetros del filtro ---
TEST_ORDER_SIZE_USD = 5_000.0
MIN_VOLUME_USD = 2_000_000.0
TAKER_FEE = 0.001                 # 0.1% por pata
ROUND_TRIP_TAKER = 4 * TAKER_FEE  # spot+perp en entrada y salida
MAX_BREAKEVEN_DAYS = 5.0
ORDERBOOK_DEPTH = 50
CALL_TIMEOUT_S = 15.0


class LiquidityStatus(str, Enum):
    EXECUTABLE = "🟢 EXECUTABLE"
    NO_SPOT = "🔴 NO SPOT"
    LOW_VOL = "🔴 LOW VOL"
    NO_BOOK = "🔴 NO BOOK"
    SLIPPAGE = "🔴 SLIPPAGE"       # break-even real > umbral


@dataclass(slots=True)
class LiquidityReport:
    symbol_perp: str
    symbol_spot: str
    status: LiquidityStatus
    spot_volume_usd: float = 0.0
    perp_volume_usd: float = 0.0
    slippage_spot: float = 0.0        # fracción (0.001 = 0.1%)
    slippage_perp: float = 0.0
    real_cost: float = 0.0            # fricción total round-trip (fracción)
    breakeven_days_real: float = float("inf")


def _spot_symbol(perp_symbol: str) -> str:
    """MARSCOIN/USDT:USDT -> MARSCOIN/USDT (quita el settle del perp)."""
    return perp_symbol.split(":")[0]


def simulate_market_slippage(book_side: list[list[float]], size_usd: float) -> float | None:
    """Barre un lado del libro (asks para comprar, bids para vender).

    Devuelve el slippage como fracción: |precio_medio − mejor_precio| / mejor_precio.
    None si el libro no tiene profundidad suficiente para `size_usd`.
    """
    if not book_side:
        return None
    best_price = book_side[0][0]
    remaining_usd = size_usd
    units = 0.0
    for price, amount in book_side:
        level_usd = price * amount
        take_usd = min(remaining_usd, level_usd)
        units += take_usd / price          # unidades compradas/vendidas en este nivel
        remaining_usd -= take_usd
        if remaining_usd <= 0:
            break
    if remaining_usd > 0 or units <= 0:
        return None                        # libro demasiado fino para la orden
    avg_price = size_usd / units
    return abs(avg_price - best_price) / best_price


async def evaluate_liquidity(
    exchange_spot: ccxt_async.Exchange,
    exchange_perp: ccxt_async.Exchange,
    perp_symbol: str,
    spot_markets: dict,
    tickers_spot: dict,
    tickers_perp: dict,
) -> LiquidityReport:
    """Escrutinio completo de una señal GO. `exchange_*` ya con markets cargados."""
    spot_symbol = _spot_symbol(perp_symbol)
    report = LiquidityReport(perp_symbol, spot_symbol, LiquidityStatus.NO_SPOT)

    # 1) Spot-Perp match.
    if spot_symbol not in spot_markets:
        return report

    # 2) Volumen 24h de ambas patas.
    spot_vol = (tickers_spot.get(spot_symbol, {}) or {}).get("quoteVolume") or 0.0
    perp_vol = (tickers_perp.get(perp_symbol, {}) or {}).get("quoteVolume") or 0.0
    report.spot_volume_usd = float(spot_vol)
    report.perp_volume_usd = float(perp_vol)
    if spot_vol < MIN_VOLUME_USD or perp_vol < MIN_VOLUME_USD:
        report.status = LiquidityStatus.LOW_VOL
        return report

    # 3) Slippage real barriendo el order book de ambas patas.
    try:
        spot_book, perp_book = await asyncio.gather(
            asyncio.wait_for(exchange_spot.fetch_order_book(spot_symbol, ORDERBOOK_DEPTH), CALL_TIMEOUT_S),
            asyncio.wait_for(exchange_perp.fetch_order_book(perp_symbol, ORDERBOOK_DEPTH), CALL_TIMEOUT_S),
        )
    except (ccxt_async.BaseError, asyncio.TimeoutError):
        report.status = LiquidityStatus.NO_BOOK
        return report

    # Delta-neutral: COMPRAR spot (asks) + VENDER perp (bids).
    slip_spot = simulate_market_slippage(spot_book.get("asks", []), TEST_ORDER_SIZE_USD)
    slip_perp = simulate_market_slippage(perp_book.get("bids", []), TEST_ORDER_SIZE_USD)
    if slip_spot is None or slip_perp is None:
        report.status = LiquidityStatus.NO_BOOK
        return report

    report.slippage_spot = slip_spot
    report.slippage_perp = slip_perp
    report.real_cost = ROUND_TRIP_TAKER + slip_spot + slip_perp
    return report


def real_breakeven_days(rate_ma: float, interval_h: float, real_cost: float) -> float:
    """Break-even en días usando la fricción REAL (taker + slippage)."""
    if rate_ma <= 0:
        return float("inf")
    epochs_needed = real_cost / rate_ma
    return epochs_needed * interval_h / 24.0


def finalize_status(report: LiquidityReport, breakeven_days: float) -> LiquidityReport:
    """Asigna EXECUTABLE/SLIPPAGE según el break-even real."""
    report.breakeven_days_real = breakeven_days
    if report.status in (LiquidityStatus.NO_SPOT, LiquidityStatus.LOW_VOL, LiquidityStatus.NO_BOOK):
        return report
    report.status = (
        LiquidityStatus.EXECUTABLE if breakeven_days < MAX_BREAKEVEN_DAYS
        else LiquidityStatus.SLIPPAGE
    )
    return report
