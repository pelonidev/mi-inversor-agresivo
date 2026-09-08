"""Modelos de datos compartidos entre módulos (DTOs).

Se usan dataclasses con `slots=True` por rendimiento: en un bucle HFT se crean
miles de objetos por segundo y `__slots__` reduce memoria y acelera el acceso.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass(slots=True, frozen=True)
class PriceLevel:
    """Un nivel de precio dentro del libro de órdenes."""

    price: float
    amount: float


@dataclass(slots=True)
class OrderBook:
    """Snapshot del libro de órdenes (top-N) para un símbolo.

    `bids` y `asks` están ordenados: bids de mayor a menor precio,
    asks de menor a mayor precio.
    """

    symbol: str
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float | None:
        """Spread en puntos básicos (basis points) respecto al mid."""
        mid = self.mid_price
        if mid is None or self.spread is None or mid == 0:
            return None
        return (self.spread / mid) * 10_000.0

    @property
    def age_ms(self) -> float:
        """Antigüedad del snapshot en milisegundos (control de latencia)."""
        return (time.time() - self.timestamp) * 1_000.0


@dataclass(slots=True)
class Position:
    """Posición abierta en un instrumento."""

    symbol: str
    side: Side
    quantity: float          # unidades del activo base
    entry_price: float       # precio medio de entrada
    current_price: float = 0.0

    @property
    def notional(self) -> float:
        """Valor nocional absoluto en la divisa de cotización (USD)."""
        price = self.current_price or self.entry_price
        return abs(self.quantity) * price

    @property
    def signed_notional(self) -> float:
        """Nocional con signo: positivo si long, negativo si short."""
        sign = 1.0 if self.side is Side.BUY else -1.0
        return sign * self.notional

    @property
    def unrealized_pnl(self) -> float:
        if self.current_price == 0.0:
            return 0.0
        direction = 1.0 if self.side is Side.BUY else -1.0
        return direction * (self.current_price - self.entry_price) * abs(self.quantity)
