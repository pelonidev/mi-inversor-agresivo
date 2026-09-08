"""Modelos de señales emitidas por el StrategyEngine (DTOs).

El motor de estrategia NO ejecuta órdenes: solo emite estas señales, que el
ExecutionEngine consumirá en el paso siguiente. Mantener las señales como
dataclasses inmutables facilita el logging, el backtesting y las pruebas.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SignalType(str, Enum):
    """Tipo de señal de trading para un par cointegrado.

    Convención del "spread" = Precio A - HedgeRatio * Precio B:
        - LONG_SPREAD  -> spread barato: COMPRAR A, VENDER B.
        - SHORT_SPREAD -> spread caro:   VENDER A, COMPRAR B.
        - EXIT         -> cerrar cualquier posición abierta del par.
        - HOLD         -> sin acción.
    """

    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"
    EXIT = "exit"
    HOLD = "hold"


@dataclass(slots=True, frozen=True)
class PairSignal:
    """Señal de arbitraje estadístico para un par de activos."""

    symbol_a: str
    symbol_b: str
    signal: SignalType
    z_score: float
    hedge_ratio: float
    spread: float
    price_a: float
    price_b: float
    half_life: float = float("inf")     # velas hasta reversión (OU)
    entry_threshold: float = 2.0        # umbral dinámico (fee hurdle) en Z
    is_tradeable: bool = False          # gate: half-life válido y hurdle superado
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def is_actionable(self) -> bool:
        """True si la señal implica abrir o cerrar posiciones."""
        return self.signal in (
            SignalType.LONG_SPREAD,
            SignalType.SHORT_SPREAD,
            SignalType.EXIT,
        )
