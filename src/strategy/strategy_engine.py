"""StrategyEngine: arbitraje estadístico (Pairs Trading) adaptativo.

Consume flujos de order books de DOS activos y emite señales sobre el *spread*.
No ejecuta órdenes: delega en el ExecutionEngine y el sizing en el RiskManager.

Pipeline matemático adaptativo (por cada vela alineada de A y B):

    1.  Hedge ratio β dinámico vía Filtro de Kalman (vela a vela, sin
        look-ahead). El spread es el forecast error del filtro.
    2.  Z-Score con media/desviación estrictamente móviles (rolling window)
        sobre el spread — sin contaminación del futuro.
    3.  Gate de Half-Life (OU): si la reversión es demasiado rápida (ruido HF)
        o demasiado lenta, no se opera.
    4.  Umbral de entrada dinámico (Fee Hurdle): solo se abre si el Z-Score
        supera el coste de las 4 patas · margen -> gana a la fricción.
    5.  Máquina de estados de señales (evita reenvíos).

Rendimiento: buffers `deque` O(1) y Kalman incremental O(1) por vela.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.core.logger import get_logger
from src.core.models import OrderBook
from src.strategy.kalman import KalmanHedgeOnline
from src.strategy.signals import PairSignal, SignalType

log = get_logger("strategy_engine")

FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class PairConfig:
    """Parámetros de una estrategia de pairs trading adaptativa."""

    symbol_a: str
    symbol_b: str
    window: int = 300               # ventana móvil del Z-Score y half-life
    min_observations: int = 100     # mínimo para empezar a evaluar
    exit_band: float = 0.1          # banda alrededor de 0 para cerrar
    # Kalman
    kalman_delta: float = 1e-4      # velocidad de adaptación de β
    kalman_obs_cov: float = 2.0     # varianza del ruido de observación (R)
    # Gate de half-life (en velas)
    min_half_life: float = 5.0
    max_half_life: float = 1_440.0
    # Fee hurdle
    taker_fee: float = 0.001
    slippage: float = 0.0005
    hurdle_mult: float = 1.5
    n_legs: int = 4

    def __post_init__(self) -> None:
        if self.symbol_a == self.symbol_b:
            raise ValueError("symbol_a y symbol_b deben ser distintos")
        if self.min_observations < 30:
            raise ValueError("min_observations demasiado bajo para inferencia fiable")
        if self.min_observations > self.window:
            raise ValueError("min_observations no puede exceder window")


class StrategyEngine:
    """Motor de señales de arbitraje estadístico adaptativo (Kalman)."""

    def __init__(self, config: PairConfig) -> None:
        self._cfg = config

        # Filtro de Kalman incremental para el hedge ratio dinámico.
        self._kalman = KalmanHedgeOnline(
            delta=config.kalman_delta,
            obs_cov=config.kalman_obs_cov,
        )
        # Buffer del spread (forecast error) para el Z-Score y half-life rolling.
        self._spread_buf: deque[float] = deque(maxlen=config.window)
        self._kalman_ready = False

        # Último precio conocido de cada símbolo (alinear flujos asíncronos).
        self._last_a: float | None = None
        self._last_b: float | None = None

        # Estado de la posición sobre el spread (para no re-emitir señales).
        self._position: SignalType = SignalType.HOLD

        # Serializa el cálculo ante actualizaciones concurrentes de A y B.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Ingesta asíncrona desde el DataFetcher
    # ------------------------------------------------------------------ #
    async def on_order_book(self, book: OrderBook) -> PairSignal | None:
        """Recibe un snapshot y (si procede) emite una señal.

        Solo cuando hay tick nuevo de AMBOS activos se cierra una observación
        alineada, se actualiza el Kalman y se recalcula la señal.
        """
        mid = book.mid_price
        if mid is None or mid <= 0:
            return None

        async with self._lock:
            if book.symbol == self._cfg.symbol_a:
                self._last_a = mid
            elif book.symbol == self._cfg.symbol_b:
                self._last_b = mid
            else:
                return None

            if self._last_a is None or self._last_b is None:
                return None

            # Spread económico causal: usa β de la vela ANTERIOR (sin look-ahead).
            # No usamos la innovación del Kalman: por construcción es ruido
            # blanco y su half-life sería ~0.
            prev_beta = self._kalman.beta if self._kalman_ready else self._last_a / self._last_b
            spread = self._last_a - prev_beta * self._last_b

            # Ahora sí actualizamos el filtro con la observación actual.
            hedge_ratio, _innovation = self._kalman.update(self._last_a, self._last_b)
            self._kalman_ready = True
            self._spread_buf.append(spread)

            if len(self._spread_buf) < self._cfg.min_observations:
                return None

            return self._evaluate(hedge_ratio, spread)

    # ------------------------------------------------------------------ #
    #  Núcleo matemático
    # ------------------------------------------------------------------ #
    def _evaluate(self, hedge_ratio: float, spread: float) -> PairSignal:
        """Calcula Z-Score rolling, half-life y umbral dinámico -> señal."""
        buf = np.fromiter(self._spread_buf, dtype=np.float64)

        mean = buf.mean()
        std = buf.std(ddof=1)
        z_score = float((spread - mean) / std) if std > 0 and np.isfinite(std) else 0.0

        half_life = self._half_life(buf)
        entry_threshold = self._fee_hurdle_z(self._last_a or 0.0, std)

        half_life_ok = self._cfg.min_half_life <= half_life <= self._cfg.max_half_life
        is_tradeable = half_life_ok and np.isfinite(entry_threshold)

        signal = self._derive_signal(z_score, entry_threshold, is_tradeable)

        pair_signal = PairSignal(
            symbol_a=self._cfg.symbol_a,
            symbol_b=self._cfg.symbol_b,
            signal=signal,
            z_score=z_score,
            hedge_ratio=hedge_ratio,
            spread=spread,
            price_a=float(self._last_a or 0.0),
            price_b=float(self._last_b or 0.0),
            half_life=half_life,
            entry_threshold=entry_threshold,
            is_tradeable=is_tradeable,
        )

        if signal is not SignalType.HOLD:
            log.info(
                "senal_generada",
                signal=signal.value,
                z=round(z_score, 3),
                umbral=round(entry_threshold, 3),
                half_life=round(half_life, 1),
                hedge_ratio=round(hedge_ratio, 5),
            )
        return pair_signal

    @staticmethod
    def _half_life(spread: FloatArray) -> float:
        """Half-life de reversión (OU) sobre la ventana: −ln(2)/λ, en velas."""
        lag = spread[:-1]
        delta = spread[1:] - lag
        if lag.size < 2 or np.std(lag) == 0:
            return float("inf")
        # OLS: delta = intercepto + λ·lag  (np.polyfit -> [λ, intercepto]).
        lam, _ = np.polyfit(lag, delta, 1)
        if lam >= 0:
            return float("inf")
        return float(-np.log(2) / lam)

    def _fee_hurdle_z(self, price_a: float, std: float) -> float:
        """Umbral de entrada en Z que garantiza ganancia >= margen·coste.

        z_hurdle = hurdle_mult · n_legs · (fee + slippage) · P_A / σ_spread
        """
        cfg = self._cfg
        if std <= 0 or not np.isfinite(std):
            return float("inf")
        cost = cfg.taker_fee + cfg.slippage
        return (cfg.hurdle_mult * cfg.n_legs * cost) * (price_a / std)

    # ------------------------------------------------------------------ #
    #  Máquina de estados de señales (evita señales repetidas)
    # ------------------------------------------------------------------ #
    def _derive_signal(self, z: float, entry_z: float, is_tradeable: bool) -> SignalType:
        """Deriva la señal a partir del Z-Score y el umbral dinámico.

        Reglas:
            - Con posición: cierre al cruzar ~0 (banda exit_band) o si el gate
              (half-life/hurdle) deja de ser válido (invalidación del modelo).
            - Sin posición y gate válido:
                z >= +entry_z  -> SHORT_SPREAD  (spread caro)
                z <= -entry_z  -> LONG_SPREAD   (spread barato)
        """
        cfg = self._cfg

        if self._position in (SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD):
            if not is_tradeable:
                log.warning("gate_invalidado_cierre", z=round(z, 3))
                self._position = SignalType.HOLD
                return SignalType.EXIT
            if abs(z) <= cfg.exit_band:
                self._position = SignalType.HOLD
                return SignalType.EXIT
            return SignalType.HOLD

        # Sin posición: solo abrimos si el gate es válido y se supera el hurdle.
        if not is_tradeable:
            return SignalType.HOLD

        if z >= entry_z:
            self._position = SignalType.SHORT_SPREAD
            return SignalType.SHORT_SPREAD
        if z <= -entry_z:
            self._position = SignalType.LONG_SPREAD
            return SignalType.LONG_SPREAD

        return SignalType.HOLD

    # ------------------------------------------------------------------ #
    #  Introspección / utilidades
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> SignalType:
        """Estado actual de la posición sobre el spread."""
        return self._position

    @property
    def ready(self) -> bool:
        """True si hay suficientes observaciones para operar."""
        return len(self._spread_buf) >= self._cfg.min_observations
