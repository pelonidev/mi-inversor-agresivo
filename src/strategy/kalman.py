"""Filtro de Kalman para hedge ratio adaptativo (sin look-ahead bias).

Modela la relación entre dos activos como una regresión lineal cuyo estado
[β, α] evoluciona en el tiempo (random walk). En cada vela:

    observación:   A_t = β_t · B_t + α_t + ruido        (R = obs_cov)
    transición:    x_t = x_{t-1} + w,   Q = δ/(1-δ) · I  (β/α derivan lento)

El *forecast error* e_t = A_t − [B_t, 1]·x_{t-1} es el spread instantáneo y
depende SOLO del pasado (estado previo), por lo que es libre de contaminación
del futuro. δ (delta) controla cómo de rápido se adapta β: pequeño = suave.

Se ofrecen dos APIs:
    - `kalman_hedge_batch`: recorrido vectorizado/numba para backtesting.
    - `KalmanHedgeOnline`: actualización incremental vela a vela para el
      StrategyEngine en tiempo real.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

FloatArray = npt.NDArray[np.float64]


@njit(cache=True)
def _kalman_core(
    a: FloatArray,
    b: FloatArray,
    delta: float,
    obs_cov: float,
    p0: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Recursión de Kalman (jit) devolviendo beta, alpha y spread por vela."""
    n = a.shape[0]
    beta = np.zeros(n)
    alpha = np.zeros(n)
    spread = np.zeros(n)

    # Estado x=[beta, alpha]; beta inicial = ratio de la primera vela.
    x0 = a[0] / b[0] if b[0] != 0.0 else 1.0
    x = np.array([x0, 0.0])

    # Covarianza del estado y del proceso (Q) y observación (R).
    p = np.array([[p0, 0.0], [0.0, p0]])
    q = (delta / (1.0 - delta)) * np.eye(2)

    for t in range(n):
        # Predicción del estado (random walk): x_pred = x; P_pred = P + Q.
        p = p + q

        h0 = b[t]   # matriz de observación H = [B_t, 1]
        h1 = 1.0

        # Error de predicción (spread instantáneo), solo usa estado previo.
        e = a[t] - (h0 * x[0] + h1 * x[1])

        # Covarianza de innovación S = H P Hᵀ + R.
        hp0 = h0 * p[0, 0] + h1 * p[1, 0]
        hp1 = h0 * p[0, 1] + h1 * p[1, 1]
        s = hp0 * h0 + hp1 * h1 + obs_cov

        # Ganancia de Kalman K = P Hᵀ / S.
        k0 = (p[0, 0] * h0 + p[0, 1] * h1) / s
        k1 = (p[1, 0] * h0 + p[1, 1] * h1) / s

        # Actualización del estado.
        x[0] = x[0] + k0 * e
        x[1] = x[1] + k1 * e

        # Actualización de la covarianza P = P − K (H P).
        p00 = p[0, 0] - k0 * hp0
        p01 = p[0, 1] - k0 * hp1
        p10 = p[1, 0] - k1 * hp0
        p11 = p[1, 1] - k1 * hp1
        p[0, 0] = p00
        p[0, 1] = p01
        p[1, 0] = p10
        p[1, 1] = p11

        beta[t] = x[0]
        alpha[t] = x[1]
        spread[t] = e

    return beta, alpha, spread


def kalman_hedge_batch(
    a: FloatArray,
    b: FloatArray,
    delta: float = 1e-4,
    obs_cov: float = 2.0,
    p0: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Ejecuta el filtro de Kalman sobre series completas (backtesting).

    Parámetros:
        a, b:    arrays de precios de los activos A y B (mismo tamaño).
        delta:   velocidad de adaptación de β (1e-5 suave .. 1e-3 reactivo).
        obs_cov: varianza del ruido de observación (R).
        p0:      incertidumbre inicial del estado.

    Devuelve (beta, alpha, spread), todos de longitud len(a).
    """
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    if a64.shape != b64.shape:
        raise ValueError("a y b deben tener la misma longitud")
    return _kalman_core(a64, b64, delta, obs_cov, p0)


class KalmanHedgeOnline:
    """Filtro de Kalman incremental para uso en tiempo real (una vela a la vez)."""

    __slots__ = ("_x", "_p", "_q", "_r", "_initialized")

    def __init__(self, delta: float = 1e-4, obs_cov: float = 2.0, p0: float = 1.0) -> None:
        self._x = np.zeros(2)                       # [beta, alpha]
        self._p = np.eye(2) * p0
        self._q = (delta / (1.0 - delta)) * np.eye(2)
        self._r = obs_cov
        self._initialized = False

    def update(self, price_a: float, price_b: float) -> tuple[float, float]:
        """Procesa una nueva vela y devuelve (beta, spread) actualizados.

        `spread` es el forecast error (usa el estado previo -> sin look-ahead).
        """
        if not self._initialized:
            self._x[0] = price_a / price_b if price_b != 0.0 else 1.0
            self._initialized = True

        # Predicción.
        self._p = self._p + self._q

        h = np.array([price_b, 1.0])
        e = float(price_a - h @ self._x)            # spread instantáneo

        s = float(h @ self._p @ h) + self._r        # innovación
        k = (self._p @ h) / s                       # ganancia

        self._x = self._x + k * e
        self._p = self._p - np.outer(k, h @ self._p)

        return float(self._x[0]), e

    @property
    def beta(self) -> float:
        return float(self._x[0])
