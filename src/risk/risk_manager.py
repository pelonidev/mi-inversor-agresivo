"""RiskManager: el guardián del capital.

Responsabilidades (ninguna estrategia opera sin pasar por aquí):

1.  **Position Sizing (Criterio de Kelly fraccional)**: calcula cuánto capital
    arriesgar por operación en función de la ventaja estadística (edge) y la
    volatilidad, aplicando una fracción de Kelly para reducir la varianza.
2.  **Control de exposición**: vigila la exposición bruta (gross) y neta (net).
    En estadística de arbitraje / market making buscamos exposición NETA ~ 0
    (beta neutral): la suma de longs y shorts debe cancelarse.
3.  **Kill Switch**: si el drawdown diario alcanza el umbral (por defecto 1%),
    marca el sistema como HALTED para cerrar todo y detener el bot.

Diseñado para ser *thread/async safe* en lectura: los métodos de cálculo son
puros; el estado mutable (equity, posiciones) se actualiza de forma explícita.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.core.logger import get_logger
from src.core.models import Position

log = get_logger("risk_manager")


class RiskState(str, Enum):
    ACTIVE = "active"        # Operativa normal
    REDUCE_ONLY = "reduce"   # Solo se permite reducir posiciones
    HALTED = "halted"        # Kill Switch disparado: cerrar todo y parar


@dataclass(slots=True)
class RiskLimits:
    """Límites duros de riesgo. Inmutables durante la sesión."""

    account_equity_usd: float
    max_daily_drawdown_pct: float = 0.01   # 1%
    kelly_fraction: float = 0.25           # 1/4 Kelly
    max_gross_exposure_pct: float = 2.0    # 200% del equity
    max_position_pct: float = 0.10         # 10% del equity por posición
    max_net_exposure_pct: float = 0.05     # 5%: tolerancia de desviación de beta-neutral

    def __post_init__(self) -> None:
        if self.account_equity_usd <= 0:
            raise ValueError("account_equity_usd debe ser > 0")
        if not 0 < self.max_daily_drawdown_pct < 1:
            raise ValueError("max_daily_drawdown_pct debe estar en (0, 1)")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction debe estar en (0, 1]")


class RiskManager:
    """Gestor central de riesgo. Única fuente de verdad sobre el capital."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._state: RiskState = RiskState.ACTIVE

        # Marcas de agua para el drawdown diario.
        self._day_start_equity: float = limits.account_equity_usd
        self._current_equity: float = limits.account_equity_usd
        self._peak_equity: float = limits.account_equity_usd
        self._session_day: int = self._utc_day()

    # ------------------------------------------------------------------ #
    #  Propiedades de estado
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def is_halted(self) -> bool:
        return self._state is RiskState.HALTED

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def daily_pnl(self) -> float:
        return self._current_equity - self._day_start_equity

    @property
    def daily_drawdown_pct(self) -> float:
        """Drawdown desde el pico de la sesión, como fracción positiva."""
        if self._peak_equity <= 0:
            return 0.0
        dd = (self._peak_equity - self._current_equity) / self._peak_equity
        return max(0.0, dd)

    # ------------------------------------------------------------------ #
    #  1. Position Sizing — Criterio de Kelly fraccional
    # ------------------------------------------------------------------ #
    def kelly_fraction_of_capital(
        self,
        win_prob: float,
        win_loss_ratio: float,
    ) -> float:
        """Fracción de capital a arriesgar según Kelly.

        Fórmula de Kelly para apuestas asimétricas:

            f* = W - (1 - W) / R

        donde:
            W = probabilidad de ganar (0..1)
            R = ratio ganancia media / pérdida media (payoff)

        Se multiplica por `kelly_fraction` (fraccional Kelly) para reducir
        la varianza y el riesgo de ruina. Se acota a [0, max_position_pct].

        Devuelve una fracción del equity (p. ej. 0.03 = 3%).
        """
        if not 0.0 <= win_prob <= 1.0:
            raise ValueError("win_prob debe estar en [0, 1]")
        if win_loss_ratio <= 0:
            raise ValueError("win_loss_ratio debe ser > 0")

        edge = win_prob - (1.0 - win_prob) / win_loss_ratio
        if edge <= 0:
            # Sin ventaja matemática -> no operar.
            return 0.0

        f = edge * self._limits.kelly_fraction
        return float(min(f, self._limits.max_position_pct))

    def position_size(
        self,
        win_prob: float,
        win_loss_ratio: float,
        price: float,
        stop_distance: float | None = None,
    ) -> float:
        """Calcula el tamaño de la posición en unidades del activo base.

        Parámetros:
            win_prob:        probabilidad estimada de acierto de la señal.
            win_loss_ratio:  ratio payoff (ganancia media / pérdida media).
            price:           precio actual del activo.
            stop_distance:   distancia al stop en unidades de precio. Si se
                             indica, el sizing se basa en riesgo por unidad
                             (capital en riesgo / riesgo por unidad). Si no,
                             se usa nocional directo.

        Devuelve la cantidad (base) a operar, ya acotada por los límites.
        """
        if price <= 0:
            raise ValueError("price debe ser > 0")
        if self._state is not RiskState.ACTIVE:
            log.warning("sizing_bloqueado", state=self._state.value)
            return 0.0

        kelly_frac = self.kelly_fraction_of_capital(win_prob, win_loss_ratio)
        capital_at_risk = self._current_equity * kelly_frac
        if capital_at_risk <= 0:
            return 0.0

        if stop_distance and stop_distance > 0:
            # Sizing basado en riesgo: nº unidades = riesgo$ / riesgo por unidad.
            quantity = capital_at_risk / stop_distance
        else:
            # Sizing nocional: capital asignado / precio.
            quantity = capital_at_risk / price

        # Acotar por nocional máximo de posición.
        max_notional = self._current_equity * self._limits.max_position_pct
        if quantity * price > max_notional:
            quantity = max_notional / price

        return float(max(0.0, quantity))

    # ------------------------------------------------------------------ #
    #  2. Control de exposición (gross / net / beta-neutral)
    # ------------------------------------------------------------------ #
    @staticmethod
    def gross_exposure(positions: list[Position]) -> float:
        """Suma de nocionales absolutos (longs + shorts)."""
        return sum(p.notional for p in positions)

    @staticmethod
    def net_exposure(positions: list[Position]) -> float:
        """Nocional con signo (longs - shorts). ~0 => beta neutral."""
        return sum(p.signed_notional for p in positions)

    def check_exposure(self, positions: list[Position]) -> bool:
        """True si la exposición está dentro de los límites permitidos."""
        gross = self.gross_exposure(positions)
        net = abs(self.net_exposure(positions))

        max_gross = self._current_equity * self._limits.max_gross_exposure_pct
        max_net = self._current_equity * self._limits.max_net_exposure_pct

        gross_ok = gross <= max_gross
        net_ok = net <= max_net

        if not gross_ok:
            log.warning("exposicion_bruta_excedida", gross=gross, limit=max_gross)
        if not net_ok:
            log.warning(
                "desviacion_beta_neutral",
                net=net,
                limit=max_net,
                hint="posiciones descompensadas: revisar hedge",
            )
        return gross_ok and net_ok

    def can_open(self, new_position: Position, current: list[Position]) -> bool:
        """Comprueba si abrir `new_position` respeta los límites y el estado."""
        if self._state is RiskState.HALTED:
            return False
        if self._state is RiskState.REDUCE_ONLY:
            log.info("solo_reduccion_activa")
            return False
        return self.check_exposure([*current, new_position])

    # ------------------------------------------------------------------ #
    #  3. Kill Switch y actualización de equity
    # ------------------------------------------------------------------ #
    def update_equity(self, equity: float) -> RiskState:
        """Actualiza el equity actual y reevalúa el Kill Switch.

        Debe llamarse en cada ciclo de marca-a-mercado (mark-to-market).
        Devuelve el nuevo `RiskState`.
        """
        if equity < 0:
            equity = 0.0

        # Reinicio automático al cambiar de día (UTC).
        today = self._utc_day()
        if today != self._session_day:
            self._reset_daily(equity, today)

        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        return self._evaluate_kill_switch()

    def _evaluate_kill_switch(self) -> RiskState:
        """Dispara HALTED si el drawdown diario supera el umbral."""
        if self._state is RiskState.HALTED:
            return self._state

        dd = self.daily_drawdown_pct
        if dd >= self._limits.max_daily_drawdown_pct:
            self._state = RiskState.HALTED
            log.critical(
                "KILL_SWITCH_ACTIVADO",
                drawdown_pct=round(dd * 100, 4),
                limite_pct=round(self._limits.max_daily_drawdown_pct * 100, 4),
                equity=round(self._current_equity, 2),
                pico=round(self._peak_equity, 2),
                accion="CERRAR_TODO_Y_DETENER",
            )
        return self._state

    def trip_kill_switch(self, reason: str) -> None:
        """Dispara manualmente el Kill Switch (p. ej. desconexión, error API)."""
        self._state = RiskState.HALTED
        log.critical("KILL_SWITCH_MANUAL", reason=reason, accion="CERRAR_TODO")

    def set_reduce_only(self, reason: str) -> None:
        """Pasa a modo REDUCE_ONLY sin llegar a apagar el bot."""
        if self._state is not RiskState.HALTED:
            self._state = RiskState.REDUCE_ONLY
            log.warning("modo_reduce_only", reason=reason)

    def reset(self) -> None:
        """Reactiva la operativa (uso manual tras revisión). Con cautela."""
        self._state = RiskState.ACTIVE
        self._peak_equity = self._current_equity
        self._day_start_equity = self._current_equity
        log.info("risk_manager_reset", equity=self._current_equity)

    # ------------------------------------------------------------------ #
    #  Utilidades internas
    # ------------------------------------------------------------------ #
    def _reset_daily(self, equity: float, day: int) -> None:
        self._session_day = day
        self._day_start_equity = equity
        self._peak_equity = equity
        if self._state is RiskState.REDUCE_ONLY:
            self._state = RiskState.ACTIVE
        log.info("nuevo_dia_trading", equity=equity)

    @staticmethod
    def _utc_day() -> int:
        now = datetime.now(timezone.utc)
        return now.toordinal()
