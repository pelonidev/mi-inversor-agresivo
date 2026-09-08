"""performance_tracker.py — Seguimiento de rendimiento en PAPER TRADING.

Persiste el estado del portfolio virtual en un JSON (montable en un volumen de
Docker) y simula la operativa delta-neutral de funding arbitrage:

    • Al recibir una señal 🟢 EXECUTABLE (aunque AUTO_TRADE=false) abre un trade
      virtual con los precios reales del momento.
    • Cada 8h (época de funding) acumula el funding real cobrado.
    • Al cumplirse el UNWIND (APR<5% o funding negativo) cierra, descuenta las
      comisiones teóricas y consolida el beneficio.

Como la posición es delta-neutral, el PnL de precio (spot vs perp) se cancela:
el beneficio del trade = funding acumulado − comisiones.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Parámetros del paper trading.
PAPER_INITIAL_CAPITAL = 100.0     # USDT virtuales
PAPER_FEE_PER_LEG = 0.001         # 0.1% taker por pata (market orders)
PAPER_N_LEGS = 4                  # spot+perp en entrada y en salida
FUNDING_INTERVAL_S = 8 * 60 * 60  # el funding se liquida cada 8h


@dataclass(slots=True)
class PaperTrade:
    """Trade virtual delta-neutral en curso."""

    ticker: str
    spot_symbol: str
    entry_ts: float
    spot_entry: float
    perp_entry: float
    notional: float
    entry_apr: float
    funding_accrued: float = 0.0
    last_accrual_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.last_accrual_ts == 0.0:
            self.last_accrual_ts = self.entry_ts


class PerformanceTracker:
    """Estado persistente del portfolio virtual + lógica de paper trading."""

    def __init__(self, path: str | Path, initial_capital: float = PAPER_INITIAL_CAPITAL) -> None:
        self._path = Path(path)
        self._initial_capital = initial_capital
        self._realized_pnl = 0.0
        self._trades: list[dict[str, Any]] = []     # trades cerrados
        self._open: PaperTrade | None = None
        self._total_traps_avoided = 0               # escudo del liquidity gate
        self._start_date = datetime.now(timezone.utc).isoformat()
        self.load()

    # ------------------------------------------------------------------ #
    #  Persistencia
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not self._path.exists():
            self.save()
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._initial_capital = data.get("initial_capital", self._initial_capital)
        self._realized_pnl = data.get("realized_pnl", 0.0)
        self._trades = data.get("trades", [])
        self._total_traps_avoided = data.get("total_traps_avoided", 0)
        self._start_date = data.get("start_date", self._start_date)
        raw_open = data.get("open_trade")
        self._open = PaperTrade(**raw_open) if raw_open else None

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "initial_capital": self._initial_capital,
            "realized_pnl": self._realized_pnl,
            "start_date": self._start_date,
            "total_traps_avoided": self._total_traps_avoided,
            "trades": self._trades,
            "open_trade": asdict(self._open) if self._open else None,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)   # escritura atómica

    # ------------------------------------------------------------------ #
    #  Ciclo de vida del trade virtual
    # ------------------------------------------------------------------ #
    def has_open_trade(self) -> bool:
        return self._open is not None

    def increment_traps_avoided(self, count: int) -> None:
        """Suma trampas de liquidez rechazadas por el gate y persiste el total."""
        if count <= 0:
            return
        self._total_traps_avoided += count
        self.save()

    @property
    def open_ticker(self) -> str | None:
        return self._open.ticker if self._open else None

    @property
    def open_spot_symbol(self) -> str | None:
        return self._open.spot_symbol if self._open else None

    def open_trade(
        self,
        ticker: str,
        spot_symbol: str,
        spot_price: float,
        perp_price: float,
        entry_apr: float,
    ) -> None:
        if self._open is not None:
            return
        self._open = PaperTrade(
            ticker=ticker,
            spot_symbol=spot_symbol,
            entry_ts=time.time(),
            spot_entry=spot_price,
            perp_entry=perp_price,
            notional=self.current_capital,   # compone sobre el capital actual
            entry_apr=entry_apr,
        )
        self.save()

    def due_for_accrual(self) -> bool:
        """True si han pasado >= 8h desde el último cobro de funding."""
        if self._open is None:
            return False
        return (time.time() - self._open.last_accrual_ts) >= FUNDING_INTERVAL_S

    def accrue_funding(self, funding_rate: float) -> float:
        """Suma el funding real cobrado de una época (short recibe si rate>0)."""
        if self._open is None:
            return 0.0
        payment = funding_rate * self._open.notional
        self._open.funding_accrued += payment
        self._open.last_accrual_ts = time.time()
        self.save()
        return payment

    def close_trade(self, spot_exit: float, perp_exit: float) -> dict[str, Any]:
        """Cierra el trade virtual: PnL = funding − comisiones (delta-neutral)."""
        assert self._open is not None
        t = self._open
        fees = t.notional * PAPER_FEE_PER_LEG * PAPER_N_LEGS
        net_pnl = t.funding_accrued - fees

        record = {
            "ticker": t.ticker,
            "entry_date": datetime.fromtimestamp(t.entry_ts, timezone.utc).isoformat(),
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "spot_entry": t.spot_entry,
            "perp_entry": t.perp_entry,
            "spot_exit": spot_exit,
            "perp_exit": perp_exit,
            "funding_collected": round(t.funding_accrued, 6),
            "fees": round(fees, 6),
            "net_pnl": round(net_pnl, 6),
            "notional": t.notional,
        }
        self._trades.append(record)
        self._realized_pnl += net_pnl
        self._open = None
        self.save()
        return record

    # ------------------------------------------------------------------ #
    #  Métricas
    # ------------------------------------------------------------------ #
    @property
    def current_capital(self) -> float:
        unrealized = self._open.funding_accrued if self._open else 0.0
        return self._initial_capital + self._realized_pnl + unrealized

    @property
    def active_days(self) -> float:
        start = datetime.fromisoformat(self._start_date)
        delta = datetime.now(timezone.utc) - start
        return max(delta.total_seconds() / 86_400, 1e-9)

    def stats(self) -> dict[str, Any]:
        capital = self.current_capital
        total_return_pct = (capital / self._initial_capital - 1.0) * 100
        closed = self._trades
        wins = sum(1 for t in closed if t["net_pnl"] > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0.0
        # Divisor con suelo de 1 día: evita "media diaria" absurda en las
        # primeras horas de ejecución (dividir por ~0 días).
        daily_avg_pct = total_return_pct / max(self.active_days, 1.0)
        return {
            "current_capital": capital,
            "total_return_pct": total_return_pct,
            "accumulated_pnl": self._realized_pnl + (self._open.funding_accrued if self._open else 0.0),
            "daily_avg_pct": daily_avg_pct,
            "n_trades": len(closed),
            "win_rate": win_rate,
            "active_position": self._open.ticker if self._open else None,
            "active_days": self.active_days,
            "total_traps_avoided": self._total_traps_avoided,
        }

    def format_status(self) -> str:
        """Mensaje limpio para el comando /pnl de Telegram."""
        s = self.stats()
        pos = s["active_position"] or "Ninguna"
        return (
            "📊 <b>ESTADO DEL RADAR (PAPER TRADING)</b>\n"
            f"• Capital Actual: {s['current_capital']:.2f} USDT "
            f"({s['total_return_pct']:+.2f}%)\n"
            f"• Beneficio Acumulado: {s['accumulated_pnl']:+.2f} USDT\n"
            f"• Media Diaria: {s['daily_avg_pct']:+.3f}% / día\n"
            f"• Trades Realizados: {s['n_trades']} (Win Rate: {s['win_rate']:.0f}%)\n"
            f"🛡️ Trampas de Liquidez Evitadas: {s['total_traps_avoided']}\n"
            f"• Posición Activa: {pos}"
        )

    def format_trades(self, limit: int = 5) -> str:
        """Lista los últimos `limit` trades cerrados para el comando /trades."""
        if not self._trades:
            return "📁 <b>ÚLTIMOS TRADES</b>\nAún no hay trades cerrados."
        recent = self._trades[-limit:][::-1]   # los más recientes primero
        lines = ["📁 <b>ÚLTIMOS TRADES CERRADOS</b>"]
        for t in recent:
            entry = datetime.fromisoformat(t["entry_date"])
            exit_ = datetime.fromisoformat(t["exit_date"])
            days = (exit_ - entry).total_seconds() / 86_400
            lines.append(
                f"• {t['ticker']} | {days:.1f}d | PnL {t['net_pnl']:+.4f} USDT"
            )
        return "\n".join(lines)
