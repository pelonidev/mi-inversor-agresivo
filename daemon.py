"""daemon.py — El Vigilante 24/7 (Pairs Trading · Z-Score · Delta-Neutral).

Corre en bucle infinito sobre Binance USDT-M Futures (Perpetuo vs Perpetuo):

    1. Descarga velas 15m de las últimas 72h para un universo de ~20 altcoins.
    2. Calcula el Z-Score del ratio de precios de todas las combinaciones de
       pares (pandas/numpy, vectorizado).
    3. GESTIÓN DE SALIDAS: revisa primero las posiciones abiertas; si el ratio
       revirtió a la media (|Z| <= 0.5), cierra ambas patas y consolida el PnL
       en el PerformanceTracker.
    4. GESTIÓN DE ENTRADAS: filtra anomalías |Z| >= 2.5, las ordena por magnitud
       y evita el shock idiosincrático con un set() `active_tickers` (una moneda
       nunca se expone en dos pares a la vez).

Semántica de las patas (ambas en Futuros, delta-neutral):
    • Z > +2.5  ->  SHORT A (numerador) + LONG B (denominador)
    • Z < −2.5  ->  LONG  A (numerador) + SHORT B (denominador)

Compatible con el listener de Telegram (/status, /pnl, /trades) y el
PerformanceTracker (paper trading persistente).

Uso:
    python daemon.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import aiohttp
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from execution import DualPerpExecutionEngine
from performance_tracker import PAPER_FEE_PER_LEG, PerformanceTracker
from src.core.logger import configure_logging, get_logger
from src.risk.risk_manager import RiskLimits, RiskManager

load_dotenv()
log = get_logger("daemon")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
PORTFOLIO_STATE_PATH = os.getenv("PORTFOLIO_STATE_PATH", "data/portfolio_state.json")
PAIRS_STATE_PATH = os.getenv("PAIRS_STATE_PATH", "data/pairs_positions.json")

# --- Universo de altcoins de alta liquidez (Binance USDT-M Perpetuos) --------
UNIVERSE: list[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
    "POL/USDT", "LTC/USDT", "TRX/USDT", "ATOM/USDT", "NEAR/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "FIL/USDT", "INJ/USDT",
]

# --- Parámetros de la estrategia ---------------------------------------------
TIMEFRAME = "15m"
LOOKBACK_HOURS = 72
CANDLE_LIMIT = LOOKBACK_HOURS * (60 // 15)     # 72h * 4 = 288 velas
ENTRY_Z = 2.5                                   # tensión del hilo elástico
ENTRY_MAX_Z = 3.8                               # techo de entrada: buffer antes del stop-loss
EXIT_Z = 0.5                                    # take-profit: reversión a la media
STOP_LOSS_Z = 4.0                               # stop-loss estadístico (divergencia)
MAX_HOLD_HOURS = 48                             # stop-loss temporal
COOLDOWN_SECONDS = 4 * 3600                     # veto de reentrada tras cerrar un par
MAX_OPEN_PAIRS = 8                              # límite de posiciones simultáneas
RISK_PER_PAIR_PCT = 0.0666                      # ≈100 USDT por par sobre 1500 de banco
SCAN_INTERVAL_SECONDS = 30                      # cadencia del bucle (control fino de la vela 15m)
RATE_LIMIT_SLEEP = 0.10                         # pausa entre descargas (anti-baneo Binance)
LEG_NOTIONAL_USD = float(os.getenv("PAIR_NOTIONAL_USD", "50"))  # fallback/riesgo
FEES_PER_TRADE = PAPER_FEE_PER_LEG * 4          # 2 patas * (entrada + salida)

HEARTBEAT_INTERVAL_S = 24 * 60 * 60.0           # latido diario


# --------------------------------------------------------------------------- #
#  Modelo de posición
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PairPosition:
    """Posición de pairs trading delta-neutral abierta."""

    sym_a: str            # numerador (ej. SOL/USDT)
    sym_b: str            # denominador (ej. AVAX/USDT)
    direction: str        # "SHORT_A_LONG_B" | "LONG_A_SHORT_B"
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    qty_a: float
    qty_b: float
    leg_notional: float
    entry_ts: float
    max_z: float = 0.0    # |Z| máximo alcanzado durante la vida del trade
    current_z: float = 0.0  # Z-Score actual del par contra el mercado

    def __post_init__(self) -> None:
        # Compat JSON antiguo: si no traía max_z, lo sembramos con |entry_z|.
        if self.max_z <= 0.0:
            self.max_z = abs(self.entry_z)

    @property
    def key(self) -> str:
        return f"{self.sym_a}|{self.sym_b}"

    @property
    def bases(self) -> tuple[str, str]:
        return self.sym_a.split("/")[0], self.sym_b.split("/")[0]

    def gross_pnl(self, price_a: float, price_b: float) -> float:
        """PnL bruto (USD) al precio de salida; las dos patas se suman."""
        if self.direction == "SHORT_A_LONG_B":
            pnl_a = self.qty_a * (self.entry_price_a - price_a)   # short A
            pnl_b = self.qty_b * (price_b - self.entry_price_b)   # long B
        else:  # LONG_A_SHORT_B
            pnl_a = self.qty_a * (price_a - self.entry_price_a)   # long A
            pnl_b = self.qty_b * (self.entry_price_b - price_b)   # short B
        return pnl_a + pnl_b


# --------------------------------------------------------------------------- #
#  Telegram (API REST vía aiohttp)
# --------------------------------------------------------------------------- #
async def send_telegram(session: aiohttp.ClientSession, text: str) -> None:
    """Envía un mensaje al chat configurado. No-op si faltan credenciales."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.info("telegram_no_configurado", preview=text[:80])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "HTML"}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                log.warning("telegram_error", status=resp.status, body=await resp.text())
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        log.warning("telegram_fallo_envio", error=str(exc))


# --------------------------------------------------------------------------- #
#  Matemática vectorizada
# --------------------------------------------------------------------------- #
def compute_zscore(price_a: pd.Series, price_b: pd.Series) -> float:
    """Z-Score de la última vela del ratio (A/B) sobre la ventana completa."""
    ratio = price_a / price_b
    std = ratio.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((ratio.iloc[-1] - ratio.mean()) / std)


# --------------------------------------------------------------------------- #
#  Daemon
# --------------------------------------------------------------------------- #
class Daemon:
    def __init__(self, exchange_id: str) -> None:
        self._exchange_id = exchange_id
        self._exchange: ccxt_async.Exchange | None = None

        # Ejecución real (solo si AUTO_TRADE): exchange autenticado + motor + riesgo.
        self._auto_trade = AUTO_TRADE
        self._trade_exchange: ccxt_async.Exchange | None = None
        self._engine: DualPerpExecutionEngine | None = None
        self._risk = RiskManager(RiskLimits(account_equity_usd=LEG_NOTIONAL_USD * 4))

        self._tracker = PerformanceTracker(PORTFOLIO_STATE_PATH)
        self._positions: dict[str, PairPosition] = {}
        self._cooldowns: dict[str, float] = {}   # pair_key -> timestamp de fin de veto
        self._load_positions()
        self._reconcile_position_sizing()

        self._scan_count = 0
        self._last_heartbeat = time.time()
        self._session_start = time.time()   # inicio de esta sesión (uptime)

    # ------------------------------------------------------------------ #
    #  Persistencia de posiciones abiertas
    # ------------------------------------------------------------------ #
    def _load_positions(self) -> None:
        path = Path(PAIRS_STATE_PATH)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in raw:
            pos = PairPosition(**item)
            self._positions[pos.key] = pos

    def _save_positions(self) -> None:
        path = Path(PAIRS_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(p) for p in self._positions.values()]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _reconcile_position_sizing(self) -> None:
        """Parche único: reescala posiciones abiertas dimensionadas con el banco
        antiguo (100 USDT) al sizing actual, manteniendo entrada y Z."""
        target_leg = self._tracker.current_capital * RISK_PER_PAIR_PCT / 2.0
        if target_leg <= 0:
            return
        changed = False
        for pos in self._positions.values():
            # Muy por debajo del objetivo => venía del banco viejo; recalibrar.
            if pos.leg_notional < target_leg * 0.5:
                old = pos.leg_notional
                pos.leg_notional = target_leg
                if pos.entry_price_a > 0:
                    pos.qty_a = target_leg / pos.entry_price_a
                if pos.entry_price_b > 0:
                    pos.qty_b = target_leg / pos.entry_price_b
                changed = True
                log.warning("posicion_recalibrada", pair=pos.key,
                            antiguo=round(old, 2), nuevo=round(target_leg, 2))
        if changed:
            self._save_positions()

    # ------------------------------------------------------------------ #
    #  Ciclo de vida
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        self._exchange = getattr(ccxt_async, self._exchange_id)(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        if self._auto_trade:
            self._trade_exchange = self._build_trade_exchange()
            self._engine = DualPerpExecutionEngine(self._trade_exchange, self._risk)
            log.warning("AUTO_TRADE_ACTIVO", notional_por_pata=LEG_NOTIONAL_USD)

        mode = "AUTO (órdenes reales)" if self._auto_trade else "PAPER (virtual)"
        async with aiohttp.ClientSession() as session:
            await send_telegram(
                session,
                "🛰️ Pairs Trading daemon iniciado. "
                f"Vigilando {len(UNIVERSE)} altcoins (Z-Score reversión a la media). "
                f"Modo: <b>{mode}</b>",
            )
            try:
                await asyncio.gather(
                    self._radar_loop(session),
                    self._telegram_listener(session),
                )
            finally:
                await self._exchange.close()
                if self._trade_exchange is not None:
                    await self._trade_exchange.close()

    def _build_trade_exchange(self) -> ccxt_async.Exchange:
        """Exchange de Futuros autenticado para lanzar órdenes reales."""
        api_key = os.getenv("EXCHANGE_API_KEY", "")
        secret = os.getenv("EXCHANGE_API_SECRET", "")
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        ex = getattr(ccxt_async, self._exchange_id)(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if testnet:
            ex.set_sandbox_mode(True)
        return ex

    async def _radar_loop(self, session: aiohttp.ClientSession) -> None:
        while True:
            try:
                await self._tick(session)
                await self._maybe_heartbeat(session)
            except Exception as exc:  # noqa: BLE001 - el daemon nunca debe morir
                log.error("tick_error", error=str(exc), exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

    async def _tick(self, session: aiohttp.ClientSession) -> None:
        prices = await self._download_universe()
        if prices.empty or prices.shape[1] < 2:
            log.warning("datos_insuficientes", activos=prices.shape[1])
            return
        self._scan_count += 1

        # 1) SALIDAS: gestionamos las posiciones abiertas antes de buscar entradas.
        await self._manage_exits(session, prices)
        # 2) ENTRADAS: escaneamos anomalías con el filtro de shock idiosincrático.
        await self._scan_for_entries(session, prices)

    # ------------------------------------------------------------------ #
    #  Descarga de datos (15m, 72h)
    # ------------------------------------------------------------------ #
    async def _download_universe(self) -> pd.DataFrame:
        assert self._exchange is not None
        series: list[pd.Series] = []
        for symbol in UNIVERSE:
            try:
                ohlcv = await self._exchange.fetch_ohlcv(
                    symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT
                )
            except ccxt_async.BaseError as exc:
                log.warning("ohlcv_fallo", symbol=symbol, error=str(exc))
                await asyncio.sleep(RATE_LIMIT_SLEEP)
                continue
            if not ohlcv:
                continue
            df = pd.DataFrame(
                ohlcv, columns=["ts", "open", "high", "low", "close", "vol"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            series.append(df.set_index("ts")["close"].rename(symbol))
            await asyncio.sleep(RATE_LIMIT_SLEEP)

        if not series:
            return pd.DataFrame()
        prices = pd.concat(series, axis=1).sort_index().dropna(how="any")
        return prices

    # ------------------------------------------------------------------ #
    #  Condición de salida (reversión a la media)
    # ------------------------------------------------------------------ #
    async def _manage_exits(self, session: aiohttp.ClientSession, prices: pd.DataFrame) -> None:
        for pos in list(self._positions.values()):
            if pos.sym_a not in prices.columns or pos.sym_b not in prices.columns:
                log.warning("exit_sin_datos", pair=pos.key)
                continue
            z = compute_zscore(prices[pos.sym_a], prices[pos.sym_b])
            if np.isnan(z):
                continue

            # Tracking del Z-Score máximo alcanzado (se persiste en el JSON).
            pos.max_z = max(pos.max_z, abs(z))
            pos.current_z = z
            self._save_positions()

            abs_z = abs(z)
            held_h = (time.time() - pos.entry_ts) / 3600.0
            reason: str | None = None
            if abs_z <= EXIT_Z:
                reason = "Reversión"
            elif abs_z >= STOP_LOSS_Z:
                reason = "Stop Estadístico"
            elif held_h >= MAX_HOLD_HOURS:
                reason = "Stop Temporal"

            log.info("vigilando_par", pair=pos.key, z=round(z, 2),
                     max_z=round(pos.max_z, 2), held_h=round(held_h, 1))
            if reason is not None:
                price_a = float(prices[pos.sym_a].iloc[-1])
                price_b = float(prices[pos.sym_b].iloc[-1])
                await self._close_position(session, pos, price_a, price_b, z, reason)

    async def _close_position(
        self,
        session: aiohttp.ClientSession,
        pos: PairPosition,
        price_a: float,
        price_b: float,
        exit_z: float,
        reason: str,
    ) -> None:
        gross = pos.gross_pnl(price_a, price_b)
        fees = pos.leg_notional * FEES_PER_TRADE

        # Órdenes reales de cierre (lado opuesto de cada pata) si AUTO_TRADE.
        if self._auto_trade and self._engine is not None:
            await self._engine.close_pair(
                pos.sym_a, pos.sym_b, pos.direction, pos.qty_a, pos.qty_b
            )

        record = self._tracker.record_pairs_trade(
            pair=f"{pos.bases[0]}/{pos.bases[1]}",
            direction=pos.direction,
            entry_z=pos.entry_z,
            exit_z=exit_z,
            gross_pnl=gross,
            fees=fees,
            entry_ts=pos.entry_ts,
            notional=pos.leg_notional * 2,
            max_z=pos.max_z,
            reason=reason,
        )
        del self._positions[pos.key]
        # Veto de reentrada: bloquea reabrir este par durante COOLDOWN_SECONDS.
        self._cooldowns[pos.key] = time.time() + COOLDOWN_SECONDS
        self._save_positions()

        base_a, base_b = pos.bases
        stats = self._tracker.stats()
        await send_telegram(
            session,
            f"🔴 <b>CIERRE</b> {base_a}/{base_b}\n"
            f"Motivo: <b>{reason}</b>\n"
            f"Z entrada {pos.entry_z:+.2f} | salida {exit_z:+.2f}\n"
            f"Z-Score Máximo alcanzado: {pos.max_z:.2f}\n"
            f"PnL: {record['net_pnl']:+.4f} USDT "
            f"(bruto {record['gross_pnl']:+.4f} − fees {record['fees']:.4f})\n"
            f"Capital: {stats['current_capital']:.2f} USDT ({stats['total_return_pct']:+.2f}%)",
        )
        log.warning("PAR_CERRADO", pair=pos.key, pnl=record["net_pnl"],
                    exit_z=round(exit_z, 2), reason=reason, max_z=round(pos.max_z, 2))

    # ------------------------------------------------------------------ #
    #  Escaneo de entradas + filtro de shock idiosincrático
    # ------------------------------------------------------------------ #
    async def _scan_for_entries(self, session: aiohttp.ClientSession, prices: pd.DataFrame) -> None:
        symbols = list(prices.columns)
        anomalies: list[tuple[str, str, float]] = []
        for sym_a, sym_b in combinations(symbols, 2):
            z = compute_zscore(prices[sym_a], prices[sym_b])
            # Techo de entrada: solo dentro del buffer [ENTRY_Z, ENTRY_MAX_Z);
            # si el par ya está reventado (cerca del stop) no entramos.
            if not np.isnan(z) and ENTRY_Z <= abs(z) < ENTRY_MAX_Z:
                anomalies.append((sym_a, sym_b, z))

        anomalies.sort(key=lambda t: abs(t[2]), reverse=True)
        log.info("scan_completo", anomalias=len(anomalies), ts=time.strftime("%H:%M:%S"))
        if not anomalies:
            return

        # active_tickers arranca con las monedas ya expuestas en posiciones abiertas.
        active_tickers: set[str] = set()
        for pos in self._positions.values():
            active_tickers.update(pos.bases)

        for sym_a, sym_b, z in anomalies:
            # Límite de exposición: no superar el máximo de pares simultáneos.
            if len(self._positions) >= MAX_OPEN_PAIRS:
                log.info("max_pares_alcanzado", abiertos=len(self._positions))
                break
            pair = f"{sym_a}|{sym_b}"
            # Veto de reentrada: no reabrir un par recién cerrado (anti-bucle SL).
            if pair in self._cooldowns and time.time() < self._cooldowns[pair]:
                continue
            base_a = sym_a.split("/")[0]
            base_b = sym_b.split("/")[0]
            # Filtro de shock idiosincrático: una moneda, una sola exposición.
            if base_a in active_tickers or base_b in active_tickers:
                continue
            if pair in self._positions:
                continue
            await self._open_position(session, sym_a, sym_b, z, prices)
            active_tickers.add(base_a)
            active_tickers.add(base_b)

    async def _available_capital(self) -> float:
        """Capital para dimensionar: balance USDT real (AUTO) o capital paper."""
        if self._auto_trade and self._trade_exchange is not None:
            try:
                bal = await self._trade_exchange.fetch_balance()
                free = float((bal.get("USDT") or {}).get("free") or 0.0)
                if free > 0:
                    return free
            except ccxt_async.BaseError as exc:
                log.warning("fallo_balance_para_sizing", error=str(exc))
        return self._tracker.current_capital

    async def _open_position(
        self,
        session: aiohttp.ClientSession,
        sym_a: str,
        sym_b: str,
        z: float,
        prices: pd.DataFrame,
    ) -> None:
        price_a = float(prices[sym_a].iloc[-1])
        price_b = float(prices[sym_b].iloc[-1])
        if price_a <= 0 or price_b <= 0:
            return

        # Dynamic sizing: 6.6% del capital por par, repartido 50/50 entre patas.
        capital = await self._available_capital()
        pair_notional = capital * RISK_PER_PAIR_PCT
        leg_notional = pair_notional / 2.0
        if leg_notional <= 0:
            log.warning("sizing_invalido", capital=capital)
            return

        direction = "SHORT_A_LONG_B" if z > 0 else "LONG_A_SHORT_B"
        pos = PairPosition(
            sym_a=sym_a,
            sym_b=sym_b,
            direction=direction,
            entry_z=z,
            entry_price_a=price_a,
            entry_price_b=price_b,
            qty_a=leg_notional / price_a,
            qty_b=leg_notional / price_b,
            leg_notional=leg_notional,
            entry_ts=time.time(),
            max_z=abs(z),
        )

        base_a, base_b = pos.bases
        # Órdenes reales (50/50 long/short) si AUTO_TRADE; si fallan, no se abre.
        if self._auto_trade and self._engine is not None:
            fill = await self._engine.open_pair(
                sym_a, sym_b, direction, pair_notional
            )
            if fill is None:
                await send_telegram(
                    session,
                    f"⚠️ Orden real fallida en {base_a}/{base_b}: no se abre la posición.",
                )
                log.error("apertura_real_fallida", pair=pos.key)
                return
            # Sincroniza precios/cantidades reales de fill para el PnL posterior.
            pos.entry_price_a = fill.price_a
            pos.entry_price_b = fill.price_b
            pos.qty_a = fill.qty_a
            pos.qty_b = fill.qty_b

        self._positions[pos.key] = pos
        self._save_positions()

        if z > 0:
            action = f"SHORT {base_a}, LONG {base_b}"
        else:
            action = f"LONG {base_a}, SHORT {base_b}"
        await send_telegram(
            session,
            f"🟢 <b>ANOMALÍA DETECTADA</b> {base_a}/{base_b}\n"
            f"Z-Score: {z:+.2f} | Acción: {action}\n"
            f"Notional: {leg_notional:.2f} USDT por pata "
            f"({RISK_PER_PAIR_PCT * 100:.1f}% del capital, delta-neutral)",
        )
        log.warning("PAR_ABIERTO", pair=pos.key, z=round(z, 2),
                    direction=direction, leg_notional=round(leg_notional, 2))

    # ------------------------------------------------------------------ #
    #  Escucha de comandos de Telegram (/pnl, /status, /trades)
    # ------------------------------------------------------------------ #
    async def _telegram_listener(self, session: aiohttp.ClientSession) -> None:
        if not TELEGRAM_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        offset: int | None = None
        while True:
            try:
                params: dict[str, int] = {"timeout": 30}
                if offset is not None:
                    params["offset"] = offset
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=40)
                ) as resp:
                    data = await resp.json()
                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    text = ((upd.get("message") or {}).get("text") or "").strip().lower()
                    if text.startswith("/pnl") or text.startswith("/status"):
                        msg = (
                            f"{self._tracker.format_status()}\n"
                            f"⏱️ Uptime sesión: {self._format_uptime()}\n\n"
                            f"{self._format_open_positions()}"
                        )
                        await send_telegram(session, msg)
                        log.info("comando_telegram", cmd=text)
                    elif text.startswith("/trades"):
                        await send_telegram(session, self._tracker.format_trades(5))
                        log.info("comando_telegram", cmd=text)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(5)
            except Exception as exc:  # noqa: BLE001 - el listener nunca debe morir
                log.error("telegram_listener_error", error=str(exc))
                await asyncio.sleep(5)

    def _format_uptime(self) -> str:
        """Tiempo que lleva viva esta sesión del bot (Nd Nh Nm)."""
        elapsed = int(time.time() - self._session_start)
        days, rem = divmod(elapsed, 86_400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours or days:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)

    def _format_open_positions(self) -> str:
        if not self._positions:
            return "📂 <b>PARES ABIERTOS</b>\nNinguno."
        lines = [f"📂 <b>PARES ABIERTOS ({len(self._positions)})</b>"]
        for pos in self._positions.values():
            base_a, base_b = pos.bases
            side = "SHORT/LONG" if pos.direction == "SHORT_A_LONG_B" else "LONG/SHORT"
            lines.append(
                f"• {base_a}/{base_b} | {side} | "
                f"Z ent: {pos.entry_z:+.2f} | Z act: {pos.current_z:+.2f}"
            )
        return "\n".join(lines)

    async def _maybe_heartbeat(self, session: aiohttp.ClientSession) -> None:
        if time.time() - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        await send_telegram(
            session,
            f"🟢 [HEARTBEAT] Radar vivo. Últimas 24h: {self._scan_count} escaneos. "
            f"Pares abiertos: {len(self._positions)}. Seguimos vigilando.",
        )
        log.info("heartbeat_enviado", scans=self._scan_count, open_pairs=len(self._positions))
        self._scan_count = 0
        self._last_heartbeat = time.time()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairs Trading (Z-Score) vigilante daemon")
    p.add_argument("--exchange", type=str, default="binance")
    return p.parse_args()


def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    args = _parse_args()
    daemon = Daemon(args.exchange)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        log.info("daemon_detenido")


if __name__ == "__main__":
    main()
