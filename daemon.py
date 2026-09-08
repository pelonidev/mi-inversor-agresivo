"""daemon.py — El Vigilante 24/7 (radar + liquidity gate + alertas Telegram).

Corre en bucle infinito: escanea el funding de todo el universo, filtra las
señales por ejecutabilidad real (order book) y, cuando aparece una 🟢 EXECUTABLE,
envía una alerta a Telegram. Si AUTO_TRADE está activo, dispara la entrada
delta-neutral con el DualMarketExecutionEngine y arranca el MarginManager.

Config (variables de entorno / .env):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   -> alertas
    AUTO_TRADE=true|false                  -> ejecución automática
    EXCHANGE_API_KEY / _SECRET             -> solo si AUTO_TRADE

Uso:
    python daemon.py --interval 300 --top 100
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import aiohttp
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

from execution import (
    DualMarketExecutionEngine,
    ExecutableSignal,
    MarginManager,
)
from funding_radar import _interval_hours, _is_go, compute_metrics, run_liquidity_gate, run_once
from liquidity_gate import LiquidityStatus
from performance_tracker import PerformanceTracker
from src.core.logger import configure_logging, get_logger
from src.risk.risk_manager import RiskLimits, RiskManager

load_dotenv()
log = get_logger("daemon")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
AUTO_TRADE = os.getenv("AUTO_TRADE", "false").lower() == "true"
DEFAULT_NOTIONAL_USD = float(os.getenv("CARRY_NOTIONAL_USD", "1000"))
PORTFOLIO_STATE_PATH = os.getenv("PORTFOLIO_STATE_PATH", "data/portfolio_state.json")
ALERT_COOLDOWN_S = 3_600.0   # no repetir alerta del mismo ticker en 1h
EXIT_APR_THRESHOLD = 5.0     # UNWIND si el APR proyectado cae por debajo de esto
HEARTBEAT_INTERVAL_S = 24 * 60 * 60.0   # latido diario para confirmar que sigue vivo


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


def _format_alert(report) -> str:
    return (
        "🟢 <b>FUNDING EXECUTABLE</b>\n"
        f"<b>{report.symbol_perp}</b>\n"
        f"Break-even real: <b>{report.breakeven_days_real:.1f} días</b>\n"
        f"Coste real (fees+slippage): {report.real_cost * 100:.2f}%\n"
        f"Slippage spot/perp: {report.slippage_spot * 100:.3f}% / "
        f"{report.slippage_perp * 100:.3f}%\n"
        f"Vol spot/perp: ${report.spot_volume_usd / 1e6:.1f}M / "
        f"${report.perp_volume_usd / 1e6:.1f}M"
    )


# --------------------------------------------------------------------------- #
#  Ciclo de vigilancia
# --------------------------------------------------------------------------- #
class Daemon:
    def __init__(self, exchange_id: str, top_n: int, interval_s: float) -> None:
        self._exchange_id = exchange_id
        self._top_n = top_n
        self._interval_s = interval_s
        self._last_alert: dict[str, float] = {}   # ticker -> epoch del último aviso

        self._risk = RiskManager(RiskLimits(account_equity_usd=DEFAULT_NOTIONAL_USD * 2))
        self._engine: DualMarketExecutionEngine | None = None
        self._margin: MarginManager | None = None
        self._held: ExecutableSignal | None = None   # posición REAL activa (AUTO_TRADE)
        self._tracker = PerformanceTracker(PORTFOLIO_STATE_PATH)  # paper trading

        # Contadores del heartbeat (se resetean cada 24h).
        self._scan_count = 0
        self._traps_avoided = 0
        self._last_heartbeat = time.time()

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            await send_telegram(session, "🛰️ Funding daemon iniciado. Vigilando el régimen...")
            # Radar y escucha de comandos Telegram corren en paralelo (no se bloquean).
            await asyncio.gather(
                self._radar_loop(session),
                self._telegram_listener(session),
            )

    async def _radar_loop(self, session: aiohttp.ClientSession) -> None:
        while True:
            try:
                await self._tick(session)
                await self._maybe_heartbeat(session)
            except Exception as exc:  # noqa: BLE001 - el daemon nunca debe morir
                log.error("tick_error", error=str(exc), exc_info=True)
            await asyncio.sleep(self._interval_s)

    # ------------------------------------------------------------------ #
    #  Escucha de comandos de Telegram (/pnl, /status) — no bloqueante
    # ------------------------------------------------------------------ #
    async def _telegram_listener(self, session: aiohttp.ClientSession) -> None:
        """Long-poll de getUpdates: responde /pnl y /status de forma asíncrona."""
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
                        await send_telegram(session, self._tracker.format_status())
                        log.info("comando_telegram", cmd=text)
                    elif text.startswith("/trades"):
                        await send_telegram(session, self._tracker.format_trades(5))
                        log.info("comando_telegram", cmd=text)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(5)
            except Exception as exc:  # noqa: BLE001 - el listener nunca debe morir
                log.error("telegram_listener_error", error=str(exc))
                await asyncio.sleep(5)

    async def _maybe_heartbeat(self, session: aiohttp.ClientSession) -> None:
        """Cada 24h envía un latido a Telegram y resetea los contadores."""
        if time.time() - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        await send_telegram(
            session,
            f"🟢 [HEARTBEAT] Radar vivo. Últimas 24h: {self._scan_count} escaneos "
            f"realizados. {self._traps_avoided} trampas de liquidez evitadas. "
            f"Seguimos vigilando.",
        )
        log.info("heartbeat_enviado", scans=self._scan_count, traps=self._traps_avoided)
        self._scan_count = 0
        self._traps_avoided = 0
        self._last_heartbeat = time.time()

    async def _tick(self, session: aiohttp.ClientSession) -> None:
        # Máquina de estados del PAPER TRADING: si hay un trade virtual abierto,
        # lo gestionamos (funding + salida); si no, escaneamos buscando entrada.
        if self._tracker.has_open_trade():
            await self._manage_paper_position(session)
        else:
            await self._scan_for_entry(session)

    # ------------------------------------------------------------------ #
    #  Estado 1: búsqueda de entrada
    # ------------------------------------------------------------------ #
    async def _scan_for_entry(self, session: aiohttp.ClientSession) -> None:
        rows = await run_once(self._exchange_id, self._top_n)
        go_rows = [r for r in rows if _is_go(r)]
        self._scan_count += 1
        log.info("scan_completo", perps=len(rows), go=len(go_rows), ts=time.strftime("%H:%M:%S"))
        if not go_rows:
            return

        reports = await run_liquidity_gate(self._exchange_id, go_rows)
        executables = [
            (r, rep) for r, rep in reports if rep.status is LiquidityStatus.EXECUTABLE
        ]
        # Candidatos que superaron el radar pero el gate bloqueó = trampas evitadas.
        traps = len(reports) - len(executables)
        self._traps_avoided += traps
        self._tracker.increment_traps_avoided(traps)
        if not executables:
            log.info("go_sin_ejecutables", candidatos=len(go_rows))
            return

        for row, report in executables:
            if self._on_cooldown(report.symbol_perp):
                continue
            await send_telegram(session, _format_alert(report))
            self._last_alert[report.symbol_perp] = time.time()
            log.info("ALERTA_EXECUTABLE", symbol=report.symbol_perp,
                     breakeven=round(report.breakeven_days_real, 2))

            # PAPER TRADING: abre un trade virtual con precios reales del momento
            # (aunque AUTO_TRADE=false), solo si no hay ya una posición virtual.
            if not self._tracker.has_open_trade():
                await self._open_paper_trade(session, row, report)

            if AUTO_TRADE:
                await self._auto_trade(row, report)
            break   # una posición (virtual/real) a la vez; deja de escanear

    # ------------------------------------------------------------------ #
    #  Paper trading: apertura, acumulación de funding y salida
    # ------------------------------------------------------------------ #
    async def _open_paper_trade(self, session: aiohttp.ClientSession, row: dict, report) -> None:
        spot_price, perp_price = await self._fetch_prices(report.symbol_spot, report.symbol_perp)
        if spot_price <= 0 or perp_price <= 0:
            log.warning("paper_open_sin_precio", symbol=report.symbol_perp)
            return
        self._tracker.open_trade(
            ticker=report.symbol_perp,
            spot_symbol=report.symbol_spot,
            spot_price=spot_price,
            perp_price=perp_price,
            entry_apr=row["apr"],
        )
        log.info("PAPER_TRADE_ABIERTO", symbol=report.symbol_perp,
                 spot=spot_price, perp=perp_price)
        await send_telegram(
            session,
            f"🟡 <b>PAPER OPEN</b> {report.symbol_perp}\n"
            f"APR {row['apr']:.1f}% | notional virtual {self._tracker.current_capital:.2f} USDT",
        )

    async def _manage_paper_position(self, session: aiohttp.ClientSession) -> None:
        perp_symbol = self._tracker.open_ticker
        assert perp_symbol is not None
        apr, funding_now = await self._current_funding(perp_symbol)

        # Cobro de funding cada 8h (época real del exchange).
        if self._tracker.due_for_accrual():
            payment = self._tracker.accrue_funding(funding_now)
            log.info("paper_funding_cobrado", symbol=perp_symbol, pago=round(payment, 6))

        log.info("vigilando_paper", symbol=perp_symbol,
                 apr=round(apr, 2), funding=round(funding_now * 100, 5))

        # CONDICIÓN DE SALIDA (UNWIND): el carry dejó de pagar o se volvió negativo.
        if apr < EXIT_APR_THRESHOLD or funding_now < 0:
            await self._unwind_paper(session, apr, funding_now)

    async def _unwind_paper(self, session: aiohttp.ClientSession, apr: float, funding_now: float) -> None:
        perp_symbol = self._tracker.open_ticker or ""
        spot_symbol = self._tracker.open_spot_symbol or ""
        spot_exit, perp_exit = await self._fetch_prices(spot_symbol, perp_symbol)
        record = self._tracker.close_trade(spot_exit, perp_exit)

        # Si además había una posición REAL (AUTO_TRADE), la deshacemos también.
        if self._held is not None:
            await self._unwind(self._held)

        stats = self._tracker.stats()
        await send_telegram(
            session,
            f"🔴 <b>UNWIND</b> {perp_symbol}\n"
            f"APR cayó a {apr:.1f}% | funding {funding_now * 100:+.4f}%\n"
            f"PnL del trade: {record['net_pnl']:+.4f} USDT "
            f"(funding {record['funding_collected']:+.4f} − fees {record['fees']:.4f})\n"
            f"Capital: {stats['current_capital']:.2f} USDT ({stats['total_return_pct']:+.2f}%)",
        )
        log.warning("PAPER_UNWIND", symbol=perp_symbol, pnl=record["net_pnl"])

    async def _fetch_prices(self, spot_symbol: str, perp_symbol: str) -> tuple[float, float]:
        """Precio last real de spot y perp (para abrir/cerrar el trade virtual)."""
        spot_ex = getattr(ccxt_async, self._exchange_id)(
            {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        )
        perp_ex = getattr(ccxt_async, self._exchange_id)(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        try:
            spot_t, perp_t = await asyncio.gather(
                spot_ex.fetch_ticker(spot_symbol),
                perp_ex.fetch_ticker(perp_symbol),
            )
            spot = float(spot_t.get("last") or spot_t.get("close") or 0.0)
            perp = float(perp_t.get("last") or perp_t.get("close") or 0.0)
            return spot, perp
        except ccxt_async.BaseError as exc:
            log.error("fallo_fetch_prices", error=str(exc))
            return 0.0, 0.0
        finally:
            await spot_ex.close()
            await perp_ex.close()

    async def _unwind(self, held: ExecutableSignal) -> None:
        """Cierra la posición delta-neutral y apaga el salvavidas de margen."""
        if self._engine is not None:
            await self._engine.close_position()   # comprar perp + vender spot
        if self._margin is not None:
            await self._margin.stop()
        log.warning("UNWIND_EJECUTADO", symbol=held.perp_symbol)
        self._held = None
        self._margin = None

    async def _current_funding(self, perp_symbol: str) -> tuple[float, float]:
        """Devuelve (APR proyectado %, funding_rate actual) del símbolo en curso."""
        ex = getattr(ccxt_async, self._exchange_id)(
            {"enableRateLimit": True, "options": {"defaultType": "swap"}}
        )
        try:
            since = ex.milliseconds() - 3 * 24 * 60 * 60 * 1_000
            hist = await ex.fetch_funding_rate_history(perp_symbol, since=since, limit=100)
            rates = [float(h["fundingRate"]) for h in hist if h.get("fundingRate") is not None]
            rate_ma = sum(rates) / len(rates) if rates else 0.0
            current = await ex.fetch_funding_rate(perp_symbol)
            rate_now = float(current.get("fundingRate") or rate_ma)
            apr, _be = compute_metrics(rate_ma, _interval_hours(current))
            return apr, rate_now
        finally:
            await ex.close()

    def _on_cooldown(self, symbol: str) -> bool:
        last = self._last_alert.get(symbol, 0.0)
        return (time.time() - last) < ALERT_COOLDOWN_S

    async def _auto_trade(self, row: dict, report) -> None:
        """Entra delta-neutral y arranca el salvavidas de margen (modo Auto)."""
        signal = ExecutableSignal(
            perp_symbol=report.symbol_perp,
            spot_symbol=report.symbol_spot,
            notional_usd=DEFAULT_NOTIONAL_USD,
            funding_apr=row["apr"],
            breakeven_days=report.breakeven_days_real,
        )
        spot_ex, perp_ex = self._build_trade_exchanges()
        try:
            self._engine = DualMarketExecutionEngine(spot_ex, perp_ex, self._risk)
            self._margin = MarginManager(spot_ex, perp_ex, self._risk)
            pos = await self._engine.open_position(signal)
            if pos is not None:
                self._held = signal          # pasamos a estado IN_TRADE
                self._margin.start()         # salvavidas activo mientras haya posición
                log.warning("AUTO_TRADE_ABIERTO", symbol=signal.perp_symbol)
        except Exception as exc:  # noqa: BLE001
            log.error("auto_trade_error", error=str(exc), exc_info=True)

    def _build_trade_exchanges(self) -> tuple[ccxt_async.Exchange, ccxt_async.Exchange]:
        api_key = os.getenv("EXCHANGE_API_KEY", "")
        secret = os.getenv("EXCHANGE_API_SECRET", "")
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        cls = getattr(ccxt_async, self._exchange_id)
        spot = cls({"apiKey": api_key, "secret": secret, "enableRateLimit": True,
                    "options": {"defaultType": "spot"}})
        perp = cls({"apiKey": api_key, "secret": secret, "enableRateLimit": True,
                    "options": {"defaultType": "swap"}})
        if testnet:
            spot.set_sandbox_mode(True)
            perp.set_sandbox_mode(True)
        return spot, perp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Funding vigilante daemon")
    p.add_argument("--interval", type=int, default=300, help="Segundos entre escaneos")
    p.add_argument("--top", type=int, default=100, help="Top-N perps por volumen")
    p.add_argument("--exchange", type=str, default="binance")
    return p.parse_args()


def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    args = _parse_args()
    daemon = Daemon(args.exchange, args.top, float(args.interval))
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        log.info("daemon_detenido")


if __name__ == "__main__":
    main()
