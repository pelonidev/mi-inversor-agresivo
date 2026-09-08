"""funding_radar.py — Radar de régimen de funding (caza de prima estructural).

Escanea TODO el universo de perpetuos USDT-margined del exchange, calcula el
APR anualizado proyectado (usando la media de 3 días para filtrar picos falsos)
y el break-even time de la fricción, y muestra un dashboard `rich` ordenado por
APR. Marca 🟢 GO solo donde la ineficiencia es real y explotable.

Todas las llamadas de historial se lanzan concurrentemente (asyncio.gather con
semáforo) para escanear el exchange en segundos, no en minutos.

Uso:
    python funding_radar.py
    python funding_radar.py --top 100 --exchange binance
    python funding_radar.py --watch 30          # refresco cada 30s
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import ccxt.async_support as ccxt_async
from rich.console import Console
from rich.live import Live
from rich.table import Table

from liquidity_gate import (
    LiquidityReport,
    LiquidityStatus,
    _spot_symbol,
    evaluate_liquidity,
    finalize_status,
    real_breakeven_days,
)

console = Console()

# --- Gate de disparo ---
APR_GO_THRESHOLD = 15.0      # % anualizado mínimo para 🟢 GO
BREAKEVEN_GO_DAYS = 4.0      # días máximos de break-even para 🟢 GO

# --- Fricción ---
TAKER_FEE = 0.001            # 0.1% por pata
ROUND_TRIP_COST = 4 * TAKER_FEE   # spot+perp en entrada y salida = 0.4%

MA_DAYS = 3                  # ventana de media móvil del funding (anti-spike)
CONCURRENCY = 12             # llamadas de historial simultáneas
CALL_TIMEOUT_S = 15.0


# --------------------------------------------------------------------------- #
#  Universo: perpetuos USDT lineales, Top-N por volumen
# --------------------------------------------------------------------------- #
async def select_universe(exchange: ccxt_async.Exchange, top_n: int) -> list[str]:
    """Devuelve los Top-N símbolos de perp USDT por volumen en quote."""
    markets = await exchange.load_markets()
    perps = [
        m["symbol"]
        for m in markets.values()
        if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")
    ]

    tickers = await exchange.fetch_tickers(perps)
    ranked = sorted(
        perps,
        key=lambda s: (tickers.get(s, {}) or {}).get("quoteVolume") or 0.0,
        reverse=True,
    )
    return ranked[:top_n]


# --------------------------------------------------------------------------- #
#  Métricas por símbolo
# --------------------------------------------------------------------------- #
def _interval_hours(entry: dict) -> float:
    """Deduce el intervalo de funding en horas (8h por defecto)."""
    interval = entry.get("interval") or (entry.get("info", {}) or {}).get("fundingIntervalHours")
    if isinstance(interval, str) and interval.endswith("h"):
        try:
            return float(interval[:-1])
        except ValueError:
            return 8.0
    if interval:
        try:
            return float(interval)
        except (ValueError, TypeError):
            return 8.0
    return 8.0


async def funding_ma_3d(
    exchange: ccxt_async.Exchange,
    symbol: str,
    sem: asyncio.Semaphore,
    interval_h: float,
) -> float | None:
    """Media del funding de los últimos MA_DAYS días (anti falso positivo)."""
    since = exchange.milliseconds() - MA_DAYS * 24 * 60 * 60 * 1_000
    async with sem:
        try:
            hist = await asyncio.wait_for(
                exchange.fetch_funding_rate_history(symbol, since=since, limit=100),
                timeout=CALL_TIMEOUT_S,
            )
        except (ccxt_async.BaseError, asyncio.TimeoutError):
            return None
    if not hist:
        return None
    rates = [float(h["fundingRate"]) for h in hist if h.get("fundingRate") is not None]
    return sum(rates) / len(rates) if rates else None


def compute_metrics(rate_ma: float, interval_h: float) -> tuple[float, float]:
    """Devuelve (APR_anualizado_%, break_even_dias) a partir del funding medio."""
    epochs_per_year = 365 * 24 / interval_h
    apr = rate_ma * epochs_per_year * 100.0

    # Break-even: nº de epochs para recuperar la fricción round-trip.
    if rate_ma <= 0:
        breakeven_days = float("inf")
    else:
        epochs_needed = ROUND_TRIP_COST / rate_ma
        breakeven_days = epochs_needed * interval_h / 24.0
    return apr, breakeven_days


# --------------------------------------------------------------------------- #
#  Escaneo concurrente
# --------------------------------------------------------------------------- #
async def scan(exchange: ccxt_async.Exchange, symbols: list[str]) -> list[dict]:
    """Obtiene funding actual + MA 3d de todos los símbolos concurrentemente."""
    # Rate actual de TODOS de una sola llamada (si el exchange lo soporta).
    current: dict[str, dict] = {}
    if exchange.has.get("fetchFundingRates"):
        try:
            current = await exchange.fetch_funding_rates(symbols)
        except ccxt_async.BaseError:
            current = {}

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(symbol: str) -> dict | None:
        entry = current.get(symbol, {}) or {}
        interval_h = _interval_hours(entry)
        rate_now = entry.get("fundingRate")

        rate_ma = await funding_ma_3d(exchange, symbol, sem, interval_h)
        if rate_ma is None:
            rate_ma = float(rate_now) if rate_now is not None else None
        if rate_ma is None:
            return None

        apr, breakeven = compute_metrics(rate_ma, interval_h)
        return {
            "symbol": symbol,
            "rate_now": float(rate_now) if rate_now is not None else rate_ma,
            "rate_ma": rate_ma,
            "interval_h": interval_h,
            "apr": apr,
            "breakeven": breakeven,
        }

    rows = await asyncio.gather(*(_one(s) for s in symbols))
    return [r for r in rows if r is not None]


# --------------------------------------------------------------------------- #
#  Dashboard rich
# --------------------------------------------------------------------------- #
def build_table(rows: list[dict], exchange_id: str) -> Table:
    rows_sorted = sorted(rows, key=lambda r: r["apr"], reverse=True)
    n_go = sum(1 for r in rows_sorted if _is_go(r))

    table = Table(
        title=f"FUNDING RADAR — {exchange_id.upper()}  |  {len(rows_sorted)} perps  "
              f"|  🟢 GO: {n_go}  |  {time.strftime('%H:%M:%S')}",
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="white", no_wrap=True)
    table.add_column("Funding Actual", justify="right")
    table.add_column("Funding 3d MA", justify="right")
    table.add_column("APR Anualizado", justify="right")
    table.add_column("Break-Even", justify="right")
    table.add_column("Estado", justify="center")

    for r in rows_sorted:
        go = _is_go(r)
        apr_style = "bold green" if r["apr"] > APR_GO_THRESHOLD else (
            "yellow" if r["apr"] > 0 else "red")
        be = r["breakeven"]
        be_str = "∞" if be == float("inf") else f"{be:.1f}d"
        table.add_row(
            r["symbol"],
            f"{r['rate_now'] * 100:+.4f}%",
            f"{r['rate_ma'] * 100:+.4f}%",
            f"[{apr_style}]{r['apr']:+.1f}%[/{apr_style}]",
            be_str,
            "🟢 GO" if go else "🔴 WAIT",
        )
    return table


def _is_go(r: dict) -> bool:
    return r["apr"] > APR_GO_THRESHOLD and r["breakeven"] < BREAKEVEN_GO_DAYS


# --------------------------------------------------------------------------- #
#  Filtro de ejecutabilidad (liquidity gate) sobre las señales GO
# --------------------------------------------------------------------------- #
async def run_liquidity_gate(exchange_id: str, go_rows: list[dict]) -> list[tuple[dict, LiquidityReport]]:
    """Somete cada señal GO al escrutinio de estructura/volumen/slippage real."""
    spot_ex = getattr(ccxt_async, exchange_id)(
        {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    )
    perp_ex = getattr(ccxt_async, exchange_id)(
        {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    )
    out: list[tuple[dict, LiquidityReport]] = []
    try:
        spot_markets = await spot_ex.load_markets()
        await perp_ex.load_markets()

        existing_spot = [s for r in go_rows if (s := _spot_symbol(r["symbol"])) in spot_markets]
        tickers_spot = await spot_ex.fetch_tickers(existing_spot) if existing_spot else {}
        tickers_perp = await perp_ex.fetch_tickers([r["symbol"] for r in go_rows])

        for r in go_rows:
            report = await evaluate_liquidity(
                spot_ex, perp_ex, r["symbol"], spot_markets, tickers_spot, tickers_perp
            )
            be = real_breakeven_days(r["rate_ma"], r["interval_h"], report.real_cost)
            finalize_status(report, be)
            out.append((r, report))
    finally:
        await spot_ex.close()
        await perp_ex.close()
    return out


def build_liquidity_table(reports: list[tuple[dict, LiquidityReport]]) -> Table:
    table = Table(title="FILTRO DE EJECUTABILIDAD (fricción real del order book)",
                  header_style="bold magenta")
    table.add_column("Ticker", no_wrap=True)
    table.add_column("Vol Spot", justify="right")
    table.add_column("Vol Perp", justify="right")
    table.add_column("Slip Spot", justify="right")
    table.add_column("Slip Perp", justify="right")
    table.add_column("Coste Real", justify="right")
    table.add_column("Break-Even Real", justify="right")
    table.add_column("Estado", justify="center")

    for r, rep in reports:
        be = rep.breakeven_days_real
        be_str = "∞" if be == float("inf") else f"{be:.1f}d"
        exe = rep.status is LiquidityStatus.EXECUTABLE
        table.add_row(
            _spot_symbol(rep.symbol_perp),
            f"${rep.spot_volume_usd / 1e6:.1f}M" if rep.spot_volume_usd else "—",
            f"${rep.perp_volume_usd / 1e6:.1f}M" if rep.perp_volume_usd else "—",
            f"{rep.slippage_spot * 100:.3f}%" if rep.slippage_spot else "—",
            f"{rep.slippage_perp * 100:.3f}%" if rep.slippage_perp else "—",
            f"{rep.real_cost * 100:.2f}%" if rep.real_cost else "—",
            f"[{'bold green' if exe else 'red'}]{be_str}[/]",
            rep.status.value,
        )
    return table


# --------------------------------------------------------------------------- #
#  Orquestación
# --------------------------------------------------------------------------- #
async def run_once(exchange_id: str, top_n: int) -> list[dict]:
    exchange = getattr(ccxt_async, exchange_id)(
        {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    )
    try:
        symbols = await select_universe(exchange, top_n)
        console.print(f"[dim]Escaneando {len(symbols)} perps USDT (top volumen)...[/dim]")
        return await scan(exchange, symbols)
    finally:
        await exchange.close()


async def main(exchange_id: str, top_n: int, watch: int) -> None:
    if watch <= 0:
        rows = await run_once(exchange_id, top_n)
        console.print(build_table(rows, exchange_id))

        go_rows = [r for r in rows if _is_go(r)]
        if not go_rows:
            console.print("\n[bold red]Sin señales 🟢 GO.[/bold red] "
                          "Régimen comprimido: cash sigue siendo la posición.")
            return

        console.print(f"\n[dim]Sometiendo {len(go_rows)} señal(es) GO al filtro de "
                      f"ejecutabilidad (order book real)...[/dim]")
        reports = await run_liquidity_gate(exchange_id, go_rows)
        console.print(build_liquidity_table(reports))
        _print_executable_summary(reports)
        return

    # Modo watch: refresco periódico con rich.Live.
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            rows = await run_once(exchange_id, top_n)
            live.update(build_table(rows, exchange_id))
            await asyncio.sleep(watch)


def _print_executable_summary(reports: list[tuple[dict, LiquidityReport]]) -> None:
    execs = [rep for _, rep in reports if rep.status is LiquidityStatus.EXECUTABLE]
    if not execs:
        console.print("\n[bold red]Ninguna señal sobrevive a la fricción real.[/bold red] "
                      "El espejismo de liquidez confirmado: descartadas.")
        return
    console.print(f"\n[bold green]{len(execs)} señal(es) 🟢 EXECUTABLE:[/bold green]")
    for rep in execs:
        console.print(f"  • {rep.symbol_perp}: coste real {rep.real_cost * 100:.2f}%  "
                      f"break-even {rep.breakeven_days_real:.1f}d")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Funding regime radar")
    p.add_argument("--top", type=int, default=50, help="Top-N perps por volumen")
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--watch", type=int, default=0, help="Segundos de refresco (0 = snapshot)")
    return p.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _a = _parse_args()
    asyncio.run(main(exchange_id=_a.exchange, top_n=_a.top, watch=_a.watch))
