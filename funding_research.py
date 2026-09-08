"""funding_research.py — Viabilidad de Funding Rate Arbitrage (Cash & Carry).

Estrategia delta-neutral: comprar 1 unidad en SPOT y vender (short) 1 unidad en
el PERPETUO del mismo activo (apalancamiento 1x). El PnL direccional se cancela
(delta ≈ 0) y se cobra el Funding Rate cada 8h mientras sea positivo.

FASE 1: descarga histórica de funding rates + precios (perp) de 6 meses.
FASE 2: backtest del yield neto (funding acumulado − fricción taker de 4 patas).
FASE 3: informe de rearquitectura (qué módulos reciclar / reescribir).

Uso:
    python funding_research.py
    python funding_research.py --months 6 --exchange binance
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Perpetuos lineales USDT (formato ccxt unificado: BASE/QUOTE:SETTLE).
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

TAKER_FEE = 0.001            # 0.1% por pata (entrada y salida, ambas patas)
FUNDING_INTERVAL_H = 8       # Binance/Bybit liquidan funding cada 8h
CANDLES_PER_CALL = 1_000

DATA_DIR = Path("data")

MAX_RETRIES = 6
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


# --------------------------------------------------------------------------- #
#  FASE 1: Data Fetcher estructural (funding + precio)
# --------------------------------------------------------------------------- #
async def _retry(coro_factory, what: str):
    """Ejecuta una corrutina con reintentos y backoff (rate limits/red)."""
    backoff = BASE_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except ccxt_async.RateLimitExceeded:
            print(f"  [429] {what}: espera {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    raise RuntimeError(f"Fallo definitivo: {what}")


async def fetch_funding_history(exchange: ccxt_async.Exchange, symbol: str, since: int, until: int) -> pd.DataFrame:
    """Descarga el historial de funding rates paginando por `since`."""
    rows: list[dict] = []
    cursor = since
    step_ms = FUNDING_INTERVAL_H * 3_600_000
    while cursor < until:
        batch = await _retry(
            lambda: exchange.fetch_funding_rate_history(symbol, since=cursor, limit=CANDLES_PER_CALL),
            f"funding {symbol}",
        )
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        nxt = last_ts + step_ms
        if nxt <= cursor:
            break
        cursor = nxt
        await asyncio.sleep(exchange.rateLimit / 1_000.0)

    df = pd.DataFrame(
        [{"timestamp": r["timestamp"], "funding_rate": float(r["fundingRate"])} for r in rows]
    )
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df[df.index <= pd.to_datetime(until, unit="ms", utc=True)]


async def fetch_price_8h(exchange: ccxt_async.Exchange, symbol: str, since: int, until: int) -> pd.Series:
    """Descarga closes 8h del perpetuo (para valorar el nocional del funding)."""
    rows: list[list[float]] = []
    cursor = since
    step_ms = FUNDING_INTERVAL_H * 3_600_000
    while cursor < until:
        batch = await _retry(
            lambda: exchange.fetch_ohlcv(symbol, timeframe="8h", since=cursor, limit=CANDLES_PER_CALL),
            f"ohlcv {symbol}",
        )
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step_ms
        if nxt <= cursor:
            break
        cursor = nxt
        await asyncio.sleep(exchange.rateLimit / 1_000.0)

    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "close", "v"]).drop_duplicates("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df["close"].rename(symbol)


async def download_all(exchange_id: str, months: int) -> dict[str, pd.DataFrame]:
    """Descarga funding + precio de todos los símbolos y los alinea por 8h."""
    exchange = getattr(ccxt_async, exchange_id)(
        {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    )
    now = exchange.milliseconds()
    since = now - months * 30 * 24 * 60 * 60 * 1_000

    data: dict[str, pd.DataFrame] = {}
    try:
        for sym in SYMBOLS:
            print(f"> {sym}")
            funding = await fetch_funding_history(exchange, sym, since, now)
            price = await fetch_price_8h(exchange, sym, since, now)
            merged = pd.concat([funding, price.rename("price")], axis=1, sort=True)
            merged = merged.dropna()
            data[sym] = merged
            span = (merged.index[-1] - merged.index[0]).days if not merged.empty else 0
            print(f"  {len(merged)} registros de funding (8h, ~{span}d)  "
                  f"| media={merged['funding_rate'].mean() * 100:.4f}% / 8h")
    finally:
        await exchange.close()
    return data


# --------------------------------------------------------------------------- #
#  FASE 2: Backtest del yield delta-neutral (1x)
# --------------------------------------------------------------------------- #
def backtest_funding(df: pd.DataFrame, symbol: str) -> dict[str, float]:
    """Simula 1 unidad spot long + 1 unidad perp short (delta-neutral, 1x).

    - PnL direccional ≈ 0 (spot y perp se cancelan).
    - Ingreso = funding_rate · nocional cobrado cada 8h (short cobra si rate>0).
    - Fricción = 4 patas taker (spot+perp en entrada y en salida).
    """
    if df.empty:
        return {}

    price = df["price"]
    rate = df["funding_rate"]

    entry_price = float(price.iloc[0])
    exit_price = float(price.iloc[-1])
    notional = entry_price  # 1 unidad

    # Funding cobrado por periodo ($): short recibe cuando el rate es positivo.
    funding_pnl = rate * price  # $ por cada 8h sobre 1 unidad

    # Fricción taker: 2 patas al entrar + 2 al salir (0.1% cada una).
    fee_entry = 2 * TAKER_FEE * entry_price
    fee_exit = 2 * TAKER_FEE * exit_price
    total_fees = fee_entry + fee_exit

    # Curva de capital: parte del nocional, menos fees de entrada, +funding.
    equity = notional - fee_entry + funding_pnl.cumsum()
    equity.iloc[-1] -= fee_exit  # fees de salida al cerrar

    gross_funding = float(funding_pnl.sum())
    net_pnl = gross_funding - total_fees
    total_return_pct = net_pnl / notional * 100

    days = (df.index[-1] - df.index[0]).total_seconds() / 86_400
    apr_gross = (gross_funding / notional) * (365 / days) * 100
    apr_net = (net_pnl / notional) * (365 / days) * 100

    running_max = equity.cummax()
    max_dd = float(((equity - running_max) / running_max).min() * 100)

    pos_periods = int((rate > 0).sum())
    return {
        "symbol": symbol,
        "days": days,
        "gross_funding": gross_funding,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "total_return_pct": total_return_pct,
        "apr_gross": apr_gross,
        "apr_net": apr_net,
        "max_dd": max_dd,
        "pct_positive": pos_periods / len(rate) * 100,
        "avg_rate_8h": float(rate.mean() * 100),
    }


def print_tear_sheet(results: list[dict[str, float]]) -> None:
    line = "=" * 64
    print(f"\n{line}")
    print("  TEAR SHEET — FUNDING ARBITRAGE (Delta-Neutral 1x, Taker in/out)")
    print(line)
    print(f"  {'Símbolo':<14}{'días':>6}{'APR neto':>10}{'APR bruto':>11}"
          f"{'Return':>9}{'MaxDD':>9}{'%+fund':>9}")
    for r in results:
        print(f"  {r['symbol']:<14}{r['days']:>6.0f}{r['apr_net']:>9.2f}%{r['apr_gross']:>10.2f}%"
              f"{r['total_return_pct']:>8.2f}%{r['max_dd']:>8.2f}%{r['pct_positive']:>8.1f}%")
    print(line)

    if results:
        avg_apr = float(np.mean([r["apr_net"] for r in results]))
        print(f"  APR NETO MEDIO (cartera equiponderada): {avg_apr:>8.2f}%")
        rf = 5.0
        verdict = "SUPERA la tasa libre de riesgo ✅" if avg_apr > rf else "NO supera el 5% ❌"
        print(f"  Umbral tasa libre de riesgo (~{rf:.0f}%): {verdict}")
    print(line + "\n")


# --------------------------------------------------------------------------- #
#  FASE 3: Informe de rearquitectura
# --------------------------------------------------------------------------- #
def print_rearchitecture_report() -> None:
    report = """
############################################################
  FASE 3 — INFORME DE REARQUITECTURA (Spot vs Derivados)
############################################################

RECICLAR DIRECTAMENTE (sin cambios o mínimos):
  • RiskManager: el Kill Switch, el control de exposición y el sizing
    por Kelly siguen siendo válidos. Se añade un monitor de RATIO DE
    MARGEN de la pata corta (perp) para anticipar liquidaciones.
  • DataFetcher (WebSocket): la infraestructura async de order books se
    reutiliza; se amplía para suscribir DOS mercados (spot + swap) y un
    nuevo canal de FUNDING RATE / mark price.
  • Logger, models (OrderBook, Position), configuración: intactos.

REESCRIBIR / EXTENDER:
  • StrategyEngine -> FundingStrategyEngine: ya NO calcula spread/z-score.
    Su señal es: entrar si funding_rate proyectado (neto de fees) > umbral;
    salir/rotar si el funding se vuelve negativo o cae bajo el hurdle.
  • ExecutionEngine -> DualMarketExecutionEngine: debe enrutar a DOS
    endpoints (spot y derivados) de forma atómica: comprar spot + vender
    perp simultáneamente (asyncio.gather), y garantizar neutralidad delta.
    Gestiona símbolos distintos (BTC/USDT vs BTC/USDT:USDT).

NUEVO MÓDULO CRÍTICO:
  • MarginManager (gestión de colateral): vigila el margin ratio del perp
    corto y transfiere USDT del balance spot al de futuros ANTES de que el
    mark price dispare una liquidación. Sin esto, un rally del subyacente
    liquida la pata corta y rompe la neutralidad (riesgo direccional).
  • BasisMonitor: vigila la divergencia mark-index (fuente del único
    drawdown real de esta estrategia) y la convergencia en el settlement.

############################################################
"""
    print(report)


# --------------------------------------------------------------------------- #
#  Orquestación
# --------------------------------------------------------------------------- #
async def main(months: int, exchange_id: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FASE 1: descargando funding + precio ({months} meses, {exchange_id})...\n")
    data = await download_all(exchange_id, months)

    # Cachea a parquet para inspección posterior.
    for sym, df in data.items():
        safe = sym.replace("/", "_").replace(":", "_")
        df.to_parquet(DATA_DIR / f"funding_{safe}.parquet")

    print("\nFASE 2: backtest del yield delta-neutral...")
    results = [backtest_funding(df, sym) for sym, df in data.items() if not df.empty]
    print_tear_sheet(results)

    print_rearchitecture_report()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Funding rate arbitrage research")
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--exchange", type=str, default="binance")
    return p.parse_args()


if __name__ == "__main__":
    _a = _parse_args()
    asyncio.run(main(months=_a.months, exchange_id=_a.exchange))
