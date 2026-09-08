"""screener.py — Screener de ineficiencias (matriz de cointegración).

Descarga velas de 1h de los últimos 30 días para un universo de Layer-1s y
oráculos líquidos, evalúa TODAS las combinaciones de pares con el test de
cointegración de Engle-Granger, calcula el half-life de cada spread y filtra/
ranquea los pares operables:

    a) Cointegración estacionaria: p-value < 0.05
    b) Half-life explotable:       entre 4 y 24 velas (horas)

Toma el par #1 y guarda sus dos series (1h) en `data/screener_top_pair.parquet`
para pasarlo por `backtest.py`.

Uso:
    python screener.py
    python screener.py --days 30 --exchange binance
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from pathlib import Path

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Universo: Layer-1s y oráculos líquidos contra USDT.
UNIVERSE = [
    "ETH/USDT", "SOL/USDT", "LINK/USDT", "POL/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "NEAR/USDT",
]
TIMEFRAME = "1h"
TIMEFRAME_MS = 3_600_000
CANDLES_PER_CALL = 1_000

# Criterios de filtrado.
P_VALUE_MAX = 0.05
HALF_LIFE_MIN = 4.0      # velas (horas)
HALF_LIFE_MAX = 24.0     # velas (horas)

OUTPUT_PARQUET = Path("data") / "screener_top_pair.parquet"
UNIVERSE_PARQUET = Path("data") / "universe_1h.parquet"

MAX_RETRIES = 6
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


# --------------------------------------------------------------------------- #
#  Descarga paginada (reutiliza el patrón de download_history)
# --------------------------------------------------------------------------- #
async def _fetch_page(exchange: ccxt_async.Exchange, symbol: str, since: int) -> list[list[float]]:
    backoff = BASE_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await exchange.fetch_ohlcv(
                symbol, timeframe=TIMEFRAME, since=since, limit=CANDLES_PER_CALL
            )
        except ccxt_async.RateLimitExceeded:
            print(f"  [429] {symbol}: espera {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    raise RuntimeError(f"No se pudo descargar {symbol}")


async def _download_symbol(exchange: ccxt_async.Exchange, symbol: str, since: int, until: int) -> pd.Series:
    rows: list[list[float]] = []
    cursor = since
    while cursor < until:
        batch = await _fetch_page(exchange, symbol, cursor)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + TIMEFRAME_MS
        if nxt <= cursor:
            break
        cursor = nxt
        await asyncio.sleep(exchange.rateLimit / 1_000.0)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset="ts").set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df["close"].rename(symbol)


async def download_universe(exchange_id: str, days: int) -> pd.DataFrame:
    """Descarga los closes de 1h del universo y los alinea (forward-fill)."""
    exchange = getattr(ccxt_async, exchange_id)({"enableRateLimit": True})
    now = exchange.milliseconds()
    since = now - days * 24 * 60 * 60 * 1_000

    print(f"Descargando {len(UNIVERSE)} símbolos ({days}d, {TIMEFRAME})...")
    series: list[pd.Series] = []
    try:
        for sym in UNIVERSE:
            try:
                s = await _download_symbol(exchange, sym, since, now)
                if s.empty:
                    print(f"  {sym}: OMITIDO (sin datos / delistado)")
                    continue
                series.append(s)
                print(f"  {sym}: {len(s)} velas")
            except Exception as exc:  # noqa: BLE001 - símbolo puede no existir en el exchange
                print(f"  {sym}: OMITIDO ({exc})")
    finally:
        await exchange.close()

    # Descarta columnas totalmente vacías antes de alinear (evita borrar todo).
    merged = pd.concat(series, axis=1).dropna(axis=1, how="all")
    full = pd.date_range(merged.index.min(), merged.index.max(), freq="h", tz="UTC")
    return merged.reindex(full).ffill().dropna()


# --------------------------------------------------------------------------- #
#  Matemática: half-life y evaluación de un par
# --------------------------------------------------------------------------- #
def half_life(spread: np.ndarray) -> float:
    """Half-life de reversión (OU): −ln(2)/λ, en velas."""
    lag = spread[:-1]
    delta = spread[1:] - lag
    if lag.size < 2 or np.std(lag) == 0:
        return float("inf")
    lam, _ = np.polyfit(lag, delta, 1)
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def evaluate_pair(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Devuelve (p_value, half_life, hedge_ratio) del par A~B (Engle-Granger)."""
    # Hedge ratio por OLS cerrado: β = cov/var.
    beta = float(np.cov(a, b)[0, 1] / np.var(b))
    spread = (a - beta * b).to_numpy()
    try:
        _t, p_value, _c = coint(a, b, trend="c")
    except (ValueError, np.linalg.LinAlgError):
        p_value = 1.0
    return float(p_value), half_life(spread), beta


def screen(prices: pd.DataFrame) -> pd.DataFrame:
    """Evalúa todas las combinaciones y devuelve un ranking filtrado."""
    records: list[dict[str, object]] = []
    for sym_a, sym_b in itertools.combinations(prices.columns, 2):
        a, b = prices[sym_a], prices[sym_b]
        # Se evalúan ambas orientaciones; nos quedamos con el menor p-value.
        p_ab, hl_ab, beta_ab = evaluate_pair(a, b)
        p_ba, hl_ba, beta_ba = evaluate_pair(b, a)
        if p_ab <= p_ba:
            rec = dict(symbol_a=sym_a, symbol_b=sym_b, p_value=p_ab,
                       half_life=hl_ab, hedge_ratio=beta_ab)
        else:
            rec = dict(symbol_a=sym_b, symbol_b=sym_a, p_value=p_ba,
                       half_life=hl_ba, hedge_ratio=beta_ba)
        records.append(rec)

    df = pd.DataFrame(records)
    passed = df[
        (df["p_value"] < P_VALUE_MAX)
        & (df["half_life"] >= HALF_LIFE_MIN)
        & (df["half_life"] <= HALF_LIFE_MAX)
    ].copy()
    # Ranking: menor p-value primero (cointegración más significativa).
    return passed.sort_values("p_value").reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Orquestación
# --------------------------------------------------------------------------- #
async def main(days: int, exchange_id: str) -> None:
    prices = await download_universe(exchange_id, days)
    print(f"\nMatriz alineada: {prices.shape[1]} activos × {len(prices)} velas\n")

    # Guarda el universo completo para backtestear cualquier par por columnas.
    UNIVERSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(UNIVERSE_PARQUET)

    ranking = screen(prices)

    if ranking.empty:
        print("Ningún par supera los filtros (p<0.05 y half-life 4-24h).")
        print("El universo es demasiado eficiente en esta ventana.")
        return

    print("=" * 64)
    print("  TOP PARES COINTEGRADOS (p<0.05, half-life 4-24h)")
    print("=" * 64)
    print(f"  {'#':>2}  {'PAR':<22} {'p-value':>9} {'half-life':>10} {'β':>10}")
    for i, row in ranking.head(10).iterrows():
        pair = f"{row['symbol_a']}/{row['symbol_b']}"
        print(f"  {i + 1:>2}  {pair:<22} {row['p_value']:>9.4f} "
              f"{row['half_life']:>8.1f}h {row['hedge_ratio']:>10.4f}")
    print("=" * 64)

    # Guardar el par #1 para el backtest.
    winner = ranking.iloc[0]
    sym_a, sym_b = str(winner["symbol_a"]), str(winner["symbol_b"])
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    prices[[sym_a, sym_b]].to_parquet(OUTPUT_PARQUET)

    print(f"\nPar ganador: {sym_a}/{sym_b}  (guardado en {OUTPUT_PARQUET})")
    print("Backtestéalo con:")
    print(f'  python backtest.py --data "{OUTPUT_PARQUET}" '
          f'--symbol-a "{sym_a}" --symbol-b "{sym_b}" --timeframes 1h')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screener de cointegración")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--exchange", type=str, default="binance")
    return parser.parse_args()


if __name__ == "__main__":
    _a = _parse_args()
    asyncio.run(main(days=_a.days, exchange_id=_a.exchange))
