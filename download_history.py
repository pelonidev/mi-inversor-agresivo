"""download_history.py — Descargador de velas OHLCV históricas de 1 minuto.

Descarga los últimos N días de velas de 1m para dos símbolos vía
`ccxt.async_support`, con paginación robusta (parámetro `since`), manejo de
Rate Limits (HTTP 429) con backoff, alineación de timestamps de ambos activos
y forward-fill de huecos. Guarda el resultado en `.parquet`.

Uso:
    python download_history.py
    python download_history.py --days 30 --exchange binance

Salida:
    data/pair_prices.parquet  (columnas: "BTC/USDT", "ETH/USDT"; index datetime)
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1m"
TIMEFRAME_MS = 60_000               # 1 minuto en milisegundos
CANDLES_PER_CALL = 1_000            # límite típico por petición
OUTPUT_PARQUET = Path("data") / "pair_prices.parquet"

# Backoff ante Rate Limits / errores de red.
MAX_RETRIES = 6
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


async def _fetch_page(
    exchange: ccxt_async.Exchange,
    symbol: str,
    since: int,
) -> list[list[float]]:
    """Descarga una página de velas con reintentos y backoff exponencial."""
    backoff = BASE_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=since,
                limit=CANDLES_PER_CALL,
            )
        except ccxt_async.RateLimitExceeded:
            print(f"  [429] Rate limit en {symbol}; espera {backoff:.1f}s "
                  f"(intento {attempt}/{MAX_RETRIES})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable) as exc:
            print(f"  [red] {type(exc).__name__} en {symbol}; reintento en {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
    raise RuntimeError(f"No se pudo descargar {symbol} desde since={since}")


async def download_symbol(
    exchange: ccxt_async.Exchange,
    symbol: str,
    since: int,
    until: int,
) -> pd.DataFrame:
    """Descarga todas las velas de `symbol` en [since, until] paginando."""
    all_rows: list[list[float]] = []
    cursor = since

    while cursor < until:
        batch = await _fetch_page(exchange, symbol, cursor)
        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # Avanza el cursor una vela más allá de la última recibida.
        next_cursor = last_ts + TIMEFRAME_MS
        if next_cursor <= cursor:  # sin progreso: evita bucle infinito
            break
        cursor = next_cursor

        print(f"  {symbol}: {len(all_rows)} velas "
              f"(hasta {pd.to_datetime(last_ts, unit='ms')})")

        # Respeta el rate limit anunciado por el exchange.
        await asyncio.sleep(exchange.rateLimit / 1_000.0)

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df[df.index <= pd.to_datetime(until, unit="ms", utc=True)]


def align_and_fill(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Alinea los closes de ambos activos en un índice común y rellena huecos.

    Se reindexa a una malla regular de 1 minuto y se aplica forward-fill para
    los minutos sin vela (baja liquidez / gaps del exchange).
    """
    closes = {sym: df["close"].rename(sym) for sym, df in frames.items()}
    merged = pd.concat(closes.values(), axis=1)

    full_index = pd.date_range(
        start=merged.index.min(),
        end=merged.index.max(),
        freq="min",
        tz="UTC",
    )
    merged = merged.reindex(full_index)
    merged = merged.ffill().dropna()  # ffill huecos; dropna para el arranque
    merged.index.name = "timestamp"
    return merged


def _build_exchange(exchange_id: str) -> ccxt_async.Exchange:
    exchange_class = getattr(ccxt_async, exchange_id)
    return exchange_class({"enableRateLimit": True, "options": {"defaultType": "spot"}})


async def main(days: int, exchange_id: str) -> None:
    exchange = _build_exchange(exchange_id)
    now = exchange.milliseconds()
    since = now - days * 24 * 60 * 60 * 1_000

    print(f"Descargando {days} días de velas {TIMEFRAME} desde {exchange_id}...")
    try:
        # Secuencial por símbolo para no saturar el rate limit compartido.
        frames: dict[str, pd.DataFrame] = {}
        for symbol in SYMBOLS:
            print(f"\n> {symbol}")
            frames[symbol] = await download_symbol(exchange, symbol, since, now)
    finally:
        await exchange.close()

    merged = align_and_fill(frames)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET)
    print(f"\nGuardado: {OUTPUT_PARQUET.resolve()}")
    print(f"Filas alineadas: {len(merged)}  |  Rango: {merged.index[0]} -> {merged.index[-1]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descargador OHLCV 1m -> parquet")
    parser.add_argument("--days", type=int, default=30, help="Días de histórico")
    parser.add_argument("--exchange", type=str, default="binance", help="ID de ccxt")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(days=args.days, exchange_id=args.exchange))
