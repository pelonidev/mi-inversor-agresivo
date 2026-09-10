"""
proof_of_concept.py
====================

Prueba de concepto aislada para validar empíricamente la estrategia de
**Pairs Trading (Reversión a la media) basada en Z-Score** sobre Binance
USDT-M Futures (Perpetuo vs Perpetuo, Delta-Neutral).

No modifica el daemon ni el motor de ejecución: solo lee datos públicos de
mercado, calcula el ratio de precios entre cada par de altcoins y reporta
las anomalías estadísticas (|Z-Score| > 2.5) que representarían una
oportunidad teórica de entrada.

Uso:
    python proof_of_concept.py
"""

from __future__ import annotations

import time
from itertools import combinations

import ccxt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuración                                                               #
# --------------------------------------------------------------------------- #

# Universo de activos: altcoins de alta capitalización y liquidez en Binance
# USDT-M Perpetuos. Se usan símbolos spot-style que ccxt mapea al perpetuo
# lineal cuando defaultType='future'.
UNIVERSE: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOGE/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "POL/USDT",
    "LTC/USDT",
    "TRX/USDT",
    "ATOM/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "FIL/USDT",
    "INJ/USDT",
]

TIMEFRAME: str = "15m"            # Intervalo de las velas
LOOKBACK_HOURS: int = 72          # Ventana histórica (y ventana rolling completa)
Z_SCORE_THRESHOLD: float = 2.5    # Umbral de tensión del "hilo elástico"
RATE_LIMIT_SLEEP: float = 0.25    # Pausa entre descargas para evitar baneos

# 72 horas de velas de 15m = 72 * 4 = 288 velas.
CANDLES_PER_HOUR: int = 60 // 15
LIMIT: int = LOOKBACK_HOURS * CANDLES_PER_HOUR


# --------------------------------------------------------------------------- #
# Descarga de datos                                                           #
# --------------------------------------------------------------------------- #

def build_exchange() -> ccxt.binance:
    """Crea el cliente de Binance USDT-M Futures (solo lectura pública)."""
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )
    exchange.load_markets()
    return exchange


def fetch_close_series(exchange: ccxt.binance, symbol: str) -> pd.Series | None:
    """
    Descarga las velas de cierre para `symbol` y devuelve una Series indexada
    por timestamp. Devuelve None si el símbolo no está disponible o falla.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    except Exception as exc:  # noqa: BLE001 - PoC: reportar y continuar
        print(f"   ⚠️  No se pudo descargar {symbol}: {exc}")
        return None

    if not ohlcv:
        print(f"   ⚠️  Sin datos para {symbol}.")
        return None

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")["close"].rename(symbol)


def download_universe(exchange: ccxt.binance) -> pd.DataFrame:
    """
    Descarga los precios de cierre de todo el universo y los alinea en un
    único DataFrame (una columna por activo), respetando el rate limit.
    """
    series_list: list[pd.Series] = []

    print(f"📥 Descargando {len(UNIVERSE)} activos "
          f"({TIMEFRAME}, últimas {LOOKBACK_HOURS}h, {LIMIT} velas)...")

    for i, symbol in enumerate(UNIVERSE, start=1):
        print(f"   [{i:>2}/{len(UNIVERSE)}] {symbol}")
        series = fetch_close_series(exchange, symbol)
        if series is not None:
            series_list.append(series)
        time.sleep(RATE_LIMIT_SLEEP)  # Cortesía con los rate limits de Binance

    if not series_list:
        raise RuntimeError("No se pudo descargar ningún activo del universo.")

    # Alinea por timestamp (join externo) y elimina filas incompletas para
    # garantizar que todos los ratios comparten exactamente la misma ventana.
    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.dropna(how="any")
    return prices


# --------------------------------------------------------------------------- #
# Matemática vectorizada                                                       #
# --------------------------------------------------------------------------- #

def compute_zscore(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Calcula el Z-Score de la última vela para el ratio (A / B) usando la
    media y desviación estándar de toda la ventana disponible.

    Devuelve np.nan si la desviación estándar es cero (ratio constante).
    """
    ratio = price_a / price_b
    mean = ratio.mean()
    std = ratio.std()

    if std == 0 or np.isnan(std):
        return float("nan")

    return float((ratio.iloc[-1] - mean) / std)


def scan_pairs(prices: pd.DataFrame) -> list[dict]:
    """
    Genera todas las combinaciones únicas de pares y calcula su Z-Score
    actual. Devuelve la lista de anomalías (|Z-Score| >= umbral) ordenada
    por magnitud descendente.
    """
    anomalies: list[dict] = []
    symbols = list(prices.columns)
    total_pairs = len(list(combinations(symbols, 2)))

    print(f"\n🔬 Analizando {total_pairs} pares únicos "
          f"(umbral |Z| >= {Z_SCORE_THRESHOLD})...\n")

    for sym_a, sym_b in combinations(symbols, 2):
        z = compute_zscore(prices[sym_a], prices[sym_b])
        if np.isnan(z):
            continue
        if abs(z) >= Z_SCORE_THRESHOLD:
            anomalies.append({"a": sym_a, "b": sym_b, "z": z})

    anomalies.sort(key=lambda item: abs(item["z"]), reverse=True)
    return anomalies


# --------------------------------------------------------------------------- #
# Salida                                                                       #
# --------------------------------------------------------------------------- #

def report(anomalies: list[dict]) -> None:
    """Imprime las anomalías detectadas con la acción teórica de trading."""
    if not anomalies:
        print("✅ Sin anomalías: ningún par supera el umbral de Z-Score.")
        return

    print(f"🚨 {len(anomalies)} anomalía(s) detectada(s):\n")
    for item in anomalies:
        sym_a, sym_b, z = item["a"], item["b"], item["z"]
        # Nombres cortos (sin '/USDT') para una acción legible.
        name_a = sym_a.split("/")[0]
        name_b = sym_b.split("/")[0]

        # Z alto positivo -> A caro respecto a B -> vender A, comprar B.
        # Z alto negativo -> A barato respecto a B -> comprar A, vender B.
        if z > 0:
            action = f"SHORT {name_a}, LONG {name_b}"
        else:
            action = f"LONG {name_a}, SHORT {name_b}"

        pair = f"{name_a}/{name_b}"
        print(f"⚠️ ANOMALÍA DETECTADA: {pair} | "
              f"Z-Score: {z:+.2f} | Acción: {action}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    exchange = build_exchange()
    prices = download_universe(exchange)

    print(f"\n📊 Matriz de precios alineada: "
          f"{prices.shape[0]} velas x {prices.shape[1]} activos.")

    anomalies = scan_pairs(prices)
    report(anomalies)


if __name__ == "__main__":
    main()
