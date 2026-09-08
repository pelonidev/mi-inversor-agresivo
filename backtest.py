"""backtest.py — Simulador vectorizado de la estrategia de pairs trading.

Enfrenta la lógica matemática del `StrategyEngine` (hedge ratio, spread,
Z-Score rolling) a la fricción real del mercado (comisiones taker + slippage)
usando `vectorbt`. Todo el cálculo es 100% vectorizado (numpy/pandas): no se
itera fila a fila.

Uso:
    python backtest.py

Salida:
    - Tear sheet institucional por consola.
    - `backtest_results.html` con la curva de capital y el Z-Score + bandas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt
from plotly.subplots import make_subplots

from src.strategy.kalman import kalman_hedge_batch

# La consola de Windows (cp1252) no codifica β, —, etc.: forzamos UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
#  Parámetros del backtest
# --------------------------------------------------------------------------- #
SYMBOL_A = "BTC/USDT"
SYMBOL_B = "ETH/USDT"

ZSCORE_WINDOW = 100          # ventana móvil del Z-Score
EXIT_Z = 0.0               # cruce de 0 -> cerrar

# --- Filtro de Kalman (hedge ratio adaptativo, sin look-ahead) ---
KALMAN_DELTA = 1e-6         # velocidad de adaptación de β (menor = más suave)
KALMAN_OBS_COV = 2.0        # varianza del ruido de observación (R)

# --- Método de hedge ratio: "kalman" | "rolling_ols" | "static_is" ---
HEDGE_METHOD = "static_is"
ROLLING_OLS_WINDOW = 120    # ventana para rolling_ols
TRAIN_FRAC = 0.5            # static_is: fracción in-sample para fijar β (walk-forward)

# --- Escenario MAKER (órdenes limit): fricción mínima ---
MAKER_FEE = 0.0001          # 0.01% por pata (nivel base Binance/Bybit)
MAKER_SLIPPAGE = 0.0        # las limit garantizan precio -> sin slippage direccional

# --- Fee Hurdle con penalización por Adverse Selection ---
# Las limit solo se llenan cuando el mercado va en tu contra: exigimos que la
# ganancia esperada multiplique por 2.0 el coste de las 4 patas (coste oculto).
ADVERSE_SELECTION_MULT = 2.0
N_LEGS = 4                  # 4 patas: entry A+B y exit A+B

# --- Gate de half-life en TIEMPO ABSOLUTO (justo entre timeframes) ---
MIN_HALF_LIFE_MINUTES = 30.0  # < 30 min de reversión = ruido de alta frecuencia
MAX_HALF_LIFE_HOURS = 48.0    # exploratorio: toleramos reversión más lenta (OOS)

# Timeframes a evaluar (resampleo desde 1m) -> (regla pandas, minutos/vela).
TIMEFRAMES: dict[str, tuple[str, int]] = {
    "15m": ("15min", 15),
    "1h": ("1h", 60),
}

INIT_CASH = 10_000.0

DATA_PARQUET = Path("data") / "pair_prices.parquet"
DATA_CSV = Path("data") / "pair_prices.csv"
OUTPUT_HTML_TMPL = "backtest_results_{tf}.html"


# --------------------------------------------------------------------------- #
#  1. Datos: carga CSV si existe, o genera datos sintéticos cointegrados
# --------------------------------------------------------------------------- #
def load_or_generate_data(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con dos columnas de precios cointegrados.

    Prioridad de fuentes: `data/pair_prices.parquet` (datos reales del
    descargador) > `data/pair_prices.csv` > datos sintéticos cointegrados
    (B paseo aleatorio, A = β·B + spread_OU) para que el script sea ejecutable
    de inmediato sin datos previos.
    """
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
        return df[[SYMBOL_A, SYMBOL_B]].dropna()
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
        return df[[SYMBOL_A, SYMBOL_B]].dropna()

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="min")

    # Activo B (p. ej. ETH): paseo aleatorio geométrico.
    b_returns = rng.normal(0.0, 0.0008, size=n)
    price_b = 3_000.0 * np.exp(np.cumsum(b_returns))

    # Spread estacionario: proceso Ornstein-Uhlenbeck (media-reversión).
    # sigma se escala para que la desviación del spread sea económicamente
    # relevante frente al precio (si no, la fricción lo hace intradeable).
    theta, mu, sigma = 0.02, 0.0, 45.0
    spread = np.zeros(n)
    for i in range(1, n):
        # Recurrencia OU (vectorizar aquí no aporta: dependencia secuencial mínima).
        spread[i] = spread[i - 1] + theta * (mu - spread[i - 1]) + sigma * rng.normal()

    beta_true = 20.0
    price_a = beta_true * price_b + spread + 100.0  # BTC ~ 20*ETH + spread

    return pd.DataFrame(
        {SYMBOL_A: price_a, SYMBOL_B: price_b},
        index=index,
    )


# --------------------------------------------------------------------------- #
#  2. Hedge ratio (Kalman adaptativo o Rolling OLS "perezoso") + Z-Score rolling
# --------------------------------------------------------------------------- #
def rolling_ols_beta(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """Hedge ratio β vía Rolling OLS (regresión móvil lenta), 100% vectorizado.

    β_t = cov_w(A, B) / var_w(B) sobre una ventana móvil. Al ser lento, β no se
    "traga" la desviación del precio, dejando que el spread respire y revierta.
    """
    cov = price_a.rolling(window).cov(price_b)
    var = price_b.rolling(window).var()
    beta = (cov / var).bfill()
    beta.name = "beta"
    return beta


def compute_spread_zscore(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calcula spread y Z-Score con un hedge ratio β causal (sin look-ahead).

    Según HEDGE_METHOD, β proviene del filtro de Kalman (reactivo) o de un
    Rolling OLS lento (perezoso). El spread económico usa β rezagado una vela;
    el Z-Score aplica media/desviación estrictamente móviles (rolling).

    Devuelve (zscore, beta, roll_std, spread) — beta es una Serie temporal.
    """
    if HEDGE_METHOD == "rolling_ols":
        beta = rolling_ols_beta(price_a, price_b, ROLLING_OLS_WINDOW)
    elif HEDGE_METHOD == "static_is":
        # β fijo estimado SOLO con el tramo in-sample (walk-forward, sin look-ahead).
        n_train = max(30, int(len(price_a) * TRAIN_FRAC))
        a_tr, b_tr = price_a.iloc[:n_train], price_b.iloc[:n_train]
        beta_val = float(np.cov(a_tr, b_tr)[0, 1] / np.var(b_tr))
        beta = pd.Series(beta_val, index=price_a.index, name="beta")
    else:  # "kalman"
        beta_arr, _alpha_arr, _spread_arr = kalman_hedge_batch(
            price_a.to_numpy(),
            price_b.to_numpy(),
            delta=KALMAN_DELTA,
            obs_cov=KALMAN_OBS_COV,
        )
        beta = pd.Series(beta_arr, index=price_a.index, name="beta")

    # Spread ECONÓMICO con β causal (rezagado una vela) para evitar look-ahead.
    beta_lag = beta.shift(1).bfill()
    spread = price_a - beta_lag * price_b
    spread.name = "spread"

    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    zscore = (spread - roll_mean) / roll_std
    return zscore, beta, roll_std, spread


# --------------------------------------------------------------------------- #
#  2b. Filtros anti-fricción: Half-Life (OU) y Fee Hurdle
# --------------------------------------------------------------------------- #
def compute_half_life(spread: pd.Series) -> float:
    """Half-life de reversión a la media del spread (proceso OU), en velas.

    Ajusta el modelo discreto de Ornstein-Uhlenbeck por MCO:
        Δspread_t = α + λ·spread_{t-1} + ε_t
    La half-life es  −ln(2)/λ  (con λ < 0 para que haya reversión). Si λ >= 0
    no hay reversión (spread explosivo/random walk) -> half-life infinita.
    """
    s = spread.dropna()
    lag = s.shift(1).dropna()
    s = s.loc[lag.index]
    delta = s - lag

    # OLS: np.polyfit devuelve [pendiente(λ), intercepto].
    lam, _intercept = np.polyfit(lag.to_numpy(), delta.to_numpy(), 1)
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def fee_hurdle_zscore(
    price_a: pd.Series,
    roll_std: pd.Series,
    fee: float,
    slippage: float,
    hurdle_mult: float,
    n_legs: int = N_LEGS,
) -> pd.Series:
    """Umbral de entrada dinámico en unidades de Z-Score (Fee Hurdle).

    Derivación (par dólar-neutral, notional N por pata):
        Beneficio esperado por reversión ($) = |z|·σ_spread·(N / P_A)
        Coste round-trip ($)                 = n_legs·(fee+slippage)·N
    Exigimos beneficio >= hurdle_mult·coste, de donde el umbral en z:

        z_hurdle = hurdle_mult · n_legs · (fee + slippage) · P_A / σ_spread

    Por debajo de este umbral la reversión NO cubre la fricción -> no operar.
    """
    cost = fee + slippage
    return (hurdle_mult * n_legs * cost) * (price_a / roll_std)


# --------------------------------------------------------------------------- #
#  3. Generación vectorizada de señales (umbral dinámico + gate de half-life)
# --------------------------------------------------------------------------- #
def build_signals(
    zscore: pd.Series,
    hurdle: pd.Series,
    half_life: float,
    min_half_life: float,
    max_half_life: float,
) -> dict[str, pd.DataFrame]:
    """Construye señales de entrada/salida con umbral dinámico (sin bucles).

    - Entrada SOLO si |Z| supera el Fee Hurdle dinámico (gana a la fricción).
    - Gate de half-life: si la reversión es demasiado rápida (ruido HF) o
      demasiado lenta, se prohíben TODAS las entradas.
    - Salida al cruzar el 0 (reversión a la media).
    """
    z = zscore.fillna(0.0)
    h = hurdle.reindex(z.index).ffill().fillna(np.inf)

    tradeable = min_half_life <= half_life <= max_half_life
    if not tradeable:
        print(f"  [FILTRO] Half-life={half_life:.1f} velas fuera de "
              f"[{min_half_life:.0f}, {max_half_life:.0f}] -> sin operativa")

    # Zonas de entrada gobernadas por el umbral dinámico (fee hurdle).
    long_zone = (z <= -h) & tradeable      # spread barato -> LONG A / SHORT B
    short_zone = (z >= h) & tradeable      # spread caro   -> SHORT A / LONG B

    # Flanco de subida de la zona = instante de apertura.
    long_entry_a = long_zone & ~long_zone.shift(1, fill_value=False)
    short_entry_a = short_zone & ~short_zone.shift(1, fill_value=False)

    # Salidas: cruce del 0 (reversión a la media).
    z_prev = z.shift(1).fillna(0.0)
    cross_zero_up = (z_prev < EXIT_Z) & (z >= EXIT_Z)
    cross_zero_dn = (z_prev > EXIT_Z) & (z <= EXIT_Z)

    cols = [SYMBOL_A, SYMBOL_B]

    def frame(series_a: pd.Series, series_b: pd.Series) -> pd.DataFrame:
        return pd.concat([series_a, series_b], axis=1, keys=cols)

    # Activo A directo; activo B invertido (hedge).
    long_entries = frame(long_entry_a, short_entry_a)     # A long / B long
    long_exits = frame(cross_zero_up, cross_zero_dn)
    short_entries = frame(short_entry_a, long_entry_a)    # A short / B short
    short_exits = frame(cross_zero_dn, cross_zero_up)

    return {
        "long_entries": long_entries,
        "long_exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
    }


# --------------------------------------------------------------------------- #
#  4. Portfolio con fricción configurable (maker/taker)
# --------------------------------------------------------------------------- #
def run_portfolio(
    prices: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    fee: float,
    slippage: float,
    freq: str,
) -> vbt.Portfolio:
    """Simula el portfolio de pairs trading con la fricción indicada."""
    return vbt.Portfolio.from_signals(
        close=prices,
        entries=signals["long_entries"],
        exits=signals["long_exits"],
        short_entries=signals["short_entries"],
        short_exits=signals["short_exits"],
        fees=fee,
        slippage=slippage,
        init_cash=INIT_CASH,
        cash_sharing=True,      # ambas patas comparten el mismo capital
        group_by=True,          # tratar A y B como un único portfolio
        freq=freq,
    )


# --------------------------------------------------------------------------- #
#  5. Reporte institucional (tear sheet)
# --------------------------------------------------------------------------- #
def print_tear_sheet(pf: vbt.Portfolio, label: str) -> dict[str, float]:
    """Imprime las métricas clave por consola y devuelve un resumen."""
    total_return = float(pf.total_return()) * 100
    ann_return = float(pf.annualized_return()) * 100
    max_dd = float(pf.max_drawdown()) * 100
    total_fees = float(pf.orders.fees.sum())

    try:
        n_trades = int(pf.trades.count())
    except Exception:  # noqa: BLE001
        n_trades = 0

    if n_trades == 0:
        sharpe = 0.0
        win_rate = 0.0
    else:
        sharpe = float(pf.sharpe_ratio())
        try:
            win_rate = float(pf.trades.win_rate()) * 100
        except Exception:  # noqa: BLE001
            win_rate = 0.0

    line = "=" * 52
    print(f"\n{line}")
    print(f"  TEAR SHEET [{label}] — MAKER — {SYMBOL_A} vs {SYMBOL_B}")
    print(line)
    print(f"  {'Total Return':<28}: {total_return:>14.2f} %")
    print(f"  {'Annualized Return':<28}: {ann_return:>14.2f} %")
    print(f"  {'Max Drawdown':<28}: {max_dd:>14.2f} %")
    print(f"  {'Sharpe Ratio':<28}: {sharpe:>14.2f}")
    print(f"  {'Win Rate':<28}: {win_rate:>14.2f} %")
    print(f"  {'Total Trades':<28}: {n_trades:>14d}")
    print(f"  {'Total Fees Paid (USD)':<28}: {total_fees:>14.2f}")
    if n_trades == 0:
        print("  [Fee Hurdle bloqueó todas las entradas: sin exposición]")
    print(line + "\n")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": float(n_trades),
        "total_fees": total_fees,
    }


# --------------------------------------------------------------------------- #
#  6. Visualización -> backtest_results.html
# --------------------------------------------------------------------------- #
def export_html(pf: vbt.Portfolio, zscore: pd.Series, hurdle: pd.Series, label: str) -> None:
    """Exporta curva de capital + Z-Score con el umbral dinámico (Fee Hurdle)."""
    equity = pf.value()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=("Equity Curve (USD)", "Z-Score vs Fee Hurdle dinámico"),
    )

    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="#2ca02c")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, name="Z-Score", line=dict(color="#1f77b4")),
        row=2, col=1,
    )
    # Umbral dinámico (fee hurdle) como banda superior e inferior.
    fig.add_trace(
        go.Scatter(x=hurdle.index, y=hurdle.values, name="+Hurdle",
                   line=dict(color="red", dash="dash")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=hurdle.index, y=-hurdle.values, name="-Hurdle",
                   line=dict(color="red", dash="dash")),
        row=2, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="gray", dash="dot"), row=2, col=1)

    fig.update_layout(
        title=f"Backtest MAKER [{label}] — {SYMBOL_A} / {SYMBOL_B}",
        template="plotly_dark",
        height=800,
        showlegend=True,
    )
    out = Path(OUTPUT_HTML_TMPL.format(tf=label))
    fig.write_html(str(out))
    print(f"Gráfica exportada a: {out.resolve()}")


# --------------------------------------------------------------------------- #
#  Orquestación multi-timeframe
# --------------------------------------------------------------------------- #
def resample_prices(prices_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resamplea los closes de 1m al timeframe indicado (last del intervalo)."""
    return prices_1m.resample(rule).last().dropna()


def run_for_timeframe(
    prices_1m: pd.DataFrame,
    label: str,
    rule: str,
    tf_minutes: int,
) -> dict[str, float]:
    """Ejecuta el pipeline completo para un timeframe y devuelve el resumen."""
    prices = resample_prices(prices_1m, rule)
    zscore, beta, roll_std, spread = compute_spread_zscore(
        prices[SYMBOL_A], prices[SYMBOL_B], ZSCORE_WINDOW
    )

    half_life = compute_half_life(spread)
    hurdle = fee_hurdle_zscore(
        prices[SYMBOL_A], roll_std,
        fee=MAKER_FEE, slippage=MAKER_SLIPPAGE, hurdle_mult=ADVERSE_SELECTION_MULT,
    )

    # Gate de half-life convertido a velas de este timeframe (tiempo absoluto).
    min_half_life = MIN_HALF_LIFE_MINUTES / tf_minutes
    max_half_life = MAX_HALF_LIFE_HOURS * 60.0 / tf_minutes

    print(f"\n----- Timeframe {label} ({len(prices)} velas) -----")
    print(f"Hedge ratio [{HEDGE_METHOD}] (β): último={beta.iloc[-1]:.4f}  "
          f"medio={beta.mean():.4f}  rango=[{beta.min():.4f}, {beta.max():.4f}]")
    print(f"Half-life: {half_life:.1f} velas ({half_life * tf_minutes / 60:.1f} h)  "
          f"| gate velas=[{min_half_life:.1f}, {max_half_life:.0f}]")
    print(f"Fee Hurdle Z medio: {hurdle.replace(np.inf, np.nan).mean():.2f} "
          f"(coste round-trip maker {N_LEGS * MAKER_FEE * 100:.2f}% × {ADVERSE_SELECTION_MULT:.1f})")

    signals = build_signals(zscore, hurdle, half_life, min_half_life, max_half_life)
    pf = run_portfolio(prices, signals, fee=MAKER_FEE, slippage=MAKER_SLIPPAGE, freq=rule)

    summary = print_tear_sheet(pf, label)
    export_html(pf, zscore, hurdle, label)
    return summary


def main() -> None:
    global SYMBOL_A, SYMBOL_B, DATA_PARQUET

    parser = argparse.ArgumentParser(description="Backtest de pairs trading")
    parser.add_argument("--data", type=str, default=None, help="Parquet de precios")
    parser.add_argument("--symbol-a", type=str, default=None, help="Columna activo A")
    parser.add_argument("--symbol-b", type=str, default=None, help="Columna activo B")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="Lista separada por comas, p.ej. '15m,1h' o '1h'")
    args = parser.parse_args()

    if args.symbol_a:
        SYMBOL_A = args.symbol_a
    if args.symbol_b:
        SYMBOL_B = args.symbol_b
    if args.data:
        DATA_PARQUET = Path(args.data)

    timeframes = TIMEFRAMES
    if args.timeframes:
        wanted = {tf.strip() for tf in args.timeframes.split(",")}
        timeframes = {k: v for k, v in TIMEFRAMES.items() if k in wanted}

    prices_1m = load_or_generate_data()
    print(f"Par: {SYMBOL_A} / {SYMBOL_B}  |  Método hedge: {HEDGE_METHOD}")
    print(f"Datos base: {len(prices_1m)} velas "
          f"({prices_1m.index[0]} -> {prices_1m.index[-1]})")

    results: dict[str, dict[str, float]] = {}
    for label, (rule, tf_minutes) in timeframes.items():
        results[label] = run_for_timeframe(prices_1m, label, rule, tf_minutes)

    # Resumen comparativo y veredicto frente al gate (Sharpe>1.5 y profit>0).
    line = "#" * 60
    print(f"\n{line}")
    print("  VEREDICTO FINAL (gate: Sharpe > 1.5 y Profit neto > 0)")
    print(line)
    for label, r in results.items():
        passed = r["sharpe"] > 1.5 and r["total_return"] > 0
        verdict = "PASA" if passed else "NO PASA"
        print(f"  {label:>4}: Return={r['total_return']:>8.2f}%  "
              f"Sharpe={r['sharpe']:>6.2f}  Trades={int(r['n_trades']):>4d}  -> {verdict}")
    print(line + "\n")


if __name__ == "__main__":
    main()
