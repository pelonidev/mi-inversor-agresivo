"""walk_forward.py — Walk-Forward Analysis riguroso (aniquila el look-ahead bias).

Valida la estrategia de pairs trading en OUT-OF-SAMPLE puro mediante ventanas
rodantes. En cada iteración:

    1.  El screener SOLO ve los 30 días In-Sample (Train) para elegir el/los
        mejores pares cointegrados (p<0.05, half-life 4-48h). No ve el futuro.
    2.  Ese par se opera EXCLUSIVAMENTE en los 15 días siguientes (Test/OOS),
        con β actualizado por Rolling OLS causal y fricción Maker + hurdle ×2.0.

Los retornos OOS de las 4 ventanas se concatenan en una única curva de capital
continua (~60 días de trading real) y se reporta su Tear Sheet agregado.

Uso:
    python walk_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from screener import UNIVERSE, evaluate_pair, half_life

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
#  Parámetros del Walk-Forward
# --------------------------------------------------------------------------- #
CANDLES_PER_DAY = 24         # velas de 1h por día
TRAIN_CANDLES = 30 * CANDLES_PER_DAY   # 720  (30 días In-Sample)
TEST_CANDLES = 15 * CANDLES_PER_DAY    # 360  (15 días Out-of-Sample)
STEP_CANDLES = 15 * CANDLES_PER_DAY    # 360  (desplazamiento -> 4 ventanas)

# Selección de pares (solo sobre el train).
P_VALUE_MAX = 0.05
HALF_LIFE_MIN = 4.0          # velas (horas)
HALF_LIFE_MAX = 48.0         # velas (horas) — holding period 1-2 días aceptado
TOP_N = 2                    # nº de pares a operar por ventana

# Modelo y fricción.
ZSCORE_WINDOW = 100
ROLLING_OLS_WINDOW = 120
MAKER_FEE = 0.0001           # 0.01% por pata
ADVERSE_SELECTION_MULT = 2.0
N_LEGS = 4
INIT_CASH = 10_000.0
FREQ = "1h"
PERIODS_PER_YEAR = 365 * 24

DATA_PARQUET = Path("data") / "universe_90d_1h.parquet"
OUTPUT_HTML = Path("walk_forward_results.html")


# --------------------------------------------------------------------------- #
#  Indicadores causales (Rolling OLS + Z-Score + Fee Hurdle)
# --------------------------------------------------------------------------- #
def rolling_ols_beta(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """β = cov_w(A,B)/var_w(B) móvil (causal)."""
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    return (cov / var).bfill()


def spread_zscore_hurdle(
    a: pd.Series,
    b: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Devuelve (zscore, hurdle) causales para el par (Rolling OLS β)."""
    beta = rolling_ols_beta(a, b, ROLLING_OLS_WINDOW)
    beta_lag = beta.shift(1).bfill()
    spread = a - beta_lag * b

    roll_mean = spread.rolling(ZSCORE_WINDOW).mean()
    roll_std = spread.rolling(ZSCORE_WINDOW).std()
    zscore = (spread - roll_mean) / roll_std

    cost = MAKER_FEE  # maker sin slippage
    hurdle = (ADVERSE_SELECTION_MULT * N_LEGS * cost) * (a / roll_std)
    return zscore, hurdle


def build_signals(
    zscore: pd.Series,
    hurdle: pd.Series,
    sym_a: str,
    sym_b: str,
) -> dict[str, pd.DataFrame]:
    """Entradas por umbral dinámico (fee hurdle); salidas al cruzar 0."""
    z = zscore.fillna(0.0)
    h = hurdle.reindex(z.index).ffill().fillna(np.inf)

    long_zone = z <= -h
    short_zone = z >= h
    long_entry = long_zone & ~long_zone.shift(1, fill_value=False)
    short_entry = short_zone & ~short_zone.shift(1, fill_value=False)

    z_prev = z.shift(1).fillna(0.0)
    cross_zero_up = (z_prev < 0) & (z >= 0)
    cross_zero_dn = (z_prev > 0) & (z <= 0)

    cols = [sym_a, sym_b]

    def frame(sa: pd.Series, sb: pd.Series) -> pd.DataFrame:
        return pd.concat([sa, sb], axis=1, keys=cols)

    return {
        "long_entries": frame(long_entry, short_entry),
        "long_exits": frame(cross_zero_up, cross_zero_dn),
        "short_entries": frame(short_entry, long_entry),
        "short_exits": frame(cross_zero_dn, cross_zero_up),
    }


# --------------------------------------------------------------------------- #
#  Selección In-Sample (aislada del futuro)
# --------------------------------------------------------------------------- #
def select_pairs_in_sample(train: pd.DataFrame) -> list[tuple[str, str]]:
    """Escanea todas las combinaciones SOLO con datos train y devuelve Top-N."""
    import itertools

    records = []
    for sym_a, sym_b in itertools.combinations(train.columns, 2):
        a, b = train[sym_a], train[sym_b]
        p_ab, hl_ab, _ = evaluate_pair(a, b)
        p_ba, hl_ba, _ = evaluate_pair(b, a)
        if p_ab <= p_ba:
            records.append((sym_a, sym_b, p_ab, hl_ab))
        else:
            records.append((sym_b, sym_a, p_ba, hl_ba))

    df = pd.DataFrame(records, columns=["a", "b", "p_value", "half_life"])
    passed = df[
        (df["p_value"] < P_VALUE_MAX)
        & (df["half_life"] >= HALF_LIFE_MIN)
        & (df["half_life"] <= HALF_LIFE_MAX)
    ].sort_values("p_value")
    return [(r.a, r.b) for r in passed.head(TOP_N).itertuples()]


# --------------------------------------------------------------------------- #
#  Ejecución OOS de un par en una ventana de test
# --------------------------------------------------------------------------- #
def run_oos_pair(
    full: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    warmup_start: int,
    test_start: int,
    test_end: int,
) -> vbt.Portfolio | None:
    """Calcula indicadores con warmup y opera SOLO en [test_start, test_end)."""
    # Indicadores sobre [warmup_start, test_end) para que el rolling tenga historia.
    window = full.iloc[warmup_start:test_end]
    a, b = window[sym_a], window[sym_b]
    zscore, hurdle = spread_zscore_hurdle(a, b)
    signals = build_signals(zscore, hurdle, sym_a, sym_b)

    # Recorte estricto a la ventana OOS (aquí sí se abre/cierra dinero).
    test_index = full.index[test_start:test_end]
    prices_test = window.loc[test_index, [sym_a, sym_b]]
    sig_test = {k: v.loc[test_index] for k, v in signals.items()}

    return vbt.Portfolio.from_signals(
        close=prices_test,
        entries=sig_test["long_entries"],
        exits=sig_test["long_exits"],
        short_entries=sig_test["short_entries"],
        short_exits=sig_test["short_exits"],
        fees=MAKER_FEE,
        slippage=0.0,
        init_cash=INIT_CASH,
        cash_sharing=True,
        group_by=True,
        freq=FREQ,
    )


# --------------------------------------------------------------------------- #
#  Métricas agregadas sobre la curva OOS concatenada
# --------------------------------------------------------------------------- #
def aggregate_and_report(
    returns: pd.Series,
    n_trades: int,
    n_wins: int,
    total_fees: float,
) -> None:
    equity = INIT_CASH * (1.0 + returns).cumprod()
    total_return = (equity.iloc[-1] / INIT_CASH - 1.0) * 100 if len(equity) else 0.0

    mean, std = returns.mean(), returns.std()
    sharpe = float(mean / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0

    running_max = equity.cummax()
    max_dd = float(((equity - running_max) / running_max).min() * 100) if len(equity) else 0.0
    win_rate = (n_wins / n_trades * 100) if n_trades else 0.0

    line = "=" * 56
    print(f"\n{line}")
    print("  TEAR SHEET — WALK-FORWARD OOS COMBINADO (~60 días)")
    print(line)
    print(f"  {'Total Return (OOS)':<30}: {total_return:>14.2f} %")
    print(f"  {'Sharpe Ratio (OOS)':<30}: {sharpe:>14.2f}")
    print(f"  {'Max Drawdown (OOS)':<30}: {max_dd:>14.2f} %")
    print(f"  {'Win Rate':<30}: {win_rate:>14.2f} %")
    print(f"  {'Total Trades':<30}: {n_trades:>14d}")
    print(f"  {'Total Fees Paid (USD)':<30}: {total_fees:>14.2f}")
    print(line)
    passed = sharpe > 1.2 and total_return > 0
    print(f"  GATE (Sharpe>1.2 y profit>0): {'PASA ✅' if passed else 'NO PASA ❌'}")
    print(line + "\n")

    _export_html(equity)


def _export_html(equity: pd.Series) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity OOS",
                             line=dict(color="#2ca02c")))
    fig.add_hline(y=INIT_CASH, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title="Walk-Forward OOS combinado — Equity Curve continua",
        template="plotly_dark", height=600,
    )
    fig.write_html(str(OUTPUT_HTML))
    print(f"Gráfica exportada a: {OUTPUT_HTML.resolve()}")


# --------------------------------------------------------------------------- #
#  Orquestación del Walk-Forward
# --------------------------------------------------------------------------- #
def main() -> None:
    prices = pd.read_parquet(DATA_PARQUET)[UNIVERSE].dropna()
    n = len(prices)
    print(f"Dataset: {n} velas 1h ({prices.index[0]} -> {prices.index[-1]})")

    all_returns: list[pd.Series] = []
    total_trades = 0
    total_wins = 0
    total_fees = 0.0
    window_id = 0

    start = 0
    while start + TRAIN_CANDLES + TEST_CANDLES <= n:
        train_start = start
        test_start = start + TRAIN_CANDLES
        test_end = test_start + TEST_CANDLES
        window_id += 1

        train = prices.iloc[train_start:test_start]
        selected = select_pairs_in_sample(train)

        t0 = prices.index[test_start]
        t1 = prices.index[test_end - 1]
        print(f"\n--- Ventana {window_id}: TEST {t0.date()} -> {t1.date()} ---")
        if not selected:
            print("  Sin pares cointegrados en el train -> ventana en efectivo")
            start += STEP_CANDLES
            continue
        print(f"  Pares elegidos (in-sample): {[f'{a}/{b}' for a, b in selected]}")

        # Cada par elegido recibe una fracción igual del capital de la ventana.
        window_returns: list[pd.Series] = []
        for sym_a, sym_b in selected:
            pf = run_oos_pair(prices, sym_a, sym_b, train_start, test_start, test_end)
            if pf is None:
                continue
            window_returns.append(pf.returns())
            n_tr = int(pf.trades.count())
            total_trades += n_tr
            total_wins += int(pf.trades.winning.count()) if n_tr else 0
            total_fees += float(pf.orders.fees.sum())
            print(f"    {sym_a}/{sym_b}: return={float(pf.total_return()) * 100:>6.2f}%  "
                  f"trades={n_tr}")

        if window_returns:
            # Media de los retornos de los pares (cartera equiponderada).
            combined = pd.concat(window_returns, axis=1).mean(axis=1)
            all_returns.append(combined)

        start += STEP_CANDLES

    if not all_returns:
        print("\nNingún trade OOS en ninguna ventana.")
        return

    oos_returns = pd.concat(all_returns)
    aggregate_and_report(oos_returns, total_trades, total_wins, total_fees)


if __name__ == "__main__":
    main()
