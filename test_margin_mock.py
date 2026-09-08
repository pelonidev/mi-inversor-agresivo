"""test_margin_mock.py — Prueba aislada del MarginManager (sin exchange real).

Inyecta un `MockExchange` en el MarginManager y simula que el perpetuo corto se
mueve en nuestra contra: el margin ratio salta de 0.50 a 0.85. Verifica que el
manager transfiere el importe EXACTO de spot a futuros para devolver el ratio a
la zona segura (target 0.60).

Ejecutar:
    python test_margin_mock.py
"""
from __future__ import annotations

import asyncio

from execution import MarginConfig, MarginManager
from src.core.logger import configure_logging
from src.risk.risk_manager import RiskLimits, RiskManager


class MockExchange:
    """Simula fetch_balance() y transfer() de un exchange (spot + futuros)."""

    def __init__(self, maint: float, margin_balance: float, spot_free: float) -> None:
        self.maint = maint                    # maintenance margin (constante)
        self.margin_balance = margin_balance  # equity de futuros (baja si perdemos)
        self.spot_free = spot_free
        self.transfers: list[dict] = []

    async def fetch_balance(self, params: dict | None = None) -> dict:
        return {
            "info": {
                "totalMaintMargin": str(self.maint),
                "totalMarginBalance": str(self.margin_balance),
            },
            "USDT": {
                "free": self.spot_free,
                "used": 0.0,
                "total": self.spot_free,
            },
        }

    async def transfer(self, code: str, amount: float, from_acc: str, to_acc: str) -> dict:
        # Mover USDT de spot -> futuros: baja el spot, sube el equity de futuros.
        self.spot_free -= amount
        self.margin_balance += amount
        self.transfers.append(
            {"code": code, "amount": amount, "from": from_acc, "to": to_acc}
        )
        return {"id": f"mock-{len(self.transfers)}", "amount": amount}

    def simulate_adverse_move(self, new_margin_balance: float) -> None:
        """El corto pierde -> el unrealized PnL reduce el equity de futuros."""
        self.margin_balance = new_margin_balance


def _ratio(mock: MockExchange) -> float:
    return mock.maint / mock.margin_balance


async def main() -> None:
    configure_logging("WARNING")  # silencioso salvo eventos de riesgo

    # Estado inicial SANO: maint=50, equity=100 -> ratio 0.50.
    mock = MockExchange(maint=50.0, margin_balance=100.0, spot_free=1_000.0)
    risk = RiskManager(RiskLimits(account_equity_usd=2_000.0))
    mm = MarginManager(
        spot_exchange=mock,   # type: ignore[arg-type]
        perp_exchange=mock,   # type: ignore[arg-type]
        risk_manager=risk,
        config=MarginConfig(danger_ratio=0.80, target_ratio=0.60, min_spot_reserve_usd=100.0),
    )

    line = "=" * 56
    print(line)
    print("  TEST — MarginManager (recapitalización exacta)")
    print(line)

    # 1) Estado sano: NO debe transferir.
    r1 = await mm.check_once()
    print(f"  [1] Ratio inicial          : {r1:.4f}  (sano, sin acción)")
    assert abs(r1 - 0.50) < 1e-6, "ratio inicial esperado 0.50"
    assert not mock.transfers, "no debería transferir en zona segura"

    # 2) El perp se dispara en contra: el equity de futuros cae -> ratio 0.85.
    mock.simulate_adverse_move(new_margin_balance=50.0 / 0.85)  # ratio = 0.85
    print(f"  [2] Movimiento adverso     : ratio sube a {_ratio(mock):.4f}  (PELIGRO)")

    # 3) El manager detecta el peligro y recapitaliza el importe EXACTO.
    r_before = await mm.check_once()
    print(f"  [3] Ratio detectado        : {r_before:.4f}  -> dispara top-up")

    # 4) Verificación del transfer y del ratio final.
    r_after = await mm.get_margin_ratio()
    transferred = mock.transfers[-1]["amount"] if mock.transfers else 0.0
    expected = 50.0 / 0.60 - (50.0 / 0.85)  # X = maint/target - marginBalance

    print(line)
    print(f"  Transferencias ejecutadas  : {len(mock.transfers)}")
    print(f"  Importe transferido        : {transferred:.2f} USDT")
    print(f"  Importe teórico exacto     : {expected:.2f} USDT")
    print(f"  Spot restante              : {mock.spot_free:.2f} USDT")
    print(f"  Ratio FINAL                : {r_after:.4f}  (objetivo 0.60)")
    print(line)

    assert len(mock.transfers) == 1, "debía ejecutar exactamente 1 transferencia"
    assert abs(transferred - expected) < 0.05, "importe transferido no es el exacto"
    assert abs(r_after - 0.60) < 1e-3, "el ratio no volvió a la zona segura 0.60"
    assert not risk.is_halted, "no debía dispararse el Kill Switch"

    print("  ✅ TEST OK: la inyección de capital devuelve el ratio a 0.60.")
    print(line)


if __name__ == "__main__":
    asyncio.run(main())
