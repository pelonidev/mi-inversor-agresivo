"""Carga y validación centralizada de la configuración del bot.

Usa pydantic-settings para leer variables de entorno desde `.env` y validarlas
en el arranque. Si falta un parámetro crítico, el bot falla rápido (fail-fast)
antes de arriesgar capital.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración inmutable del bot, validada al arranque."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Exchange ---
    exchange_id: str = Field(default="binance")
    exchange_api_key: str = Field(default="")
    exchange_api_secret: str = Field(default="")
    use_testnet: bool = Field(default=True)

    # --- Riesgo ---
    account_equity_usd: float = Field(default=10_000.0, gt=0)
    max_daily_drawdown_pct: float = Field(default=0.01, gt=0, lt=1)
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)
    max_gross_exposure_pct: float = Field(default=2.0, gt=0)
    max_position_pct: float = Field(default=0.10, gt=0, le=1)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @field_validator("exchange_id")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower().strip()


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia única (singleton) de la configuración."""
    return Settings()
