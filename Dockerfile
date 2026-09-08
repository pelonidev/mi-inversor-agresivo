# --- Imagen ultraligera para el Funding Radar daemon (24/7) ---
FROM python:3.11-slim

# Logs sin buffer (flush inmediato a stdout -> docker logs) y sin .pyc.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

# Instala solo las dependencias del runtime del daemon (build rápido y ligero).
COPY requirements-daemon.txt .
RUN pip install --no-cache-dir -r requirements-daemon.txt

# Usuario NO-root por seguridad (el contenedor no corre como root).
RUN useradd --create-home --uid 10001 radar

# Directorio de estado (paper trading) escribible por el usuario radar.
# Se monta como volumen en runtime para persistir entre actualizaciones.
RUN mkdir -p /app/data && chown radar:radar /app/data

# Copia únicamente el código necesario para el daemon (no research/datos).
COPY --chown=radar:radar src/ ./src/
COPY --chown=radar:radar daemon.py funding_radar.py liquidity_gate.py execution.py performance_tracker.py ./

USER radar

# Vigilante: escanea el top-100 de perps cada 5 minutos.
CMD ["python", "daemon.py", "--interval", "300", "--top", "100"]
