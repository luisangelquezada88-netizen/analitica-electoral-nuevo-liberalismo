# Tablero electoral — imagen de despliegue (Fase 6)
# Build:  docker build -t tablero-electoral .
# Run:    docker run --rm -p 8501:8501 tablero-electoral
# Datos: solo data/processed/ (<4 MB: parquets + geojson ligero). data/raw,
# notebooks y reports quedan fuera a propósito (no los usa el runtime).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY .streamlit/ .streamlit/
COPY src/ src/
COPY data/processed/ data/processed/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/dashboard_electoral.py", "--server.port=8501", "--server.address=0.0.0.0"]
