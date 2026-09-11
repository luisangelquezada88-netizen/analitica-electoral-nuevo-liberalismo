# Analítica electoral — Nuevo Liberalismo y Carlos Fernando Galán

Repositorio de analítica electoral del partido Nuevo Liberalismo y de la
trayectoria del actual alcalde de Bogotá, Carlos Fernando Galán.

## Datasets canónicos

Usamos **dos** datasets finales en `data/processed/`, no uno:

1. `dataset_final.parquet` (1.092.364 filas x 12 cols) — histórico electoral
   del **partido** Nuevo Liberalismo. Columnas: `ano, corporacion, coddpto,
   departamento, codmpio, municipio, votos, candidato, partido,
   clave_territorial, coddpto_canon, codmpio_canon`.
2. `historico_galan_limpio.parquet` (pendiente, Fase 3) — 5 elecciones de
   **Carlos Fernando Galán**, no todas con el partido. Fuente raw:
   `data/raw/historico_carlos_fernando_galan.csv` (~3.179 filas:
   año, corporación, lugar, votación, localidad, longitud, latitud, upz).

## Estructura

```
data/raw/1980_1988/      # 16 CSV raw 1980-1988 (se conservan por peso liviano)
data/raw/2021_2026/      # 4 CSV limpios: 80_88, 22_congreso, 23_territoriales, 26_congreso
data/raw/geodata/        # loca + upz-bogota (.shp) para análisis de Bogotá
data/raw/historico_carlos_fernando_galan.csv
data/processed/dataset_final.parquet
notebooks/               # 01_limpieza (Fase 3), 02_geo + 03_hdbscan (Fase 4)
src/dashboard_electoral.py  # Fase 5, usa path relativo a data/processed/
docs/Insumo.docx
reports/galan.pbix       # Power BI, se versiona (0.9 MB)
```

Eliminados en Fase 1: `data/2021-actualidad/` (crudos gigantes + triplicados
xlsx/csv/parciales), `data/master/` (redundante), `output/` (intermedios).

## Reproducibilidad

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Consultas SQL en notebooks: solo **DuckDB** (`duckdb.sql()` sobre DataFrames).
No usar SQLAlchemy ni sqlite3 (ver Fase 4).

## Roadmap

- Fase 1-2: organización + reproducibilidad (hecho).
- Fase 3: mejorar cuaderno de limpieza.
- Fase 4: dividir `galan.ipynb` en geoespacial + HDBSCAN.
- Fase 5: rehacer tablero.
- Fase 6: desplegar.
