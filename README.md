# Analítica electoral — Nuevo Liberalismo y Carlos Fernando Galán

Repositorio de analítica electoral del partido Nuevo Liberalismo y de la
trayectoria del actual alcalde de Bogotá, Carlos Fernando Galán.

## Datasets canónicos

Usamos **dos** datasets finales en `data/processed/`, no uno:

1. `nuevo_liberalismo.parquet` (1.092.364 filas x 13 cols) — histórico electoral
   del **partido** Nuevo Liberalismo. Columnas: `ano, corporacion, coddpto,
   departamento, codmpio, municipio, votos, candidato, partido,
   clave_territorial, coddpto_canon, codmpio_canon, cod_dpto_geo`
   (`cod_dpto_geo` = código DANE que hace join con la capa de departamentos).
2. `carlos_fernando_galan.parquet` (3.179 filas x 11 cols) — 5 elecciones de
   **Carlos Fernando Galán** (2007–2023; se nombra completo para no confundir
   con Luis Carlos Galán), no todas con el partido. Columnas:
   `ano, corporacion, lugar, votacion, localidad, cod_localidad, upz, cod_upz,
   longitud, latitud, geo_flag` (`geo_flag` marca inconsistencias sin borrar filas).
   Fuente raw: `data/raw/historico_carlos_fernando_galan.csv`.
   Ambos se generan con `notebooks/01_limpieza_homologacion.ipynb`.

## Estructura

```
data/raw/1980_1988/      # 16 CSV raw 1980-1988 (se conservan por peso liviano)
data/raw/2021_2026/      # 4 CSV limpios: 80_88, 22_congreso, 23_territoriales, 26_congreso
data/raw/geodata/        # departamentos, localidades, upz (.shp) para joins y mapas
data/raw/historico_carlos_fernando_galan.csv
data/processed/nuevo_liberalismo.parquet
data/processed/carlos_fernando_galan.parquet
notebooks/01_limpieza_homologacion.ipynb  # genera los 2 parquet (Fase 3 lista)
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
