# Analítica electoral — Nuevo Liberalismo y Carlos Fernando Galán

Repositorio de analítica electoral geoespacial del partido Nuevo Liberalismo y de la trayectoria del actual alcalde de Bogotá, Carlos Fernando Galán.

## Tablero: cómo verlo

1. **Nube (público):** https://nuevo-liberalismo.streamlit.app/ — sin instalar nada.
   *Nota:* si la app está dormida por inactividad, presiona `Yes, get this app back up!` y espera ~1 min a que despierte.
2. **Local (Windows):** doble clic a `run_dashboard.ps1` (o `Ejecutar con PowerShell`),
   o en terminal desde la raíz:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   streamlit run src/dashboard_electoral.py
   ```
   Requiere `pip install -r requirements.txt` una sola vez. Abre http://localhost:8501.
3. **Docker (local o VPS):**
   ```bash
   docker compose up --build -d   # http://localhost:8501
   # o manual:
   docker build -t tablero-electoral .
   docker run --rm -p 8501:8501 tablero-electoral
   ```
   La imagen solo lleva el runtime (`requirements-docker.txt`: pandas, plotly,
   streamlit, pydeck, scikit-learn…), `src/`, `data/processed/` (<4 MB) y
   `.streamlit/`. Requiere Docker Engine en ejecución.


## Fuentes de datos

Todos los resultados electorales provienen de la **Registraduría Nacional del Estado Civil**, vía dos portales:

| Portal | URL | Cobertura en este repo |
|---|---|---|
| Observatorio Electoral | https://observatorio.registraduria.gov.co/views/electoral/historicos-resultados.php | Registros más recientes (2021–2026) |
| Centro de Estudios en Democracia y Asuntos Electorales (CEDAE) | https://cedae.datasketch.co/datos-democracia/resultados-electorales/descarga-los-datos/ | Registros más viejos (1980–1988) |

## Datasets canónicos

| Dataset | Filas × Cols | Contenido |
|---|---|---|
| `data/processed/nuevo_liberalismo.parquet` | 1.092.477 × 13 | Histórico electoral del **partido** (1982–2026). `cod_dpto_geo` = código DANE para el join con departamentos. Incluye 113 filas de excepción (Luis Carlos Galán, Senado 1982, Partido Liberal). |
| `data/processed/carlos_fernando_galan.parquet` | 3.179 × 14 | 5 elecciones de **Carlos Fernando Galán** (2007–2023; nombre completo para no confundir con Luis Carlos Galán), no todas con el partido. `geo_flag` marca inconsistencias sin borrar filas; `nk_*`/`hdb` precomputados para el tablero. |

Archivos de apoyo (generados, no editar a mano): `agg_depto.parquet` (33×3, agregado por departamento), `agg_puestos_galan.parquet` (1.136×7, agregado por puesto de votación) y `geo/*.geojson` (3 capas ligeras pre-simplificadas). Todo se genera con `notebooks/01_limpieza.ipynb`; raw de Galán en `data/raw/historico_carlos_fernando_galan.csv`.

## Estructura

```
.streamlit/config.toml          # tema claro fijo + ajustes de servidor para deploy
Dockerfile / compose.yaml       # despliegue local o VPS (Fase 6)
requirements.txt                # entorno completo (Cloud + notebooks)
requirements-docker.txt         # solo runtime del tablero (imagen Docker)
run_dashboard.ps1               # arranque local en Windows
data/raw/1980_1988/             # 16 CSV raw 1980-1988 (livianos, se conservan)
data/raw/2021_2026/             # 4 CSV limpios: 80_88, 22_congreso, 23_territoriales*, 26_congreso
                                # *23_territoriales (103 MB) se ignora en git por límite de GitHub
data/raw/geodata/               # departamentos, localidades, upz (.shp) para joins y mapas
data/raw/historico_carlos_fernando_galan.csv
data/processed/                 # 2 parquet canónicos + agg_* + geo/*.geojson (<4 MB en total)
notebooks/01_limpieza.ipynb         # genera data/processed
notebooks/02_eda_geoespacial.ipynb  # EDA nacional + distrital
notebooks/03_clustering.ipynb       # HDBSCAN + K-means
src/dashboard_electoral.py      # tablero Streamlit (Nacional + Distrital)
reports/galan.pbix              # Power BI, se versiona (0.9 MB)
```


