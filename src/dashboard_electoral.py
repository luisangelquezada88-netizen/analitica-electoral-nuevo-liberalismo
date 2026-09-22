"""Dashboard Electoral final — Nuevo Liberalismo + Carlos Fernando Galán.
Versión optimizada para despliegue:
- Parquets con categorical/int32 (24 MB vs 210 MB), HDBSCAN + nk precomputados.
- GeoJSON ligero pre-simplificado en data/processed/geo/ (sin geopandas/folium en runtime).
- Nacional: Plotly choropleth con geojson cacheado + uirevision.
- Distrital: PyDeck nativo (WebGL) en vez de Folium/Leaflet.
- Sin copias de 200 MB: filtros por máscara, groupbys observed=True, KMeans n_init=10.
"""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from unidecode import unidecode

HDB_PAL = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF6B6B",
           "#4ECDC4", "#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#EF476F", "#1B998B",
           "#2D3047", "#FF9A00", "#9D4EDD", "#3A86FF", "#FB5607", "#8338EC", "#00BFFF"]

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

HDB_RGB = [_hex_to_rgb(h) + [130] for h in HDB_PAL]
# Centroides tipo MarkerCluster: mismo color pero sólido (más opaco que puntos individuales)
HDB_RGB_SOLID = [_hex_to_rgb(h) + [205] for h in HDB_PAL]

def norm_key(s):
    if pd.isna(s):
        return ""
    return unidecode(str(s).lower().strip())

# ── Config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Nuevo Liberalismo — Analítica Electoral", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GEO_LIGHT = PROC / "geo"

COLORS = {"bg":"#FDF8F8","card":"#FFFFFF","text":"#111827","muted":"#374151","grid":"#E5E7EB",
          "red":"#B42318","red_soft":"#E11D48","red_dark":"#7C2D12","teal":"#0F766E","slate":"#1F2937"}

st.markdown(f"""
<style>
.stApp{{background-color:{COLORS["bg"]}}}
.block-container{{padding-top:1rem;max-width:96rem}}
h1,h2,h3,[data-testid="stSubheader"] h3,[data-testid="stHeader"]{{color:{COLORS["text"]} !important;letter-spacing:-0.02em;opacity:1 !important}}
.page-title{{color:{COLORS["text"]} !important;font-size:1.6rem;font-weight:800;letter-spacing:-0.02em;margin:0.2rem 0 0.6rem 0}}
.kpi{{background:{COLORS["card"]};border:1px solid {COLORS["grid"]};border-radius:14px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.04)}}
.kpi-label{{color:{COLORS["muted"]};font-size:0.82rem}}.kpi-value{{color:{COLORS["text"]};font-size:1.55rem;font-weight:700}} .kpi-sub{{color:{COLORS["muted"]};font-size:0.78rem}}
.nav-title{{color:{COLORS["text"]} !important;font-size:1.15rem;font-weight:800;letter-spacing:0.02em;margin:0.4rem 0 0.2rem 0;text-align:center}}
[data-testid="stButton"] button{{font-weight:700 !important;border-radius:12px !important}}
[data-testid="stButton"] button[kind="primary"]{{background-color:{COLORS["red"]} !important;border-color:{COLORS["red_dark"]} !important;color:white !important;font-size:1.1rem !important;padding:0.7rem 1rem !important;box-shadow:0 4px 12px rgba(180,35,24,0.35)}}
[data-testid="stButton"] button[kind="secondary"]{{font-size:1.0rem !important;padding:0.6rem 1rem !important}}
/* Toolbar (pantalla completa/descarga) siempre visible: por defecto Streamlit la oculta hasta hover */
[data-testid="stElementToolbar"]{{opacity:1 !important;visibility:visible !important;transform:none !important}}
div[data-testid="stPydeckChart"] [data-testid="stElementToolbar"]{{opacity:1 !important;visibility:visible !important}}
</style>
""", unsafe_allow_html=True)

def fmt(x):
    return f"{int(round(x)):,}".replace(",", ".") if pd.notna(x) else "-"

@st.cache_data(show_spinner=False)
def load_nl():
    df = pd.read_parquet(PROC / "nuevo_liberalismo.parquet")
    # dtypes ya optimizados offline (category/int16/int32); aseguramos ano numérico
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_gal():
    df = pd.read_parquet(PROC / "carlos_fernando_galan.parquet")
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    if "corporacion_norm" not in df.columns:
        df["corporacion_norm"] = df["corporacion"].astype(str).str.lower()
    # compat: si el parquet es anterior a F1, crea nk/hdb al vuelo (una sola vez por caché)
    if "nk_upz" not in df.columns:
        df["nk_upz"] = df["upz"].astype(str).map(norm_key)
    if "nk_loc" not in df.columns:
        df["nk_loc"] = df["localidad"].astype(str).map(norm_key)
    return df

@st.cache_resource(show_spinner=False)
def load_dept_geo():
    """GeoJSON ligero pre-simplificado. Fallback a SHP solo si falta el generado."""
    p = GEO_LIGHT / "deptos_light.geojson"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    import geopandas as gpd  # fallback dev
    shp = ROOT / "data" / "raw" / "geodata" / "departamentos" / "MGN_ADM_DPTO_POLITICO.shp"
    g = gpd.read_file(str(shp)).to_crs(4326)
    g["dpto_ccdgo"] = g["dpto_ccdgo"].astype(str).str.zfill(2)
    g["geometry"] = g["geometry"].simplify(0.008, preserve_topology=True)
    return json.loads(g.to_json())

@st.cache_resource(show_spinner=False)
def load_dis_geo():
    p_loc = GEO_LIGHT / "localidades_light.geojson"
    p_upz = GEO_LIGHT / "upz_light.geojson"
    if p_loc.exists() and p_upz.exists():
        return (json.loads(p_loc.read_text(encoding="utf-8")),
                json.loads(p_upz.read_text(encoding="utf-8")))
    import geopandas as gpd
    loc = gpd.read_file(str(ROOT / "data" / "raw" / "geodata" / "localidades" / "Loca.shp")).to_crs(4326)
    upz = gpd.read_file(str(ROOT / "data" / "raw" / "geodata" / "upz" / "upz-bogota.shp")).to_crs(4326)
    return (json.loads(loc.to_json()), json.loads(upz.to_json()))

try:
    nl = load_nl()
    gal = load_gal()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# HDBSCAN precomputado: agregación por puesto estable (usa full gal, no filtros)
@st.cache_data(show_spinner=False)
def _puesto_hdb_cached():
    if "hdb" not in gal.columns:
        return pd.DataFrame(columns=["lugar", "latitud", "longitud", "votos", "hdb"])
    gh = gal[["lugar", "localidad", "upz", "ano", "latitud", "longitud", "votacion", "hdb"]].dropna(subset=["latitud", "longitud"]).copy()
    moda = gh.groupby(["lugar", "latitud", "longitud"], observed=True)["hdb"].apply(
        lambda s: s.mode().iloc[0] if not s.mode().empty else -1).reset_index()
    ph = gh.groupby(["lugar", "latitud", "longitud"], observed=True, as_index=False).agg(
        votos=("votacion", "sum"), localidad=("localidad", "first"),
        upz=("upz", "first"), anos=("ano", "nunique")).merge(moda, on=["lugar", "latitud", "longitud"], how="left")
    ph["hdb"] = ph["hdb"].fillna(-1).astype(int)
    return ph

@st.cache_data(show_spinner=False)
def _cluster_totals():
    """Totales por clúster HDBSCAN (excluye ruido y clústeres de 1 puesto, como el mapa)."""
    clu = _puesto_hdb_cached()
    clu = clu[clu["hdb"] != -1]
    if clu.empty:
        return pd.DataFrame(columns=["cluster", "hdb", "puestos", "votos_totales"])
    t = clu.groupby("hdb").agg(puestos=("votos", "size"), votos_totales=("votos", "sum")).reset_index()
    t = t[t["puestos"] >= 2].sort_values("votos_totales", ascending=False).reset_index(drop=True)
    t["cluster"] = t["hdb"].astype(int).map(lambda c: f"Cluster {int(c)}")
    return t[["cluster", "hdb", "puestos", "votos_totales"]]

# ── PyDeck: constructor del mapa distrital (objeto fresco por llamada) ──
def _fill_for(v, vmax):
    if pd.isna(v) or v == 0 or not vmax:
        return [240, 240, 240, 140]
    nn = v / vmax
    if nn > 0.7:
        return [124, 45, 18, 180]
    if nn > 0.4:
        return [180, 35, 24, 180]
    return [252, 165, 165, 180]

def _color_puesto(v, pmax):
    q = (v / pmax) if pmax else 0
    if q > 0.75:
        return [255, 0, 0, 185]
    if q > 0.5:
        return [255, 153, 0, 185]
    if q > 0.25:
        return [0, 170, 0, 185]
    return [0, 102, 204, 185]

def _build_pydeck(corp_t, loc_t, upz_t, ano_t, show_upz, show_loc, show_puestos, show_hdb, show_heat):
    gsel = gal
    if corp_t:
        gsel = gsel[gsel["corporacion"].isin(list(corp_t))]
    if loc_t:
        gsel = gsel[gsel["localidad"].isin(list(loc_t))]
    if upz_t:
        gsel = gsel[gsel["upz"].isin(list(upz_t))]
    if ano_t:
        gsel = gsel[gsel["ano"].isin(list(ano_t))]
    if gsel.empty:
        raise ValueError("sin datos con esos filtros")
    lat_c = float(gsel["latitud"].dropna().mean())
    lon_c = float(gsel["longitud"].dropna().mean())
    if pd.isna(lat_c) or pd.isna(lon_c):
        raise ValueError("sin coords")
    loc_geo, upz_geo = load_dis_geo()
    layers = []

    if show_upz:
        ua = gsel.groupby("upz", observed=True).agg(votos=("votacion", "sum")).reset_index()
        ua["nk"] = ua["upz"].astype(str).map(norm_key)
        vmap = dict(zip(ua["nk"], ua["votos"]))
        vmax = float(ua["votos"].max()) if len(ua) else 1.0
        _locmap = gal.dropna(subset=["upz"]).groupby("upz", observed=True)["localidad"].agg(
            lambda s: str(s.mode().iloc[0]) if not s.mode().empty else "").to_dict()
        _locmap_nk = {norm_key(k): v for k, v in _locmap.items()}
        gj = copy.deepcopy(upz_geo)
        for f in gj["features"]:
            nm = f["properties"].get("nombre", "")
            v = vmap.get(norm_key(nm), 0)
            v = int(v) if pd.notna(v) else 0
            f["properties"]["votos"] = v
            f["properties"]["votos_fmt"] = fmt(v)
            f["properties"]["titulo"] = f"UPZ {nm}" if nm else "UPZ"
            f["properties"]["sub"] = f"{_locmap_nk.get(norm_key(nm), 'Bogotá')} · Bogotá"
            f["properties"]["fill"] = _fill_for(v, vmax)
        layers.append(pdk.Layer("GeoJsonLayer", data=gj, id="upz",
                                filled=True, stroked=True, get_fill_color="properties.fill",
                                get_line_color="[60,60,60,180]", line_width_min_pixels=1,
                                pickable=True, auto_highlight=True))

    if show_loc:
        la = gsel.groupby("localidad", observed=True).agg(votos=("votacion", "sum")).reset_index()
        la["nk"] = la["localidad"].astype(str).map(norm_key).replace({"la candelaria": "candelaria"})
        vmap2 = dict(zip(la["nk"], la["votos"]))
        vmax2 = float(la["votos"].max()) if len(la) else 1.0
        gj2 = copy.deepcopy(loc_geo)
        for f in gj2["features"]:
            nm = f["properties"].get("LocNombre", "")
            v = vmap2.get(norm_key(nm), float("nan"))
            v = int(v) if pd.notna(v) else 0
            f["properties"]["votos"] = v
            f["properties"]["votos_fmt"] = fmt(v)
            f["properties"]["titulo"] = f"Localidad {nm}" if nm else "Localidad"
            f["properties"]["sub"] = "Bogotá · Carlos F. Galán"
            f["properties"]["fill"] = _fill_for(v if pd.notna(v) else 0, vmax2)
        layers.append(pdk.Layer("GeoJsonLayer", data=gj2, id="loc",
                                filled=True, stroked=True, get_fill_color="properties.fill",
                                get_line_color="[17,17,17,220]", line_width_min_pixels=2,
                                pickable=True, auto_highlight=True))

    puestos = gsel.groupby(["lugar", "localidad", "upz", "latitud", "longitud"], observed=True).agg(
        votos=("votacion", "sum"), anos=("ano", "nunique")).reset_index().dropna(subset=["latitud", "longitud"])
    if len(puestos) > 600:
        puestos = puestos.nlargest(600, "votos")
    pmax = float(puestos["votos"].max()) if len(puestos) else 1.0

    if show_puestos and len(puestos):
        puestos = puestos.copy()
        puestos["color"] = [_color_puesto(v, pmax) for v in puestos["votos"]]
        puestos["radio"] = [120 + (v / pmax) * 480 for v in puestos["votos"]]
        puestos["titulo"] = puestos["lugar"].astype(str)
        puestos["sub"] = [f"{loc} · UPZ {upz} · {int(a)} {'elección' if int(a) == 1 else 'elecciones'}"
                          for loc, upz, a in zip(puestos["localidad"].astype(str),
                                                 puestos["upz"].astype(str), puestos["anos"])]
        puestos["votos_fmt"] = puestos["votos"].map(fmt)
        layers.append(pdk.Layer("ScatterplotLayer", data=puestos, id="puestos",
                                get_position=["longitud", "latitud"], get_fill_color="color",
                                get_radius="radio", radius_min_pixels=4, radius_max_pixels=18,
                                stroked=True, get_line_color="[0,0,0,200]", line_width_min_pixels=1,
                                pickable=True, auto_highlight=True))

    if show_hdb:
        # MarkerCluster liviano sin JS extra y SIN TextLayer:
        # el TextLayer (48 etiquetas blancas opacas) era la mancha que blanqueaba el mapa
        # y al hacer zoom out se apilaba en una franja de números/letras sobre Bogotá.
        # - Vista agrupada (default): SOLO 48 badges coloreados (el tamaño codifica el total).
        # - Totales legibles: tooltip al hover + tabla "Totales por clúster" clicable bajo el mapa.
        # - Clic en un badge: expande ese clúster (muestra sus puestos) y atenúa el resto.
        clu = _puesto_hdb_cached()
        clu = clu[clu["hdb"] != -1]
        if len(clu):
            _exp = st.session_state.get("expanded_hdb", None)
            try:
                _exp = int(_exp) if _exp is not None else None
            except Exception:
                _exp = None
            vmax = float(clu["votos"].max()) if len(clu) else 1.0
            import pandas as pd_  # local alias
            # Badges siempre (vista agrupada): 48 filas, radios contenidos para no tapar el mapa
            _cents_src = clu.groupby("hdb").agg(latitud=("latitud", "mean"), longitud=("longitud", "mean"),
                                                 votos=("votos", "sum"), n=("votos", "size")).reset_index()
            _cents_src = _cents_src[_cents_src["n"] >= 2].copy()
            if len(_cents_src):
                _cents_src["cid"] = _cents_src["hdb"].astype(int)
                _cents_src["color"] = [HDB_RGB_SOLID[int(c) % len(HDB_RGB_SOLID)] for c in _cents_src["cid"]]
                _tmax = float(_cents_src["votos"].max()) if len(_cents_src) else 1.0
                # Badges compactos: 200-520m + píxeles 16-24 (antes 380-1000m y 22-34 -> velo blanco)
                _cents_src["radio"] = [200 + (float(v) / _tmax) ** 0.5 * 320 for v in _cents_src["votos"]]
                _cents_src["nombre"] = _cents_src["cid"].map(lambda c: f"Cluster {int(c)}")
                _cents_src["lugar"] = _cents_src["nombre"]
                _cents_src["titulo"] = [f"Clúster {int(c)} · {int(n)} puestos"
                                        for c, n in zip(_cents_src["cid"], _cents_src["n"])]
                _cents_src["sub"] = "Clic para expandir · Total del clúster"
                _cents_src["votos_fmt"] = _cents_src["votos"].map(fmt)
                if _exp is not None and _exp in set(_cents_src["cid"].tolist()):
                    # Expandido: otros badges atenuados, el activo resaltado
                    _cents_show = _cents_src.copy()
                    _cents_show["color"] = [
                        c if int(cid) == _exp else [c[0], c[1], c[2], 90]
                        for cid, c in zip(_cents_show["cid"], _cents_show["color"])]
                    layers.append(pdk.Layer("ScatterplotLayer", data=_cents_show, id="hdb-badges",
                                            get_position=["longitud", "latitud"], get_fill_color="color",
                                            get_radius="radio", radius_min_pixels=18, radius_max_pixels=26,
                                            stroked=True, get_line_color="[255,255,255,255]", line_width_min_pixels=3,
                                            pickable=True, auto_highlight=True))
                    # Puestos del clúster expandido (estilo puesto, no velo): pocos puntos, bien visibles
                    _sub = clu[clu["hdb"] == _exp].copy()
                    _sub["color"] = [HDB_RGB_SOLID[int(_exp) % len(HDB_RGB_SOLID)]] * len(_sub)
                    _sub["radio"] = [110 + (float(v) / vmax) * 240 for v in _sub["votos"]]
                    _sub["titulo"] = _sub["lugar"].astype(str)
                    _sub["sub"] = [f"Clúster {int(_exp)} · {loc} · UPZ {upz}"
                                   for loc, upz in zip(_sub["localidad"].astype(str), _sub["upz"].astype(str))]
                    _sub["votos_fmt"] = _sub["votos"].map(fmt)
                    layers.append(pdk.Layer("ScatterplotLayer", data=_sub, id="hdb-expanded",
                                            get_position=["longitud", "latitud"], get_fill_color="color",
                                            get_radius="radio", radius_min_pixels=5, radius_max_pixels=16,
                                            stroked=True, get_line_color="[255,255,255,230]", line_width_min_pixels=2,
                                            pickable=True, auto_highlight=True))
                else:
                    # Badge distintivo: más grande que un puesto + halo blanco grueso (estilo divIcon Folium).
                    # Los puestos llevan borde oscuro fino; imposible confundirlos.
                    layers.append(pdk.Layer("ScatterplotLayer", data=_cents_src, id="hdb-badges",
                                            get_position=["longitud", "latitud"], get_fill_color="color",
                                            get_radius="radio", radius_min_pixels=18, radius_max_pixels=26,
                                            stroked=True, get_line_color="[255,255,255,255]", line_width_min_pixels=3,
                                            pickable=True, auto_highlight=True))

    if show_heat and len(puestos):
        layers.append(pdk.Layer("HeatmapLayer", data=puestos, id="heat",
                                get_position=["longitud", "latitud"], get_weight="votos",
                                radius_pixels=60, intensity=1, threshold=0.05,
                                color_range=[[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]))

    if not layers:
        raise ValueError("activa al menos una capa")
    view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=10.5, pitch=0)
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                    tooltip={"html": "<b>{titulo}</b><br/><span>{sub}</span><br/>Votos: <b>{votos_fmt}</b>"})
    return deck

@st.fragment
def _frag_mapa_pydeck(corp_t, loc_t, upz_t, ano_t, show_upz, show_loc, show_puestos, show_hdb, show_heat):
    """Mapa PyDeck: solo se re-ejecuta si cambian filtros o capas."""
    st.markdown("**Visor geográfico distrital**")
    try:
        with st.spinner("Construyendo mapa distrital…"):
            deck = _build_pydeck(corp_t, loc_t, upz_t, ano_t, show_upz, show_loc, show_puestos, show_hdb, show_heat)
        # La key incluye el clúster expandido: fuerza remontaje limpio y purga la selección
        # vieja (evita el loop expandir->rerun->misma selección->replegar->pantalla blanca).
        _exp_key = st.session_state.get("expanded_hdb", None)
        ev = st.pydeck_chart(deck, key=f"mapa_dis_{st.session_state.nonce_dis}_{_exp_key}_{st.session_state._up_tick}",
                             height=520, use_container_width=True,
                             on_select="rerun", selection_mode="single-object")
        objs = ((ev or {}).get("selection", {}) or {}).get("objects", {}) or {}
        cluster_pick, map_pick = None, None
        try:
            for _lid, _items in objs.items():
                if not _items:
                    continue
                props = _items[0] or {}
                if _lid in ("hdb-badges", "hdb-expanded"):
                    _cid = props.get("cid")
                    if _cid is None:
                        _nm = str(props.get("nombre") or props.get("lugar") or "")
                        if _nm.lower().startswith("cluster"):
                            try:
                                _cid = int("".join(ch for ch in _nm if ch.isdigit()) or -1)
                            except Exception:
                                _cid = None
                    if _cid is not None and int(_cid) >= 0:
                        cluster_pick = int(_cid)
                else:
                    _p = props.get("upz") or props.get("nombre") or props.get("LocNombre") or props.get("lugar")
                    if _p:
                        map_pick = str(_p)
                if cluster_pick is not None and map_pick is not None:
                    break
        except Exception:
            pass
        if cluster_pick is not None:
            # Solo expandir ante selección NUEVA. El repliegue es por clic en vacío
            # o cambio de filtros (ver abajo): así la selección persistente no dispara
            # un toggle infinito (pantalla blanca).
            _cur = st.session_state.get("expanded_hdb", None)
            _cur = int(_cur) if _cur is not None else None
            _last = st.session_state.get("_last_hdb_pick", None)
            _last = int(_last) if _last is not None else None
            if cluster_pick != _cur and cluster_pick != _last:
                st.session_state.expanded_hdb = cluster_pick
                st.session_state._last_hdb_pick = cluster_pick
                st.session_state._hdb_just_expanded = True
                st.rerun(scope="fragment")
            # si es la misma selección persistente, ignorar (no rerun)
        elif map_pick:
            # Filtro UPZ/puesto con fuente: ignora la selección rancia que dejó el K-means.
            _ucur = st.session_state.sel_upz
            _usrc = st.session_state.get("_sel_upz_src")
            _ulast = st.session_state.get("_last_pdk_pick")
            if map_pick == _ucur and _usrc == "pdk":
                st.session_state._last_pdk_pick = map_pick
            elif map_pick == _ulast and _usrc == "kdis":
                pass
            else:
                st.session_state.sel_upz = map_pick
                st.session_state._sel_upz_src = "pdk"
                st.session_state._last_pdk_pick = map_pick
                st.rerun(scope="app")
        else:
            # Sin selección nueva: o es un clic en zona vacía (deselect) o el montaje
            # fresco tras expandir. Solo actuar si el mapa está realmente vacío.
            # (Un clic en un puesto ya filtrado deja la selección vieja y cae aquí:
            # en ese caso NO se repliega.)
            _objs_empty = not any(objs.values())
            if _objs_empty:
                if st.session_state.get("_hdb_just_expanded", False):
                    st.session_state._hdb_just_expanded = False
                elif (show_hdb and st.session_state.get("expanded_hdb", None) is not None
                        and st.session_state.get("_last_hdb_pick", None) is not None):
                    st.session_state.expanded_hdb = None
                    st.session_state._last_hdb_pick = None
                    st.rerun(scope="fragment")
        # Tabla de totales por clúster: reemplaza las etiquetas sobre el mapa (que lo blanqueaban).
        # Clic en una fila = expandir ese clúster en el mapa, como el MarkerCluster anterior.
        if show_hdb:
            _ct = _cluster_totals()
            if not _ct.empty:
                with st.expander(f"🏷️ Totales por clúster HDBSCAN ({len(_ct)}) — clic en una fila para expandir", expanded=False):
                    _show = _ct[["cluster", "puestos", "votos_totales"]].copy()
                    _show["votos_totales"] = _show["votos_totales"].map(lambda v: f"{int(v):,}".replace(",", "."))
                    _sel_ct = st.dataframe(_show, width="stretch", height=280,
                                            key=f"hdb_tot_{st.session_state.nonce_dis}",
                                            on_select="rerun", selection_mode="single-row")
                    try:
                        _rows = ((_sel_ct or {}).get("selection", {}) or {}).get("rows", []) or []
                    except Exception:
                        _rows = []
                    if _rows:
                        _cid_pick = int(_ct.iloc[int(_rows[0])]["hdb"])
                        _cur2 = st.session_state.get("expanded_hdb", None)
                        _cur2 = int(_cur2) if _cur2 is not None else None
                        if _cid_pick != _cur2:
                            st.session_state.expanded_hdb = _cid_pick
                            st.session_state._last_hdb_pick = _cid_pick
                            st.session_state._hdb_just_expanded = True
                            st.rerun(scope="fragment")
    except Exception as e:
        st.error(f"Mapa no disponible: {e}")

@st.fragment
def _frag_kmeans_tablas(corp_t, loc_t, upz_t, ano_t, sel_upz):
    """K-means + tablas: se re-ejecuta solo con sus filtros, su Top N o sus clics."""
    gsel = gal
    if corp_t:
        gsel = gsel[gsel["corporacion"].isin(list(corp_t))]
    if loc_t:
        gsel = gsel[gsel["localidad"].isin(list(loc_t))]
    if upz_t:
        gsel = gsel[gsel["upz"].isin(list(upz_t))]
    if ano_t:
        gsel = gsel[gsel["ano"].isin(list(ano_t))]
    upz_agg = gsel.groupby("upz", observed=True).agg(votos=("votacion", "sum"), puestos=("votacion", "count"),
                                      lat=("latitud", "mean"), lon=("longitud", "mean")).reset_index().dropna(subset=["lat", "lon"])
    upz_agg["votos_por_puesto"] = upz_agg["votos"] / upz_agg["puestos"]
    if len(upz_agg) >= 4:
        feat = upz_agg[["votos", "puestos", "votos_por_puesto"]].copy()
        feat["log_votos"] = np.log1p(feat["votos"])
        Xup = StandardScaler().fit_transform(feat[["log_votos", "puestos", "votos_por_puesto"]])
        km_up = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Xup)
        upz_agg["cluster"] = km_up.labels_.astype(str)
    else:
        upz_agg["cluster"] = "0"
    _dt1, _dt2 = st.columns([3, 1])
    with _dt2:
        top_n_dis = st.selectbox("Top N", [5, 10, 15, 20, 30, 50], index=1, key="w_dis_topn")
    c1, c2 = st.columns([1, 1])
    with c1:
        figk = px.scatter(upz_agg, x="puestos", y="votos", color="cluster", hover_name="upz",
                          custom_data=["upz"],
                          color_discrete_sequence=[COLORS["red"], COLORS["slate"], COLORS["teal"], "#F59E0B"],
                          title="Clústeres distritales (K-means, K=4)", log_y=True)
        figk.update_traces(marker=dict(size=9, line=dict(width=1, color="black")))
        selk = st.plotly_chart(base_layout(figk, 380), width="stretch",
                               key=f"kmdis_{st.session_state.nonce_dis}_{st.session_state._up_tick}",
                               on_select="rerun", selection_mode="points")
        ptsk = _sel_points(selk)
        _ukup = ptsk[0].get("customdata") if ptsk else None
        if isinstance(_ukup, (list, tuple)):
            _ukup = _ukup[0] if _ukup else None
        _ucur = st.session_state.sel_upz
        _usrc = st.session_state.get("_sel_upz_src")
        _ulast = st.session_state.get("_last_kdis_pick")
        if _ukup:
            if _ukup == _ucur and _usrc == "kdis":
                st.session_state._last_kdis_pick = _ukup
            elif _ukup == _ulast and _usrc == "pdk":
                pass  # selección rancia del mapa: ignorar
            else:
                st.session_state.sel_upz = _ukup
                st.session_state._sel_upz_src = "kdis"
                st.session_state._last_kdis_pick = _ukup
                st.rerun(scope="app")
        elif _usrc == "kdis" and _ucur is not None and _ulast == _ucur:
            # Deselect explícito: soltar y remontar ambos (purga selecciones rancias).
            st.session_state.sel_upz = None
            st.session_state._sel_upz_src = None
            st.session_state._last_pdk_pick = None
            st.session_state._last_kdis_pick = None
            st.session_state._up_tick += 1
            st.rerun(scope="app")
        upz_show = upz_agg.copy()
        if st.session_state.sel_upz:
            q = norm_key(st.session_state.sel_upz)
            mask = upz_show["upz"].fillna("").astype(str).apply(norm_key).str.contains(q, na=False)
            if mask.any():
                upz_show = upz_show[mask]
    with c2:
        puestos_tbl = gsel.groupby(["lugar", "localidad", "upz"], observed=True).agg(votos=("votacion", "sum"), anos=("ano", "nunique"), registros=("votacion", "size")).sort_values("votos", ascending=False).head(top_n_dis * 2)
        if st.session_state.sel_upz and not puestos_tbl.empty:
            q = norm_key(st.session_state.sel_upz)
            lvl_upz = puestos_tbl.index.get_level_values("upz").astype(str).map(norm_key)
            lvl_loc = puestos_tbl.index.get_level_values("localidad").astype(str).map(norm_key)
            mask = pd.Series(lvl_upz, index=puestos_tbl.index).str.contains(q, na=False) | pd.Series(lvl_loc, index=puestos_tbl.index).str.contains(q, na=False)
            if mask.any():
                puestos_tbl = puestos_tbl[mask.values]
        st.markdown("**Puestos de votación**")
        st.dataframe(puestos_tbl, width="stretch", height=380)

# Opciones de segmentación cacheadas: evitan re-escanear 1,1M filas en cada rerun
@st.cache_data(show_spinner=False)
def _nac_opts():
    cand = nl.groupby("candidato", observed=True)["votos"].sum().sort_values(ascending=False).head(60).index.tolist()
    if "lista" not in cand:
        cand = ["lista"] + cand
    return (cand, sorted(nl["partido"].dropna().unique().tolist()),
            sorted(nl["corporacion"].dropna().unique().tolist()),
            sorted(nl["departamento"].dropna().unique().tolist()),
            sorted(nl["ano"].dropna().unique().tolist()),
            sorted(gal["corporacion"].dropna().unique().tolist()),
            sorted(gal["localidad"].dropna().unique().tolist()),
            sorted(gal["upz"].dropna().unique().tolist()),
            sorted(gal["ano"].dropna().unique().tolist()))

@st.cache_data(show_spinner=False)
def _mun_opts(dept_key):
    base = nl if not dept_key else nl[nl["departamento"].isin(list(dept_key))]
    return base.groupby("municipio", observed=True)["votos"].sum().nlargest(80).index.sort_values().tolist()

cand_opts, part_opts, corp_opts, dept_opts, nac_anos, dcorp_opts, dloc_opts, dupz_opts, dano_opts = _nac_opts()

def _sel_points(sel):
    """Puntos de una selección Plotly (lista vacía si no hay selección)."""
    try:
        return (sel or {}).get("selection", {}).get("points") or []
    except Exception:
        return []

# Estado para selección interactiva
for _k in ("sel_depto_map", "sel_upz", "prev_pagina", "expanded_hdb", "_last_hdb_pick",
             "_hdb_just_expanded", "_last_dis_sig", "_sel_src", "_last_map_pick", "_last_km_pick",
             "_sel_upz_src", "_last_pdk_pick", "_last_kdis_pick"):
    if _k not in st.session_state:
        st.session_state[_k] = None
for _k in ("nonce_nac", "nonce_dis", "_sel_tick", "_up_tick"):
    if _k not in st.session_state:
        st.session_state[_k] = 0

def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def base_layout(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Arial", size=13), margin=dict(l=20, r=20, t=50, b=20),
        title=dict(font=dict(size=15, color=COLORS["text"]), x=0, xanchor="left"), uirevision="app")
    fig.update_xaxes(gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"], size=12), title_font=dict(color=COLORS["text"], size=13))
    fig.update_yaxes(gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"], size=12), title_font=dict(color=COLORS["text"], size=13))
    return fig

if "pagina" not in st.session_state or st.session_state.pagina not in ("Nacional", "Distrital"):
    st.session_state.pagina = "Nacional"
st.markdown("<div class='nav-title'>Selecciona la vista del tablero</div>", unsafe_allow_html=True)
nav_a, nav_b = st.columns(2)
with nav_a:
    if st.button("Nacional", key="nav_nac",
                 type="primary" if st.session_state.pagina == "Nacional" else "secondary",
                 width="stretch"):
        st.session_state.pagina = "Nacional"
        st.rerun()
with nav_b:
    if st.button("Distrital", key="nav_dis",
                 type="primary" if st.session_state.pagina == "Distrital" else "secondary",
                 width="stretch"):
        st.session_state.pagina = "Distrital"
        st.rerun()
pagina = st.session_state.pagina

if st.session_state.prev_pagina != pagina:
    st.session_state.sel_depto_map = None
    st.session_state.sel_upz = None
    st.session_state.expanded_hdb = None
    st.session_state._last_hdb_pick = None
    st.session_state._hdb_just_expanded = False
    st.session_state._sel_src = None
    st.session_state._last_map_pick = None
    st.session_state._last_km_pick = None
    st.session_state._sel_upz_src = None
    st.session_state._last_pdk_pick = None
    st.session_state._last_kdis_pick = None
    st.session_state.prev_pagina = pagina

# ── NACIONAL ───────────────────────────────────────────────────────
if pagina == "Nacional":
    st.markdown('<div class="page-title">Nacional — Nuevo Liberalismo</div>', unsafe_allow_html=True)

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        sel_corp = st.multiselect("Corporación", corp_opts, placeholder="Todas", key="w_nac_corp",
                                  format_func=lambda x: str(x).title())
    with n2:
        sel_part = st.multiselect("Coalición", part_opts, placeholder="Todas", key="w_nac_part")
    with n3:
        sel_dept = st.multiselect("Departamento", dept_opts, placeholder="Todos", key="w_nac_dept")
    with n4:
        sel_nano = st.segmented_control("Año", nac_anos, selection_mode="multi", key="w_nac_ano",
                                        format_func=lambda x: str(int(x))) or []
    n5, n6, n7, n8 = st.columns(4)
    with n5:
        sel_cand = st.multiselect("Candidato", cand_opts, placeholder="Todos", key="w_nac_cand")
    with n6:
        mun_opts = _mun_opts(tuple(sorted(sel_dept)))
        sel_mun = st.multiselect("Municipio", mun_opts, placeholder="Todos", key="w_nac_mun")
    with n7:
        top_n = st.selectbox("Top N", [5, 10, 15, 20, 30, 50], index=1, key="w_nac_topn")
    with n8:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        def _clear_nac():
            for _k in ("w_nac_corp", "w_nac_part", "w_nac_dept", "w_nac_ano", "w_nac_cand", "w_nac_mun"):
                st.session_state[_k] = []
            st.session_state.sel_depto_map = None
            st.session_state._sel_src = None
            st.session_state._last_map_pick = None
            st.session_state._last_km_pick = None
            st.session_state.nonce_nac += 1
        st.button("Borrar filtros", key="w_nac_clear", on_click=_clear_nac, width="stretch")

    # Slicers → base con una sola máscara (sin copiar 1.1M filas varias veces)
    _mask = pd.Series(True, index=nl.index)
    if sel_cand:
        _mask &= nl["candidato"].isin(sel_cand)
    if sel_part:
        _mask &= nl["partido"].isin(sel_part)
    if sel_corp:
        _mask &= nl["corporacion"].isin(sel_corp)
    if sel_dept:
        _mask &= nl["departamento"].isin(sel_dept)
    if sel_nano:
        _mask &= nl["ano"].isin(sel_nano)
    if sel_mun:
        _mask &= nl["municipio"].isin(sel_mun)
    flt = nl[_mask]

    if flt.empty:
        st.warning("Filtros sin resultados.")
        st.stop()

    def _flt_sel():
        out = flt
        if st.session_state.sel_depto_map:
            out = out[out["cod_dpto_geo"] == st.session_state.sel_depto_map]
        return out

    kpi_slot = st.container()
    # Fila 1: mapa full-width protagonista; fila 2: K-means full-width debajo.
    # (Containers apilados: mismo código, sin reindentar; el letterbox por aspecto
    # de Colombia se compensa con más alto + zoom óptico leve.)
    colA = st.container()
    colB = st.container()
    with colA:
        dept_agg = flt.groupby("cod_dpto_geo", observed=True).agg(votos=("votos", "sum")).reset_index()
        dept_geo = load_dept_geo()
        # tabla base alineada al geojson (incluye deptos con 0 votos)
        _ids, _names = [], []
        for f in dept_geo["features"]:
            _ids.append(f["properties"].get("dpto_ccdgo"))
            _names.append(f["properties"].get("dpto_cnmbr"))
        gbase = pd.DataFrame({"dpto_ccdgo": _ids, "dpto_cnmbr": _names})
        gbase = gbase.merge(dept_agg, left_on="dpto_ccdgo", right_on="cod_dpto_geo", how="left")
        gbase["votos"] = gbase["votos"].fillna(0)
        gbase = gbase[gbase["dpto_ccdgo"] != "88"]
        gbase["log_votos"] = np.log1p(gbase["votos"])
        fig = px.choropleth(gbase, geojson=dept_geo, locations="dpto_ccdgo", featureidkey="properties.dpto_ccdgo",
                            color="log_votos", hover_name="dpto_cnmbr",
                            hover_data={"log_votos": False, "votos": True, "dpto_ccdgo": False},
                            color_continuous_scale=[[0, "#FFF5F5"], [0.35, "#FCA5A5"], [0.65, COLORS["red"]], [1, COLORS["red_dark"]]],
                            title="Votos por departamento")
        fig.update_geos(fitbounds="locations", visible=False, showframe=False,
                        showcoastlines=False, showland=False, showocean=False,
                        bgcolor="rgba(0,0,0,0)",
                        projection=dict(scale=1.1, minscale=1.1, maxscale=1.1))
        fig.update_layout(height=850, margin=dict(l=0, r=0, t=40, b=0), dragmode=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", uirevision="nacional",
            coloraxis_colorbar=dict(title="log(votos)", bgcolor="rgba(0,0,0,0)",
                tickfont=dict(color=COLORS["text"]), title_font=dict(color=COLORS["text"])))
        fig.update_layout(font=dict(color=COLORS["text"]))
        sel = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
            key=f"map_nac_{st.session_state.nonce_nac}_{st.session_state._sel_tick}",
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False, "showTips": False})
        st.caption("San Andrés y Providencia se excluye del croquis por escala; sus votos siguen en tarjetas, K-means y tablas.")
        # Selección con fuente: evita el ping-pong mapa<->K-means (titileo) donde el mapa
        # fijaba el depto y el K-means vacío lo borraba en el mismo ciclo. La cascada
        # (tarjetas, K-means, barras, treemap y tablas) ya consume sel_depto_map.
        pts = _sel_points(sel)
        _mpick = pts[0].get("location") if pts else None
        _mcur = st.session_state.sel_depto_map
        _msrc = st.session_state.get("_sel_src")
        _mlast = st.session_state.get("_last_map_pick")
        if _mpick:
            if _mpick == _mcur and _msrc == "map":
                st.session_state._last_map_pick = _mpick
            elif _mpick == _mlast and _msrc == "km":
                pass  # selección rancia del otro widget: ignorar
            else:
                st.session_state.sel_depto_map = _mpick
                st.session_state._sel_src = "map"
                st.session_state._last_map_pick = _mpick
                st.rerun()
        elif _msrc == "map" and _mcur is not None and _mlast == _mcur:
            # Deselect explícito en el mapa: soltar el filtro (remonta ambos gráficos).
            st.session_state.sel_depto_map = None
            st.session_state._sel_src = None
            st.session_state._last_map_pick = None
            st.session_state._last_km_pick = None
            st.session_state._sel_tick += 1
            st.rerun()
        # Montaje fresco (sin selección): ignorar, no es un deselect.
        flt_nac_inter = _flt_sel()

    with kpi_slot:
        base_kpi = flt_nac_inter if not flt_nac_inter.empty else flt
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Votos filtrados", fmt(base_kpi["votos"].sum()))
        with c2:
            kpi("Departamentos", fmt(base_kpi["departamento"].nunique()), f"de {nl['departamento'].nunique()}")
        with c3:
            kpi("Municipios", fmt(base_kpi["municipio"].nunique()))
        with c4:
            kpi("Candidatos", fmt(base_kpi["candidato"].nunique()))

    with colB:
        dept_try = flt_nac_inter.groupby("departamento", observed=True).agg(votos=("votos", "sum"), municipios=("municipio", "nunique"), cod=("cod_dpto_geo", "first")).reset_index()
        dept_try = dept_try[~dept_try["departamento"].isin(["bogota", "consulados", "sin_dato"])].copy()
        _kmeans_base = "cascada" if len(dept_try) >= 4 else "estable"
        dept_k = dept_try if _kmeans_base == "cascada" else flt.groupby("departamento", observed=True).agg(votos=("votos", "sum"), municipios=("municipio", "nunique"), cod=("cod_dpto_geo", "first")).reset_index()
        if _kmeans_base == "estable":
            dept_k = dept_k[~dept_k["departamento"].isin(["bogota", "consulados", "sin_dato"])].copy()
        if len(dept_k) >= 4:
            dept_k["log_votos"] = np.log1p(dept_k["votos"])
            X = StandardScaler().fit_transform(dept_k[["log_votos", "municipios"]])
            km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
            dept_k["cluster"] = km.labels_.astype(str)
            _sel = st.session_state.sel_depto_map
            dept_k["sel"] = (dept_k["cod"] == _sel) if _sel else False
            dept_k["size"] = np.where(dept_k["sel"], 16, 10)
            fig2 = px.scatter(dept_k, x="municipios", y="votos", color="cluster", hover_name="departamento",
                              custom_data=["cod"], log_y=True, size="size",
                              color_discrete_sequence=[COLORS["red"], COLORS["slate"], COLORS["teal"], "#F59E0B"],
                              title="Clústeres Nacionales (K-means, K=4)")
            fig2.update_traces(marker=dict(line=dict(width=1, color="black")))
            sel2 = st.plotly_chart(base_layout(fig2, 420), width="stretch",
                                   key=f"kmnac_{st.session_state.nonce_nac}_{st.session_state._sel_tick}",
                                   on_select="rerun", selection_mode="points")
            pts2 = _sel_points(sel2)
            _kpick = None
            if pts2:
                cd = pts2[0].get("customdata")
                _kpick = (cd[0] if isinstance(cd, (list, tuple)) else cd)
            _kcur = st.session_state.sel_depto_map
            _ksrc = st.session_state.get("_sel_src")
            _klast = st.session_state.get("_last_km_pick")
            if _kpick:
                if _kpick == _kcur and _ksrc == "km":
                    st.session_state._last_km_pick = _kpick
                elif _kpick == _klast and _ksrc == "map":
                    pass  # selección rancia del otro widget: ignorar
                else:
                    st.session_state.sel_depto_map = _kpick
                    st.session_state._sel_src = "km"
                    st.session_state._last_km_pick = _kpick
                    st.rerun()
            elif _ksrc == "km" and _kcur is not None and _klast == _kcur:
                st.session_state.sel_depto_map = None
                st.session_state._sel_src = None
                st.session_state._last_map_pick = None
                st.session_state._last_km_pick = None
                st.session_state._sel_tick += 1
                st.rerun()
        else:
            st.info("Muy pocos departamentos para clusterizar con filtros actuales.")

    r1, r2 = st.columns(2)
    with r1:
        # Lollipop horizontal en log: legible con skew 50:1, sin barras que mienten.
        # La cola fuera del Top N se agrega en "Otros"; el hover muestra totales reales.
        _part = flt_nac_inter.groupby("partido", observed=True)["votos"].sum().sort_values(ascending=False)
        _ntot = len(_part)
        _otr = _part.iloc[top_n:].sum() if _ntot > top_n else 0
        _dfc = _part.head(top_n).reset_index()
        _dfc.columns = ["partido", "votos"]
        if _otr > 0:
            _dfc = pd.concat([_dfc, pd.DataFrame([{"partido": f"Otros ({_ntot - top_n})",
                                                   "votos": int(_otr)}])], ignore_index=True)
        _dfc = _dfc.sort_values("votos", ascending=True).reset_index(drop=True)
        _dfc = _dfc[_dfc["votos"] > 0].reset_index(drop=True)  # log(0) no existe
        if _dfc.empty:
            st.info("Sin votos para la combinación de filtros.")
        else:
            def _short(v):
                v = int(v)
                if v >= 1_000_000:
                    return f"{v / 1_000_000:.1f} M".replace(".", ",")
                if v >= 1_000:
                    return f"{v / 1_000:.0f} mil"
                return fmt(v)

            _dfc["votos_fmt"] = _dfc["votos"].map(fmt)
            _dfc["votos_short"] = _dfc["votos"].map(_short)
            _dfc["color"] = [("#9CA3AF" if str(p).startswith("Otros (") else COLORS["red"])
                             for p in _dfc["partido"]]
            _vmax = float(_dfc["votos"].max()) if len(_dfc) else 1.0
            fig3 = go.Figure()
            for _, _r in _dfc.iterrows():
                fig3.add_shape(type="line", x0=1, x1=float(_r["votos"]), y0=_r["partido"], y1=_r["partido"],
                               line=dict(color="#D1D5DB", width=2), layer="below")
            fig3.add_trace(go.Scatter(x=_dfc["votos"].tolist(), y=_dfc["partido"].tolist(),
                mode="markers+text",
                marker=dict(size=11, color=_dfc["color"].tolist(), line=dict(width=1, color="black")),
                text=_dfc["votos_short"].tolist(), textposition="middle right",
                textfont=dict(size=11, color=COLORS["text"]),
                customdata=_dfc["votos_fmt"].tolist(), cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Votos: %{customdata}<extra></extra>", showlegend=False))
            fig3.update_layout(title="Votos por coalición")
            fig3.update_xaxes(type="log", range=[0, float(np.log10(_vmax * 7.0))], title_text="votos (log)")
            fig3.update_yaxes(title_text="", automargin=True)
            st.plotly_chart(base_layout(fig3, 440), width="stretch")
    with r2:
        if flt_nac_inter.empty:
            st.info("Sin datos para la combinación de filtros.")
        else:
            tc = flt_nac_inter.groupby("corporacion", observed=True)["votos"].sum().reset_index().sort_values("votos", ascending=False)
            figt = px.treemap(tc, path=["corporacion"], values="votos", color="votos",
                              color_continuous_scale=[[0, "#FCA5A5"], [0.6, COLORS["red"]], [1, COLORS["red_dark"]]],
                              title="Votos por corporación")
            figt.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10),
                               font=dict(color=COLORS["text"]), uirevision="nacional",
                               coloraxis_colorbar=dict(title="votos", tickfont=dict(color=COLORS["text"]),
                                                       title_font=dict(color=COLORS["text"])))
            figt.update_traces(texttemplate="%{label}<br>%{value:.2s} (%{percentRoot})",
                               textfont=dict(size=13, color="white"))
            st.plotly_chart(figt, width="stretch")

    t1, t2 = st.columns(2)
    with t1:
        cand = flt_nac_inter.groupby("candidato", observed=True).agg(votos=("votos", "sum"), municipios=("municipio", "nunique"), registros=("votos", "size")).sort_values("votos", ascending=False).head(top_n)
        st.markdown("**Top candidatos**")
        st.dataframe(cand, width="stretch")
    with t2:
        mun = flt_nac_inter.groupby(["municipio", "departamento"], observed=True).agg(votos=("votos", "sum"), candidatos=("candidato", "nunique")).sort_values("votos", ascending=False).head(top_n)
        st.markdown("**Top municipios**")
        st.dataframe(mun, width="stretch")

# ── DISTRITAL ──────────────────────────────────────────────────────
if pagina == "Distrital":
    st.markdown('<div class="page-title">Distrital — Carlos Fernando Galán (Bogotá)</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        sel_dcorp = st.segmented_control("Corporación", dcorp_opts, selection_mode="multi", key="w_dis_corp") or []
    with d2:
        sel_dloc = st.multiselect("Localidad", dloc_opts, placeholder="Todas", key="w_dis_loc")
    with d3:
        sel_dupz = st.multiselect("UPZ", dupz_opts, placeholder="Todas", key="w_dis_upz")
    with d4:
        sel_dano = st.segmented_control("Año", dano_opts, selection_mode="multi", key="w_dis_ano",
                                        format_func=lambda x: str(int(x))) or []
    d5, d6 = st.columns([3, 1])
    with d6:
        def _clear_dis():
            for _k in ("w_dis_corp", "w_dis_loc", "w_dis_upz", "w_dis_ano"):
                st.session_state[_k] = []
            st.session_state.sel_upz = None
            st.session_state.expanded_hdb = None
            st.session_state._last_hdb_pick = None
            st.session_state._hdb_just_expanded = False
            st.session_state._sel_upz_src = None
            st.session_state._last_pdk_pick = None
            st.session_state._last_kdis_pick = None
            st.session_state.nonce_dis += 1
        st.button("Borrar filtros", key="w_dis_clear", on_click=_clear_dis, width="stretch")

    _gmask = pd.Series(True, index=gal.index)
    if sel_dcorp:
        _gmask &= gal["corporacion"].isin(sel_dcorp)
    if sel_dloc:
        _gmask &= gal["localidad"].isin(sel_dloc)
    if sel_dupz:
        _gmask &= gal["upz"].isin(sel_dupz)
    if sel_dano:
        _gmask &= gal["ano"].isin(sel_dano)
    gal_flt = gal[_gmask]
    if gal_flt.empty:
        st.warning("Filtros sin resultados.")
        st.stop()

    dkpi_slot = st.container()
    with dkpi_slot:
        _gk = gal_flt
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Votos Galán", fmt(_gk["votacion"].sum()))
        with c2:
            kpi("Localidades", fmt(_gk["localidad"].nunique()))
        with c3:
            kpi("UPZ", fmt(_gk["upz"].nunique()))
        with c4:
            kpi("Puestos", fmt(_gk["lugar"].nunique()))

    # Capas del mapa (reemplazan al LayerControl de Folium, misma info, WebGL)
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown("**Capas**")
    l1, l2, l3, l4, l5 = st.columns(5)
    with l1:
        sh_upz = st.checkbox("Votos por UPZ", value=False, key="w_dis_sh_upz")
    with l2:
        sh_loc = st.checkbox("Votos por localidad", value=False, key="w_dis_sh_loc")
    with l3:
        sh_pue = st.checkbox("Votos por puesto", value=True, key="w_dis_sh_pue")
    with l4:
        sh_hdb = st.checkbox("HDBSCAN clusters", value=False, key="w_dis_sh_hdb")
    with l5:
        sh_heat = st.checkbox("Mapa de calor", value=False, key="w_dis_sh_heat")

    _sig = (tuple(sorted(sel_dcorp)), tuple(sorted(sel_dloc)), tuple(sorted(sel_dupz)),
            tuple(int(x) for x in sorted(sel_dano)))
    # Cambio de filtros repliega el clúster expandido (el cid viejo puede no existir ya)
    if st.session_state.get("_last_dis_sig", None) != _sig:
        st.session_state._last_dis_sig = _sig
        st.session_state.expanded_hdb = None
        st.session_state._last_hdb_pick = None
        st.session_state._hdb_just_expanded = False
    _frag_mapa_pydeck(*_sig, bool(sh_upz), bool(sh_loc), bool(sh_pue), bool(sh_hdb), bool(sh_heat))

    tra = gal_flt.groupby("ano", observed=True)["votacion"].sum().reset_index().sort_values("ano")
    figd = px.line(tra, x="ano", y="votacion", markers=True, title="Trayectoria anual",
                   labels={"ano": "año"}, color_discrete_sequence=[COLORS["red"]])
    figd.update_traces(line=dict(width=3), marker=dict(size=9, line=dict(width=1, color="black")))
    st.plotly_chart(base_layout(figd, 340), width="stretch")

    _frag_kmeans_tablas(*_sig, st.session_state.sel_upz)
