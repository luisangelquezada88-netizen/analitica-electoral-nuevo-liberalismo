import os
os.environ["SHAPE_ENCODING"] = "UTF-8"
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import st_folium
import hdbscan
import matplotlib
import matplotlib.pyplot as plt
from unidecode import unidecode

def norm_key(s):
    if pd.isna(s):
        return ""
    return unidecode(str(s).lower().strip())

# ── Config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Nuevo Liberalismo — Analítica Electoral", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GEO = ROOT / "data" / "raw" / "geodata"

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
/* Selector de página: botones grandes */
[data-testid="stSegmentedControl"] {{margin:0.2rem 0 0.4rem 0}}
[data-testid="stSegmentedControl"] button {{
  font-size:1.05rem !important;font-weight:800 !important;letter-spacing:0.03em;
  padding:10px 34px !important;border:2px solid {COLORS["red"]} !important;border-radius:12px !important;
  color:{COLORS["red"]} !important;background:{COLORS["card"]} !important;
}}
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {{
  background:{COLORS["red"]} !important;color:white !important;border-color:{COLORS["red_dark"]} !important;
  box-shadow:0 4px 12px rgba(180,35,24,0.35);
}}
[data-testid="stSegmentedControl"] button p {{font-size:1.05rem !important;font-weight:800 !important;margin:0 !important}}
</style>
""", unsafe_allow_html=True)

def fmt(x):
    return f"{int(round(x)):,}".replace(",", ".") if pd.notna(x) else "-"

@st.cache_data(show_spinner=False)
def load_nl():
    df=pd.read_parquet(PROC/"nuevo_liberalismo.parquet")
    df["ano"]=pd.to_numeric(df["ano"],errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_gal():
    df=pd.read_parquet(PROC/"carlos_fernando_galan.parquet")
    df["ano"]=pd.to_numeric(df["ano"],errors="coerce")
    df["corporacion_norm"]=df["corporacion"].str.lower()
    return df

@st.cache_resource(show_spinner=False)
def load_geo():
    dept=gpd.read_file(str(GEO/"departamentos"/"MGN_ADM_DPTO_POLITICO.shp")).to_crs(4326)
    dept["dpto_ccdgo"]=dept["dpto_ccdgo"].astype(str).str.zfill(2)
    # simplify for speed
    dept["geometry"]=dept["geometry"].simplify(0.005)
    loc=gpd.read_file(str(GEO/"localidades"/"Loca.shp")).to_crs(4326)
    upz=gpd.read_file(str(GEO/"upz"/"upz-bogota.shp")).to_crs(4326)
    upz["geometry"]=upz["geometry"].simplify(0.001)
    return dept, loc, upz

try:
    nl=load_nl(); gal=load_gal(); dept_gdf, loc_gdf, upz_gdf = load_geo()
except Exception as e:
    st.error(f"Error cargando datos: {e}"); st.stop()

# ── Filtros ──────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
if st.sidebar.button("🔄 Borrar filtros", use_container_width=True):
    for _k, _v in [("f_cand", []), ("f_part", []), ("f_corp", []), ("f_dept", []), ("f_mun", []), ("f_topn", 10)]:
        st.session_state[_k] = _v
    st.session_state.sel_depto_map = None
    st.session_state.sel_upz = None
    st.rerun()

# Opciones limitadas para candidatos (top 60 por votos para no saturar)
cand_opts = nl.groupby("candidato")["votos"].sum().sort_values(ascending=False).head(60).index.tolist()
if "lista" not in cand_opts: cand_opts = ["lista"]+cand_opts
part_opts = sorted(nl["partido"].dropna().unique().tolist())
corp_opts = sorted(nl["corporacion"].dropna().unique().tolist())
dept_opts = sorted(nl["departamento"].dropna().unique().tolist())

sel_cand = st.sidebar.multiselect("Candidato", cand_opts, placeholder="Todos", key="f_cand")
sel_part = st.sidebar.multiselect("Coalición", part_opts, placeholder="Todas", key="f_part")
sel_corp = st.sidebar.multiselect("Corporación", corp_opts, placeholder="Todas", key="f_corp")
sel_dept = st.sidebar.multiselect("Departamento", dept_opts, placeholder="Todos", key="f_dept")

# Municipio dependiente de departamento
mun_base = nl if not sel_dept else nl[nl["departamento"].isin(sel_dept)]
# limitar a top 80 municipios por votos para no listar 1.1k
mun_opts = mun_base.groupby("municipio")["votos"].sum().sort_values(ascending=False).head(80).index.tolist()
sel_mun = st.sidebar.multiselect("Municipio", sorted(mun_opts), placeholder="Todos", key="f_mun")

top_n = st.sidebar.slider("Top N tablas", 5, 20, 10, key="f_topn")

# Aplicar filtros a NL
flt = nl.copy()
if sel_cand: flt = flt[flt["candidato"].isin(sel_cand)]
if sel_part: flt = flt[flt["partido"].isin(sel_part)]
if sel_corp: flt = flt[flt["corporacion"].str.lower().isin([c.lower() for c in sel_corp])]
if sel_dept: flt = flt[flt["departamento"].isin(sel_dept)]
if sel_mun: flt = flt[flt["municipio"].isin(sel_mun)]

# Gal filtrado solo por corporación (normalizada)
gal_flt = gal.copy()
if sel_corp:
    gal_flt = gal_flt[gal_flt["corporacion_norm"].isin([c.lower() for c in sel_corp])]

if flt.empty:
    st.warning("Filtros sin resultados."); st.stop()

# Estado para selección interactiva
if "sel_depto_map" not in st.session_state: st.session_state.sel_depto_map=None
if "sel_upz" not in st.session_state: st.session_state.sel_upz=None

def kpi(label,value,sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def base_layout(fig,h=380):
    fig.update_layout(template="plotly_white",height=h,paper_bgcolor=COLORS["card"],plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"],family="Arial",size=13),margin=dict(l=20,r=20,t=50,b=20),
        title=dict(font=dict(size=15,color=COLORS["text"]),x=0,xanchor="left"))
    fig.update_xaxes(gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"],size=12), title_font=dict(color=COLORS["text"],size=13))
    fig.update_yaxes(gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"],size=12), title_font=dict(color=COLORS["text"],size=13))
    return fig

pagina = st.segmented_control("Vista", ["Nacional", "Distrital"], default="Nacional", label_visibility="collapsed", key="pagina")
if pagina is None:
    pagina = "Nacional"

# ── NACIONAL ───────────────────────────────────────────────────────
if pagina == "Nacional":
    # Slot superior: KPIs + botón borrar selección (se rellena tras leer el mapa para que actualice en el mismo run)
    kpi_slot = st.container()
    sel_code_init = st.session_state.sel_depto_map

    # Fila 1: Mapa deptos + K-means deptos
    colA, colB = st.columns([1.1, 1])
    with colA:
        # Mapa votos por depto — gradiente log para no aplastar por Bogotá (satélite real en pestaña Distrital)
        dept_agg = flt.groupby("cod_dpto_geo").agg(votos=("votos","sum")).reset_index()
        g = dept_gdf.merge(dept_agg, left_on="dpto_ccdgo", right_on="cod_dpto_geo", how="left")
        g["votos"] = g["votos"].fillna(0)
        g["log_votos"] = np.log1p(g["votos"])
        fig = px.choropleth(g, geojson=g.__geo_interface__, locations="dpto_ccdgo", featureidkey="properties.dpto_ccdgo",
                            color="log_votos", hover_name="dpto_cnmbr",
                            hover_data={"log_votos":False, "votos":True, "dpto_ccdgo":False},
                            color_continuous_scale=[[0,"#FFF5F5"],[0.35,"#FCA5A5"],[0.65,COLORS["red"]],[1,COLORS["red_dark"]]],
                            title="Votos por departamento (escala log — clic para filtrar)")
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=40,b=0), coloraxis_colorbar=dict(title="log(votos)", tickfont=dict(color=COLORS["text"]), title_font=dict(color=COLORS["text"])))
        fig.update_layout(font=dict(color=COLORS["text"]))
        # Interactividad: selección
        sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        # capturar selección
        if sel and sel.get("selection",{}).get("points"):
            try:
                cc = sel["selection"]["points"][0].get("location")
                if cc and cc != st.session_state.sel_depto_map:
                    st.session_state.sel_depto_map = cc
                    st.rerun()
            except: pass
        # Filtro cascada para tablas/evolución/tarjetas (mismo run, sin lag)
        flt_nac_inter = flt.copy()
        if st.session_state.sel_depto_map:
            flt_nac_inter = flt[flt["cod_dpto_geo"] == st.session_state.sel_depto_map]

    with kpi_slot:
        st.markdown('<div class="page-title">Nacional — Nuevo Liberalismo</div>', unsafe_allow_html=True)
        if st.session_state.sel_depto_map:
            _bc1, _bc2 = st.columns([5, 1])
            with _bc2:
                if st.button("🔄 Borrar filtros", key="clr_dept_top", use_container_width=True):
                    st.session_state.sel_depto_map = None
                    st.rerun()
        base_kpi = flt_nac_inter if st.session_state.sel_depto_map and not flt_nac_inter.empty else flt
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi("Votos filtrados", fmt(base_kpi["votos"].sum()))
        with c2: kpi("Departamentos", fmt(base_kpi["departamento"].nunique()), f"de {nl['departamento'].nunique()}")
        with c3: kpi("Municipios", fmt(base_kpi["municipio"].nunique()))
        with c4: kpi("Candidatos", fmt(base_kpi["candidato"].nunique()))
        with c5: kpi("Años", ", ".join(sorted(base_kpi["ano"].dropna().astype(int).astype(str).unique())))

    with colB:
        # K-means departamental SIEMPRE sobre flt completo (estable) + resalta el depto del mapa
        dept_k = flt.groupby("departamento").agg(votos=("votos","sum"), municipios=("municipio","nunique"), cod=("cod_dpto_geo","first")).reset_index()
        dept_k = dept_k[~dept_k["departamento"].isin(["bogota", "consulados", "sin_dato"])].copy()
        if len(dept_k) >= 4:
            dept_k["log_votos"] = np.log1p(dept_k["votos"])
            X = StandardScaler().fit_transform(dept_k[["log_votos", "municipios"]])
            km = KMeans(n_clusters=4, n_init=20, random_state=42).fit(X)
            dept_k["cluster"] = km.labels_.astype(str)
            _sel = st.session_state.sel_depto_map
            dept_k["sel"] = (dept_k["cod"] == _sel) if _sel else False
            dept_k["size"] = np.where(dept_k["sel"], 16, 10)
            fig2 = px.scatter(dept_k, x="municipios", y="votos", color="cluster", hover_name="departamento",
                              log_y=True, size="size",
                              color_discrete_sequence=[COLORS["red"], COLORS["slate"], COLORS["teal"], "#F59E0B"],
                              title="K-means departamental (K=4, sin Bogotá) — el mapa resalta, no filtra")
            fig2.update_traces(marker=dict(line=dict(width=1, color="black")))
            st.plotly_chart(base_layout(fig2, 420), use_container_width=True)
            st.caption(f"Silhouette: {silhouette_score(X, km.labels_):.2f} — el punto grande es tu selección del mapa")
        else:
            st.info("Muy pocos departamentos para clusterizar con filtros actuales.")

    # Fila 2: Evolución + Corporación
    c1,c2 = st.columns(2)
    with c1:
        ev = flt_nac_inter.groupby("ano").agg(votos=("votos","sum")).reset_index().sort_values("ano")
        fig3 = px.line(ev, x="ano", y="votos", markers=True, title="Evolución temporal del partido", color_discrete_sequence=[COLORS["red"]])
        fig3.update_traces(line=dict(width=3))
        st.plotly_chart(base_layout(fig3,360), use_container_width=True)
    with c2:
        corp = flt_nac_inter.groupby("corporacion").agg(votos=("votos","sum")).sort_values("votos", ascending=True)
        fig4 = px.bar(corp, x="votos", y=corp.index, orientation="h", title="Votos por corporación", color_discrete_sequence=[COLORS["red_dark"]])
        fig4.update_traces(texttemplate="%{x:.2s}", textposition="outside")
        st.plotly_chart(base_layout(fig4,360), use_container_width=True)

    # Fila 3: Tablas
    t1,t2 = st.columns(2)
    with t1:
        cand = flt_nac_inter.groupby("candidato").agg(votos=("votos","sum"), municipios=("municipio","nunique"), registros=("votos","size")).sort_values("votos", ascending=False).head(top_n)
        st.markdown("**Top candidatos (filtrado)**")
        st.dataframe(cand, use_container_width=True)
    with t2:
        mun = flt_nac_inter.groupby(["municipio","departamento"]).agg(votos=("votos","sum"), candidatos=("candidato","nunique")).sort_values("votos", ascending=False).head(top_n)
        st.markdown("**Top municipios (filtrado)**")
        st.dataframe(mun, use_container_width=True)

# ── DISTRITAL ──────────────────────────────────────────────────────
if pagina == "Distrital":
    st.markdown('<div class="page-title">Distrital — Carlos Fernando Galán (Bogotá)</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Votos Galán", fmt(gal_flt["votacion"].sum()))
    with c2: kpi("Localidades", fmt(gal_flt["localidad"].nunique()))
    with c3: kpi("UPZ", fmt(gal_flt["upz"].nunique()))
    with c4: kpi("Puestos", fmt(gal_flt["lugar"].nunique()))

    # Preparar agregados para mapa y K-means
    upz_agg = gal_flt.groupby("upz").agg(votos=("votacion","sum"), puestos=("votacion","count"), lat=("latitud","mean"), lon=("longitud","mean")).reset_index().dropna(subset=["lat","lon"])
    upz_agg["votos_por_puesto"]=upz_agg["votos"]/upz_agg["puestos"]
    # K-means UPZ (votos/puestos) K=4
    if len(upz_agg) >= 4:
        feat = upz_agg[["votos","puestos","votos_por_puesto"]].copy()
        feat["log_votos"]=np.log1p(feat["votos"])
        Xup = StandardScaler().fit_transform(feat[["log_votos","puestos","votos_por_puesto"]])
        km_up = KMeans(n_clusters=4, n_init=30, random_state=42).fit(Xup)
        upz_agg["cluster"]=km_up.labels_.astype(str)
    else:
        upz_agg["cluster"]="0"

    # HDBSCAN sobre puestos (full gal para capa, no filtrado para estabilidad)
    @st.cache_data(show_spinner=False)
    def hdb_labels(df):
        pts = df.dropna(subset=["latitud","longitud"])
        if len(pts)<10:
            return pd.Series([], dtype=int)
        coords = np.radians(pts[["latitud","longitud"]].values)
        cl = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3, metric="haversine").fit_predict(coords)
        s=pd.Series(cl, index=pts.index)
        return s
    hdb = hdb_labels(gal)
    gal_hdb = gal.copy()
    gal_hdb["hdb"]=hdb.reindex(gal.index).fillna(-1).astype(int)

    # Mapa Folium
    st.markdown("**Mapa — activa/desactiva capas y haz clic para filtrar**")
    try:
        lat_c = float(gal_flt["latitud"].dropna().mean())
        lon_c = float(gal_flt["longitud"].dropna().mean())
        if pd.isna(lat_c) or pd.isna(lon_c):
            raise ValueError("sin coords")
        center = [lat_c, lon_c]
        m = folium.Map(location=center, zoom_start=11, control_scale=True)
        # Tiles sin API key (CartoDB ahora exige key y sale en blanco)
        folium.TileLayer("OpenStreetMap", name="Calle").add_to(m)
        folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Satélite").add_to(m)

        # UPZ choropleth (join normalizado: shape MAYUS vs parquet minusculas)
        if "nk" not in upz_gdf.columns:
            upz_gdf["nk"] = upz_gdf["nombre"].apply(norm_key)
        upz_agg["nk"] = upz_agg["upz"].apply(norm_key)
        upz_join = upz_gdf.merge(upz_agg[["nk", "upz", "votos"]], on="nk", how="left")
        # normalizar votos para color
        maxv = upz_join["votos"].max() if upz_join["votos"].notna().any() else 1
        def style_upz(f):
            v=f["properties"].get("votos")
            if pd.isna(v) or v==0:
                return {"fillColor":"#F0F0F0","color":"#333","weight":1,"fillOpacity":0.5}
            norm=v/maxv
            col = "#7C2D12" if norm>0.7 else "#B42318" if norm>0.4 else "#FCA5A5"
            return {"fillColor":col,"color":"#333","weight":1,"fillOpacity":0.7}
        fg_upz=folium.FeatureGroup(name="Votos por UPZ", show=True)
        folium.GeoJson(upz_join.__geo_interface__, style_function=style_upz,
            tooltip=folium.GeoJsonTooltip(fields=["nombre","votos"], aliases=["UPZ","Votos"], localize=True)).add_to(fg_upz)
        fg_upz.add_to(m)

        # Localidad choropleth (join normalizado + alias la candelaria)
        loc_agg = gal_flt.groupby("localidad").agg(votos=("votacion","sum")).reset_index()
        if "nk" not in loc_gdf.columns:
            loc_gdf["nk"] = loc_gdf["LocNombre"].apply(norm_key)
        loc_agg["nk"] = loc_agg["localidad"].apply(norm_key).replace({"la candelaria": "candelaria"})
        loc_join = loc_gdf.merge(loc_agg[["nk", "localidad", "votos"]], on="nk", how="left")
        maxv2 = loc_join["votos"].max() if loc_join["votos"].notna().any() else 1
        def style_loc(f):
            v=f["properties"].get("votos")
            if pd.isna(v): return {"fillColor":"#F0F0F0","color":"#111","weight":2,"fillOpacity":0.3}
            norm=v/maxv2
            col="#7C2D12" if norm>0.7 else "#B42318" if norm>0.4 else "#FCA5A5"
            return {"fillColor":col,"color":"#111","weight":2,"fillOpacity":0.4}
        fg_loc=folium.FeatureGroup(name="Votos por localidad", show=False)
        folium.GeoJson(loc_join.__geo_interface__, style_function=style_loc,
            tooltip=folium.GeoJsonTooltip(fields=["LocNombre","votos"], aliases=["Localidad","Votos"])).add_to(fg_loc)
        fg_loc.add_to(m)

        # Puestos: color por votos
        puestos = gal_flt.groupby(["lugar","localidad","latitud","longitud"]).agg(votos=("votacion","sum")).reset_index().dropna(subset=["latitud","longitud"])
        # limitar a 600 más votados si hay muchos para velocidad
        if len(puestos)>600:
            puestos = puestos.nlargest(600, "votos")
        # escala color
        q = puestos["votos"].quantile([0.33,0.66]).values
        def col_p(v):
            if v>q[1]: return COLORS["red_dark"]
            if v>q[0]: return COLORS["red"]
            return "#FCA5A5"
        fg_p = folium.FeatureGroup(name="Votos por puesto", show=True)
        for _,r in puestos.iterrows():
            folium.CircleMarker(location=[r["latitud"],r["longitud"]], radius=4+ min(6, r["votos"]/3000),
                color=col_p(r["votos"]), fill=True, fillColor=col_p(r["votos"]), fillOpacity=0.8, weight=1,
                popup=f"{r['lugar']}<br>{r['localidad']}<br>Votos: {int(r['votos'])}",
                tooltip=r["lugar"]).add_to(fg_p)
        fg_p.add_to(m)

        # HDBSCAN capa
        pts_hdb = gal_hdb[gal_hdb["hdb"]!=-1].groupby(["lugar","latitud","longitud"]).agg(votos=("votacion","sum"), hdb=("hdb","first")).reset_index().dropna()
        fg_hdb=folium.FeatureGroup(name="HDBSCAN clusters", show=False)
        cmap=plt.cm.tab20
        for cid in sorted(pts_hdb["hdb"].unique())[:12]:
            sub=pts_hdb[pts_hdb["hdb"]==cid]
            col = matplotlib.colors.to_hex(cmap(int(cid)%20))
            for _,r in sub.iterrows():
                folium.CircleMarker(location=[r["latitud"],r["longitud"]], radius=5, color=col, fill=True, fillColor=col, fillOpacity=0.9, weight=1,
                    popup=f"Cluster {cid}<br>{r['lugar']}").add_to(fg_hdb)
        fg_hdb.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        map_data = st_folium(m, height=520, use_container_width=True, returned_objects=["last_object_clicked"])
        if map_data and map_data.get("last_object_clicked"):
            st.session_state.sel_upz = map_data["last_object_clicked"].get("properties",{}).get("nombre") or map_data["last_object_clicked"].get("tooltip")
            if st.session_state.sel_upz:
                st.caption(f"Selección mapa: {st.session_state.sel_upz} — filtra K-means y tabla")
                if st.button("Limpiar selección mapa"):
                    st.session_state.sel_upz=None; st.rerun()
    except Exception as e:
        st.error(f"Mapa no disponible: {e}")

    # K-means UPZ gráfico
    c1,c2 = st.columns([1,1])
    with c1:
        figk = px.scatter(upz_agg, x="puestos", y="votos", color="cluster", hover_name="upz",
                          color_discrete_sequence=[COLORS["red"],COLORS["slate"],COLORS["teal"],"#F59E0B"],
                          title="K-means UPZ (K=4) — puestos vs votos", log_y=True)
        figk.update_traces(marker=dict(size=9, line=dict(width=1,color="black")))
        st.plotly_chart(base_layout(figk,380), use_container_width=True)
        # filtrar tabla si hay selección
        upz_show = upz_agg.copy()
        if st.session_state.sel_upz:
            # intentar filtrar por upz o localidad coincidente
            q = norm_key(st.session_state.sel_upz)
            mask = upz_show["upz"].fillna("").astype(str).apply(norm_key).str.contains(q, na=False)
            if mask.any():
                upz_show = upz_show[mask]
    with c2:
        puestos_tbl = gal_flt.groupby(["lugar","localidad","upz"]).agg(votos=("votacion","sum"), anos=("ano","nunique"), registros=("votacion","size")).sort_values("votos", ascending=False).head(top_n*2)
        if st.session_state.sel_upz and not puestos_tbl.empty:
            # filtrar por selección (normalizado, tolera NaN)
            q = norm_key(st.session_state.sel_upz)
            lvl_upz = puestos_tbl.index.get_level_values("upz").astype(str).map(norm_key)
            lvl_loc = puestos_tbl.index.get_level_values("localidad").astype(str).map(norm_key)
            mask = pd.Series(lvl_upz, index=puestos_tbl.index).str.contains(q, na=False) | pd.Series(lvl_loc, index=puestos_tbl.index).str.contains(q, na=False)
            if mask.any():
                puestos_tbl = puestos_tbl[mask.values]
        st.markdown("**Puestos de votación (filtrado)**")
        st.dataframe(puestos_tbl, use_container_width=True, height=380)

    st.caption("Filtros globales + clic en mapa filtran gráficos y tablas. Capas del mapa son solo visibilidad.")
