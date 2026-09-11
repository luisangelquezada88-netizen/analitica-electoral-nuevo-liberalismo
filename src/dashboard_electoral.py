import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Dashboard Electoral Colombia",
    layout="wide",
    initial_sidebar_state="expanded"
)

PARQUET_PATH = r"C:\Users\qluis\Documents\Proyectos\Nuevo liberalismo\output\dataset_final.parquet"
TOP_N_DEFAULT = 15

COLORS = {
    "bg": "#F6F8FB",
    "card": "#FFFFFF",
    "text": "#000000",
    "muted": "#5B6475",
    "grid": "#DCE3EC",
    "primary": "#0F766E",
    "secondary": "#334155",
    "accent": "#14B8A6",
    "amber": "#B45309",
    "red": "#B42318",
    "red_soft": "#E11D48"
}

# =========================================================
# ESTILO
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS["bg"]};
    }}
    .block-container {{
        padding-top: 1.15rem;
        padding-bottom: 1.5rem;
        max-width: 96rem;
    }}
    h1, h2, h3 {{
        color: {COLORS["text"]};
        letter-spacing: -0.02em;
    }}
    .kpi {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["grid"]};
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        min-height: 108px;
    }}
    .kpi-label {{
        color: {COLORS["muted"]};
        font-size: 0.87rem;
        margin-bottom: 0.2rem;
    }}
    .kpi-value {{
        color: {COLORS["text"]};
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.05;
    }}
    .kpi-sub {{
        color: {COLORS["muted"]};
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# UTILIDADES
# =========================================================
def fmt_int(x):
    if pd.isna(x):
        return "-"
    return f"{int(round(x)):,}".replace(",", ".")

def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.2f}%"

def clean_text(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()

def base_layout(fig, height=430, show_legend=True, top_margin=70):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Arial", size=13),
        margin=dict(l=20, r=20, t=top_margin, b=20),
        title=dict(
            text=fig.layout.title.text,
            font=dict(size=18, color=COLORS["text"]),
            x=0.0,
            xanchor="left",
            y=0.96
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=12, color=COLORS["text"]),
            title=dict(font=dict(size=12, color=COLORS["text"]))
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(size=12, color=COLORS["text"]),
        title_font=dict(size=13, color=COLORS["text"])
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(size=12, color=COLORS["text"]),
        title_font=dict(size=13, color=COLORS["text"])
    )
    return fig

def show_kpi(label, value, sub=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def empty_fig(title, height=420):
    fig = go.Figure()
    fig.add_annotation(
        text="Sin datos para la selección actual",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS["muted"])
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        template="plotly_white",
        height=height,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        margin=dict(l=20, r=20, t=100, b=20),
        font=dict(color=COLORS["text"])
    )
    return fig

@st.cache_data(show_spinner=False)
def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)

    required_cols = [
        "votos", "candidato", "partido", "departamento",
        "municipio", "ano", "corporacion"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el parquet: {missing}")

    df = df.copy()

    text_cols = ["candidato", "partido", "departamento", "municipio", "corporacion"]
    for c in text_cols:
        df[c] = df[c].apply(clean_text)

    df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df = df[df["votos"] >= 0].copy()

    df["Año electoral"] = df["ano"].astype("Int64").astype(str)
    df.loc[df["ano"].isna(), "Año electoral"] = "Sin dato"

    return df

def apply_filters(df, years, corporaciones, departamentos, municipios, partidos, candidatos):
    out = df.copy()
    if years:
        out = out[out["Año electoral"].isin(years)]
    if corporaciones:
        out = out[out["corporacion"].isin(corporaciones)]
    if departamentos:
        out = out[out["departamento"].isin(departamentos)]
    if municipios:
        out = out[out["municipio"].isin(municipios)]
    if partidos:
        out = out[out["partido"].isin(partidos)]
    if candidatos:
        out = out[out["candidato"].isin(candidatos)]
    return out

# =========================================================
# CARGA
# =========================================================
try:
    df = load_data(PARQUET_PATH)
except Exception as e:
    st.error(f"Error cargando los datos: {e}")
    st.stop()

# =========================================================
# FILTROS
# =========================================================
st.sidebar.title("Filtros")

year_options = sorted([x for x in df["Año electoral"].dropna().unique().tolist() if x != "Sin dato"])
corp_options = sorted(df["corporacion"].dropna().unique().tolist())
dept_options = sorted(df["departamento"].dropna().unique().tolist())

selected_years = st.sidebar.multiselect("Año electoral", year_options, default=year_options)
selected_corp = st.sidebar.multiselect("Corporación", corp_options, default=[])
selected_dept = st.sidebar.multiselect("Departamento", dept_options, default=[])

df_mun = df.copy()
if selected_dept:
    df_mun = df_mun[df_mun["departamento"].isin(selected_dept)]
mun_options = sorted(df_mun["municipio"].dropna().unique().tolist())
selected_mun = st.sidebar.multiselect("Municipio", mun_options, default=[])

df_party = df.copy()
if selected_years:
    df_party = df_party[df_party["Año electoral"].isin(selected_years)]
if selected_corp:
    df_party = df_party[df_party["corporacion"].isin(selected_corp)]
party_options = sorted(df_party["partido"].dropna().unique().tolist())
selected_party = st.sidebar.multiselect("Partido", party_options, default=[])

df_cand = df.copy()
if selected_party:
    df_cand = df_cand[df_cand["partido"].isin(selected_party)]
if selected_years:
    df_cand = df_cand[df_cand["Año electoral"].isin(selected_years)]
cand_options = sorted(df_cand["candidato"].dropna().unique().tolist())
selected_cand = st.sidebar.multiselect("Candidato", cand_options, default=[])

top_n = st.sidebar.slider("Top N", 5, 30, TOP_N_DEFAULT)
metric_mode = st.sidebar.radio("Métrica", ["Votos", "Participación %"])

filtered = apply_filters(
    df,
    selected_years,
    selected_corp,
    selected_dept,
    selected_mun,
    selected_party,
    selected_cand
)

if filtered.empty:
    st.warning("La combinación actual de filtros no devuelve registros.")
    st.stop()

# =========================================================
# KPIs
# =========================================================
total_votos = filtered["votos"].sum()
total_partidos = filtered["partido"].nunique(dropna=True)
total_candidatos = filtered["candidato"].nunique(dropna=True)
total_departamentos = filtered["departamento"].nunique(dropna=True)
total_municipios = filtered["municipio"].nunique(dropna=True)
total_corporaciones = filtered["corporacion"].nunique(dropna=True)

party_leader = (
    filtered.groupby("partido", as_index=False)["votos"]
    .sum()
    .sort_values("votos", ascending=False)
)

leader_name = party_leader.iloc[0]["partido"] if not party_leader.empty else "-"
leader_votes = party_leader.iloc[0]["votos"] if not party_leader.empty else 0
leader_share = (leader_votes / total_votos * 100) if total_votos > 0 else np.nan

corp_leader = (
    filtered.groupby("corporacion", as_index=False)["votos"]
    .sum()
    .sort_values("votos", ascending=False)
)
corp_name = corp_leader.iloc[0]["corporacion"] if not corp_leader.empty else "-"

# =========================================================
# HEADER
# =========================================================
st.title("Dashboard electoral de Colombia")

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    show_kpi("Total de votos", fmt_int(total_votos))
with k2:
    show_kpi("Partidos", fmt_int(total_partidos))
with k3:
    show_kpi("Candidatos", fmt_int(total_candidatos))
with k4:
    show_kpi("Cobertura territorial", fmt_int(total_departamentos), f"Municipios: {fmt_int(total_municipios)}")
with k5:
    show_kpi("Corporaciones", fmt_int(total_corporaciones), f"Líder: {corp_name}")
with k6:
    show_kpi("Coalición líder", leader_name, f"{fmt_pct(leader_share)} del total filtrado")

tab1, tab2, tab3 = st.tabs(["Panorama general", "Evolución y competencia", "Territorio y detalle"])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    c1, c2 = st.columns((1.1, 1))

    with c1:
        party_rank = (
            filtered.groupby("partido", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not party_rank.empty:
            party_rank["Participación %"] = np.where(
                total_votos > 0,
                party_rank["votos"] / total_votos * 100,
                0
            )
            xcol = "votos" if metric_mode == "Votos" else "Participación %"
            xtitle = "Votos" if metric_mode == "Votos" else "Participación (%)"

            fig = px.bar(
                party_rank.sort_values(xcol, ascending=True),
                x=xcol,
                y="partido",
                orientation="h",
                text=xcol,
                title="Ranking de coaliciones",
                color_discrete_sequence=[COLORS["primary"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}" if metric_mode == "Votos" else "%{text:.2f}%",
                hovertemplate="<b>%{y}</b><br>" + xtitle + ": %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title=xtitle, yaxis_title="")
            st.plotly_chart(base_layout(fig, 460, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Ranking de coaliciones", 460), use_container_width=True)

    with c2:
        cand_rank = (
            filtered.groupby("candidato", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not cand_rank.empty:
            cand_rank["Participación %"] = np.where(
                total_votos > 0,
                cand_rank["votos"] / total_votos * 100,
                0
            )
            xcol = "votos" if metric_mode == "Votos" else "Participación %"
            xtitle = "Votos" if metric_mode == "Votos" else "Participación (%)"

            fig = px.bar(
                cand_rank.sort_values(xcol, ascending=True),
                x=xcol,
                y="candidato",
                orientation="h",
                text=xcol,
                title="Candidatos destacados",
                color_discrete_sequence=[COLORS["red_soft"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}" if metric_mode == "Votos" else "%{text:.2f}%",
                hovertemplate="<b>%{y}</b><br>" + xtitle + ": %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title=xtitle, yaxis_title="")
            st.plotly_chart(base_layout(fig, 460, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Candidatos destacados", 460), use_container_width=True)

    c3, c4 = st.columns((1, 1))

    with c3:
        trend_total = (
            filtered.groupby("Año electoral", as_index=False)["votos"]
            .sum()
            .sort_values("Año electoral")
        )
        if not trend_total.empty:
            fig = px.line(
                trend_total,
                x="Año electoral",
                y="votos",
                markers=True,
                title="Total de votos por Año electoral",
                color_discrete_sequence=[COLORS["red"]]
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="<b>Año electoral %{x}</b><br>Votos: %{y:,.0f}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Año electoral", yaxis_title="Votos")
            st.plotly_chart(base_layout(fig, 430, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Total de votos por Año electoral", 430), use_container_width=True)

    with c4:
        corp_df = (
            filtered.groupby("corporacion", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
        )
        if not corp_df.empty:
            fig = px.treemap(
                corp_df,
                path=["corporacion"],
                values="votos",
                color="votos",
                color_continuous_scale="RdPu",
                title="Peso relativo por corporación"
            )
            fig.update_layout(
                title=dict(text="Peso relativo por corporación", font=dict(size=18, color=COLORS["text"])),
                paper_bgcolor=COLORS["card"],
                plot_bgcolor=COLORS["card"],
                margin=dict(l=10, r=10, t=55, b=10),
                font=dict(color=COLORS["text"])
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Peso relativo por corporación"), use_container_width=True)

    c5, c6 = st.columns((1, 1))

    with c5:
        dept_rank = (
            filtered.groupby("departamento", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not dept_rank.empty:
            fig = px.bar(
                dept_rank.sort_values("votos", ascending=True),
                x="votos",
                y="departamento",
                orientation="h",
                text="votos",
                title="Departamentos con mayor votación",
                color_discrete_sequence=[COLORS["secondary"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                hovertemplate="<b>%{y}</b><br>Votos: %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Votos", yaxis_title="")
            st.plotly_chart(base_layout(fig, 420, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Departamentos con mayor votación", 420), use_container_width=True)

    with c6:
        mun_rank_tab1 = (
            filtered.groupby(["municipio", "departamento"], as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not mun_rank_tab1.empty:
            mun_rank_tab1["label"] = mun_rank_tab1["municipio"] + " | " + mun_rank_tab1["departamento"]
            fig = px.bar(
                mun_rank_tab1.sort_values("votos", ascending=True),
                x="votos",
                y="label",
                orientation="h",
                text="votos",
                title="Ranking de municipios",
                color_discrete_sequence=[COLORS["red"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                hovertemplate="<b>%{y}</b><br>Votos: %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Votos", yaxis_title="")
            st.plotly_chart(base_layout(fig, 420, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Ranking de municipios", 420), use_container_width=True)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    c1, c2 = st.columns((1.05, 0.95))

    with c1:
        corp_top3 = (
            filtered.groupby("corporacion", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(3)["corporacion"]
            .tolist()
        )
        trend_top3 = (
            filtered[filtered["corporacion"].isin(corp_top3)]
            .groupby(["Año electoral", "corporacion"], as_index=False)["votos"]
            .sum()
            .sort_values(["Año electoral", "votos"], ascending=[True, False])
        )

        if not trend_top3.empty:
            fig = px.line(
                trend_top3,
                x="Año electoral",
                y="votos",
                color="corporacion",
                markers=True,
                title="Evolución electoral de las 3 corporaciones líderes",
                color_discrete_sequence=[COLORS["primary"], COLORS["red"], COLORS["amber"]]
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{fullData.name}</b><br>Año electoral: %{x}<br>Votos: %{y:,.0f}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Año electoral", yaxis_title="Votos")
            st.plotly_chart(base_layout(fig, 450, show_legend=True), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Evolución electoral de las 3 corporaciones líderes", 450), use_container_width=True)

    with c2:
        corp_year = (
            filtered.groupby(["Año electoral", "corporacion"], as_index=False)["votos"]
            .sum()
        )
        if not corp_year.empty:
            fig = px.bar(
                corp_year,
                x="Año electoral",
                y="votos",
                color="corporacion",
                barmode="group",
                title="Comparación por corporación y Año electoral",
                color_discrete_sequence=[COLORS["primary"], COLORS["red"], COLORS["amber"], COLORS["secondary"], COLORS["accent"]]
            )
            fig.update_layout(xaxis_title="Año electoral", yaxis_title="Votos")
            st.plotly_chart(base_layout(fig, 430, show_legend=True), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Comparación por corporación y Año electoral", 430), use_container_width=True)

    heat = (
        filtered.groupby(["partido", "corporacion"], as_index=False)["votos"]
        .sum()
    )
    if not heat.empty:
        heat_pivot = heat.pivot(index="partido", columns="corporacion", values="votos").fillna(0)
        fig = px.imshow(
            heat_pivot,
            aspect="auto",
            color_continuous_scale="RdPu",
            labels=dict(x="Corporación", y="Coalición", color="Votos"),
            text_auto=".2s"
        )
        fig.update_traces(
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate="Coalición: %{y}<br>Corporación: %{x}<br>Votos: %{z:,.0f}<extra></extra>"
        )
        fig.update_layout(
            title=dict(
                text="Matriz de votos: coalición vs corporación",
                font=dict(size=18, color=COLORS["text"])
            ),
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=80, b=20),
            height=560,
            coloraxis_colorbar=dict(
                title=dict(
                    text="Votos",
                    font=dict(size=13, color=COLORS["text"])
                ),
                tickfont=dict(size=12, color=COLORS["text"])
            )
        )
        fig.update_xaxes(
            side="bottom",
            tickfont=dict(size=12, color=COLORS["text"]),
            title=dict(text="Corporación", font=dict(size=13, color=COLORS["text"]))
        )
        fig.update_yaxes(
            tickfont=dict(size=12, color=COLORS["text"]),
            title=dict(text="Coalición", font=dict(size=13, color=COLORS["text"]))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Matriz de votos: coalición vs corporación", 560), use_container_width=True)

# =========================================================
# TAB 3
# =========================================================
with tab3:
    c1, c2 = st.columns((1, 1))

    with c1:
        dept_rank2 = (
            filtered.groupby("departamento", as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not dept_rank2.empty:
            fig = px.bar(
                dept_rank2.sort_values("votos", ascending=True),
                x="votos",
                y="departamento",
                orientation="h",
                text="votos",
                title="Ranking departamental",
                color_discrete_sequence=[COLORS["primary"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                hovertemplate="<b>%{y}</b><br>Votos: %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Votos", yaxis_title="")
            st.plotly_chart(base_layout(fig, 500, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Ranking departamental", 500), use_container_width=True)

    with c2:
        mun_rank = (
            filtered.groupby(["municipio", "departamento"], as_index=False)["votos"]
            .sum()
            .sort_values("votos", ascending=False)
            .head(top_n)
        )
        if not mun_rank.empty:
            mun_rank["label"] = mun_rank["municipio"] + " | " + mun_rank["departamento"]
            fig = px.bar(
                mun_rank.sort_values("votos", ascending=True),
                x="votos",
                y="label",
                orientation="h",
                text="votos",
                title="Municipios con mayor votación",
                color_discrete_sequence=[COLORS["red"]]
            )
            fig.update_traces(
                texttemplate="%{text:,.0f}",
                hovertemplate="<b>%{y}</b><br>Votos: %{x}<extra></extra>"
            )
            fig.update_layout(xaxis_title="Votos", yaxis_title="")
            st.plotly_chart(base_layout(fig, 500, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Municipios con mayor votación", 500), use_container_width=True)

    c3 = st.columns(1)[0]

    with c3:
        territory = (
            filtered.groupby("departamento", as_index=False)
            .agg(
                votos=("votos", "sum"),
                municipios=("municipio", "nunique"),
                candidatos=("candidato", "nunique"),
                coaliciones=("partido", "nunique")
            )
        )

        if not territory.empty:
            fig = px.scatter(
                territory,
                x="municipios",
                y="votos",
                size="candidatos",
                color="coaliciones",
                hover_name="departamento",
                labels={
                    "municipios": "Municipios con presencia",
                    "votos": "Votos",
                    "coaliciones": "Número de coaliciones",
                    "candidatos": "Número de candidatos"
                },
                color_continuous_scale="RdPu"
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Municipios: %{x}<br>"
                    "Votos: %{y:,.0f}<br>"
                    "Candidatos: %{marker.size}<extra></extra>"
                )
            )
            fig.update_layout(title="Concentración territorial del voto")
            st.plotly_chart(base_layout(fig, 500, show_legend=False), use_container_width=True)
        else:
            st.plotly_chart(empty_fig("Concentración territorial del voto", 500), use_container_width=True)

    territory_matrix = (
        filtered.groupby(["departamento", "corporacion"], as_index=False)["votos"]
        .sum()
    )
    if not territory_matrix.empty:
        pivot_territory = territory_matrix.pivot(
            index="departamento",
            columns="corporacion",
            values="votos"
        ).fillna(0)

        fig = px.imshow(
            pivot_territory,
            aspect="auto",
            color_continuous_scale="RdPu",
            labels=dict(x="Corporación", y="Departamento", color="Votos")
        )
        fig.update_layout(
            title=dict(
                text="Matriz territorial: departamento vs corporación",
                font=dict(size=18, color=COLORS["text"])
            ),
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=80, b=20),
            height=560,
            coloraxis_colorbar=dict(
                title=dict(
                    text="Votos",
                    font=dict(size=13, color=COLORS["text"])
                ),
                tickfont=dict(size=12, color=COLORS["text"])
            )
        )
        fig.update_xaxes(
            side="bottom",
            tickfont=dict(size=12, color=COLORS["text"]),
            title=dict(text="Corporación", font=dict(size=13, color=COLORS["text"]))
        )
        fig.update_yaxes(
            tickfont=dict(size=12, color=COLORS["text"]),
            title=dict(text="Departamento", font=dict(size=13, color=COLORS["text"]))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Matriz territorial: departamento vs corporación", 560), use_container_width=True)

    st.markdown("### Detalle analítico")

    detail = (
        filtered.groupby(
            ["Año electoral", "corporacion", "partido", "candidato", "departamento", "municipio"],
            as_index=False
        )["votos"]
        .sum()
        .sort_values("votos", ascending=False)
    )
    detail["Participación %"] = np.where(total_votos > 0, detail["votos"] / total_votos * 100, 0)

    detail = detail.rename(columns={
        "corporacion": "Corporación",
        "partido": "Coalición",
        "candidato": "Candidato",
        "departamento": "Departamento",
        "municipio": "Municipio",
        "votos": "Votos"
    })

    st.dataframe(detail, use_container_width=True, hide_index=True)

    csv = detail.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar detalle filtrado en CSV",
        data=csv,
        file_name="detalle_electoral_filtrado.csv",
        mime="text/csv"
    )