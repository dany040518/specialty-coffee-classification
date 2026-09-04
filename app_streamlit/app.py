"""Dashboard de los perfiles de calidad de café arabica.

Lee el dataset plano que produce la fase 06 del proyecto
(`reports/dashboard/coffee_dashboard_data.csv`, 1310 lotes) y presenta los tres
grupos del clustering con sus atributos sensoriales, su origen y su contraste
estadístico.

Se ejecuta desde esta carpeta:

    streamlit run app.py

El tema base (fondo, tipografía de los widgets, color primario) vive en
`.streamlit/config.toml`; acá solo se ajusta lo que ese archivo no cubre.
"""

from pathlib import Path
from string import Template

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats

CSV_RELATIVO = Path("../reports/dashboard/coffee_dashboard_data.csv")
PCA_RELATIVO = Path("../models/clustering_pca.joblib")

# Interfaz: tema claro sobre crema, tarjetas blancas y un marrón café como único
# acento (botones, título de sección, cifras destacadas). El texto usa dos
# niveles de contraste, no una escala de grises completa.
COLOR = {
    "fondo": "#F7F3EC",
    "superficie": "#FFFFFF",
    "borde": "#EAE2D6",
    "acento": "#5C4632",
    "acento_claro": "#8B6F47",
    "texto": "#2B2118",
    "texto_tenue": "#8A7C6C",
}

# Colores de los grupos: se separaron de la paleta café a propósito. Tres tonos
# de marrón entre sí son casi indistinguibles en un gráfico (es justo lo que
# pasaba antes: dos de los tres colores medían una distancia perceptual de 9.4,
# muy por debajo del mínimo de 15 para que un ojo sin daltonismo los diferencie).
# Este trío se validó con el validador de paletas del proyecto: separación
# perceptual y para las tres formas de daltonismo, y contraste >= 3:1 contra el
# fondo crema y contra la tarjeta blanca.
GRUPO_COLOR = {0: "#2F80ED", 1: "#EB5757", 2: "#0F9B8E"}

# La barra de Plotly se deja solo con lo que se usa: acercar, mover y volver.
CONFIG_PLOTLY = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian", "toImage",
    ],
}

# Plus Jakarta Sans: geometría redondeada, la misma familia para títulos y
# cuerpo (los títulos solo suben de peso), sin la formalidad de una serif.
SANS = "'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif"
SERIF = SANS  # se mantiene el nombre por compatibilidad con el resto del módulo


def miles(numero):
    """Formatea un entero con espacio fino como separador de miles."""
    return f"{numero:,}".replace(",", "\u2009")


def con_alfa(hexadecimal, alfa):
    """Convierte un color #rrggbb en rgba() con la transparencia indicada."""
    r, g, b = (int(hexadecimal[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alfa})"

# Los siete atributos que entraron al agrupamiento (fase 03 del proyecto).
ATRIBUTOS_SCA = [
    "Aroma",
    "Flavor",
    "Aftertaste",
    "Acidity",
    "Body",
    "Balance",
    "Cupper.Points",
]

# Centroide aproximado de cada país productor, para ubicarlo en el mapa.
COORDENADAS_PAIS = {
    "Brazil": [-14.24, -51.93],
    "Burundi": [-3.37, 29.92],
    "China": [35.86, 104.20],
    "Colombia": [4.57, -74.30],
    "Costa Rica": [9.75, -83.75],
    "Cote d?Ivoire": [7.54, -5.55],
    "Ecuador": [-1.83, -78.18],
    "El Salvador": [13.79, -88.90],
    "Ethiopia": [9.15, 40.49],
    "Guatemala": [15.78, -90.23],
    "Haiti": [18.97, -72.29],
    "Honduras": [15.20, -86.24],
    "India": [20.59, 78.96],
    "Indonesia": [-0.79, 113.92],
    "Japan": [36.20, 138.25],
    "Kenya": [-0.02, 37.91],
    "Laos": [19.85, 102.50],
    "Malawi": [-13.25, 34.30],
    "Mauritius": [-20.35, 57.55],
    "Mexico": [23.63, -102.55],
    "Myanmar": [21.92, 95.96],
    "Nicaragua": [12.87, -85.21],
    "Panama": [8.54, -80.78],
    "Papua New Guinea": [-6.31, 143.96],
    "Peru": [-9.19, -75.02],
    "Philippines": [12.88, 121.77],
    "Rwanda": [-1.94, 29.87],
    "Taiwan": [23.70, 120.96],
    "Tanzania, United Republic Of": [-6.37, 34.89],
    "Thailand": [15.87, 100.99],
    "Uganda": [1.37, 32.29],
    "United States": [37.09, -95.71],
    "United States (Hawaii)": [19.90, -155.58],
    "United States (Puerto Rico)": [18.22, -66.59],
    "Vietnam": [14.06, 108.28],
    "Zambia": [-13.13, 27.85],
}


# --- Datos -----------------------------------------------------------------


@st.cache_data
def cargar_datos():
    """Lee el CSV de la fase 06 desde la ruta relativa a este archivo."""
    ruta = (Path(__file__).parent / CSV_RELATIVO).resolve()
    if not ruta.exists():
        st.error(
            f"No se encontró {ruta}. Se genera con la fase 06 del proyecto "
            "(notebook 06 o pipeline.interpret_clusters)."
        )
        st.stop()
    return pd.read_csv(ruta)


@st.cache_data
def resumen_por_grupo(df):
    """Estadísticas de cada grupo, en el orden en que se muestran las tarjetas."""
    filas = []
    for grupo in sorted(df["grupo"].unique()):
        subset = df[df["grupo"] == grupo]
        altitudes = subset["altitude_mean_meters"].dropna()
        filas.append(
            {
                "grupo": grupo,
                "nombre": subset["grupo_nombre"].iloc[0],
                "n": len(subset),
                "puntaje": subset["Total.Cup.Points"].mean(),
                "especialidad": subset["especialidad"].mean() * 100,
                "altitud": altitudes.mean() if len(altitudes) else np.nan,
                "pais": subset["Country.of.Origin"].mode().iloc[0],
                "proceso": subset["Processing.Method"].mode().iloc[0],
                "defectos": subset["Category.Two.Defects"].mean(),
            }
        )
    return pd.DataFrame(filas)


@st.cache_data
def circulo_correlaciones():
    """Coordenadas del círculo de correlaciones: cada atributo de catación
    proyectado en Dim 1 / Dim 2, con su cos² (qué tan bien queda representado
    en ese plano). Misma fórmula que el notebook 04: como la matriz de entrada
    del PCA ya estaba estandarizada, corr(atributo, dimensión) es el autovector
    multiplicado por la raíz del autovalor de esa dimensión.

    Returns:
        Tupla ``(tabla, var_ratio)`` o None si no está guardado el PCA de la
        fase 05 (`models/clustering_pca.joblib`).
    """
    ruta = (Path(__file__).parent / PCA_RELATIVO).resolve()
    if not ruta.exists():
        return None
    pca = joblib.load(ruta)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    cos2 = (loadings**2).sum(axis=1)
    tabla = pd.DataFrame(
        {
            "atributo": ATRIBUTOS_SCA[: loadings.shape[0]],
            "dim1": loadings[:, 0],
            "dim2": loadings[:, 1],
            "cos2": cos2,
        }
    )
    return tabla, pca.explained_variance_ratio_


# --- Estilos ---------------------------------------------------------------


HOJA_ESTILOS = Template(
    """
<style>
/* La tipografía se importa con @import, no con un <link> aparte antes de esta
   etiqueta: el parser de Markdown de Streamlit busca en todo el bloque la
   subcadena que cierra esta etiqueta para saber dónde termina, así que otra
   etiqueta abierta antes (o esa subcadena mencionada dentro de un comentario,
   como casi pasa acá) corta el resto de la hoja y lo deja como texto suelto. */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp { background: $fondo; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2.5rem 3.5rem 5rem; max-width: 1500px; }

* { font-family: $sans; }
h1, h2, h3, h4 { font-family: $sans; color: $texto; letter-spacing: -0.01em; font-weight: 700; }

.masthead { padding-top: 0.4rem; margin-bottom: 3rem; }
.masthead .marca {
    font-size: 12px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: $acento_claro; margin-bottom: 0.7rem;
}
.masthead h1 { font-size: 38px; line-height: 1.15; margin: 0 0 0.8rem; font-weight: 800; }
.masthead .pie {
    font-size: 14.5px; font-weight: 500; color: $tenue; line-height: 1.65;
    max-width: 62ch;
}

.seccion { margin: 3.2rem 0 1.5rem; }
.seccion .indice {
    display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: $superficie; background: $acento;
    border-radius: 999px; padding: 0.22rem 0.7rem; margin-bottom: 0.8rem;
}
.seccion h2 { font-size: 24px; margin: 0; }
.seccion .nota { font-size: 13.5px; font-weight: 500; color: $tenue; margin-top: 0.45rem; }

.tarjeta-base {
    background: $superficie; border: 1px solid $borde; border-radius: 18px;
    box-shadow: 0 1px 2px rgba(43,33,24,0.04), 0 10px 24px rgba(43,33,24,0.05);
}

.kpi { padding: 1.3rem 1.4rem; height: 100%; }
.kpi .etiqueta {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: $tenue; margin-bottom: 0.7rem;
}
.kpi .valor { font-size: 30px; font-weight: 800; color: $acento; line-height: 1; }
.kpi .unidad { font-size: 16px; font-weight: 600; color: $acento_claro; margin-left: 0.15rem; }
.kpi .detalle { font-size: 12px; font-weight: 500; color: $tenue; margin-top: 0.6rem; }

.perfil {
    padding: 1.5rem 1.5rem 1.3rem; height: 100%;
    border-top: 4px solid $borde;
}
.perfil .cabecera {
    display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: $superficie;
    border-radius: 999px; padding: 0.2rem 0.65rem; margin-bottom: 0.85rem;
}
.perfil h3 { font-size: 18px; line-height: 1.35; margin: 0 0 1.1rem; font-weight: 700; }
.perfil .datos { font-size: 13.5px; font-weight: 500; }
.perfil .fila {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem; padding: 0.55rem 0; border-top: 1px solid $borde;
}
.perfil .fila:first-child { border-top: none; }
.perfil .fila .etiqueta { color: $tenue; }
.perfil .fila .valor { color: $texto; font-weight: 600; text-align: right; white-space: nowrap; }

.leyenda { display: flex; flex-wrap: wrap; gap: 1.8rem; margin: 0.4rem 0 1.2rem;
    font-size: 13px; font-weight: 600; color: $texto; }
.leyenda span { display: flex; align-items: center; gap: 0.55rem; }
.leyenda i { width: 11px; height: 11px; border-radius: 4px; display: inline-block; }

.vacio { padding: 2.5rem; text-align: center; font-weight: 500; color: $tenue; }

.pie-pagina { margin-top: 4rem; border-top: 1px solid $borde; padding-top: 1.3rem;
    font-size: 12px; font-weight: 500; color: $tenue; line-height: 1.8; }

[data-testid="stSidebar"] { background: $superficie; border-right: 1px solid $borde; }
[data-testid="stSidebar"] .block-container { padding: 2.5rem 1.5rem; }
[data-testid="stSidebar"] label {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase;
    color: $tenue !important;
}
[data-testid="stSidebar"] .sidebar-marca {
    font-size: 20px; font-weight: 800; color: $texto;
    padding-bottom: 1.1rem; margin-bottom: 1.6rem;
}
[data-testid="stSidebar"] .sidebar-nota {
    font-size: 12px; font-weight: 500; color: $tenue; line-height: 1.6;
    border-top: 1px solid $borde; margin-top: 1.8rem; padding-top: 1.1rem;
}
[data-baseweb="tag"] { background: $acento !important; border-radius: 8px !important; }
[data-testid="stDataFrame"] { border: 1px solid $borde; border-radius: 14px; overflow: hidden; }
.stButton>button, .stDownloadButton>button {
    background: $acento; color: $superficie; border-radius: 999px; border: none;
    font-weight: 700;
}

div[data-testid="column"] { padding: 0 0.45rem; }
hr { border-color: $borde; }
</style>
"""
)


def aplicar_estilos():
    st.markdown(
        HOJA_ESTILOS.substitute(
            fondo=COLOR["fondo"],
            superficie=COLOR["superficie"],
            borde=COLOR["borde"],
            acento=COLOR["acento"],
            acento_claro=COLOR["acento_claro"],
            texto=COLOR["texto"],
            tenue=COLOR["texto_tenue"],
            sans=SANS,
        ),
        unsafe_allow_html=True,
    )


def registrar_plantilla_plotly():
    """Deja una sola plantilla de Plotly para todos los gráficos de la página."""
    pio.templates["cafe"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # El texto de los gráficos usa el mismo color oscuro que el resto de
            # la página ($texto), no el tenue de las etiquetas secundarias: en un
            # gráfico los números son el contenido, no un detalle de apoyo. Los
            # tamaños también quedan un escalón arriba del resto de la interfaz,
            # porque acá se vuelve a leer con el gráfico ya escalado o ampliado.
            font=dict(family=SANS, size=15, color=COLOR["texto"]),
            title=dict(
                font=dict(family=SERIF, size=18, color=COLOR["acento"]),
                x=0,
                xanchor="left",
                xref="paper",
                y=0.97,
                yanchor="top",
                yref="container",
            ),
            xaxis=dict(
                gridcolor=COLOR["borde"],
                zeroline=False,
                linecolor=COLOR["borde"],
                tickfont=dict(size=14, color=COLOR["texto"]),
                title=dict(font=dict(size=14, color=COLOR["texto"])),
            ),
            yaxis=dict(
                gridcolor=COLOR["borde"],
                zeroline=False,
                linecolor=COLOR["borde"],
                tickfont=dict(size=14, color=COLOR["texto"]),
                title=dict(font=dict(size=14, color=COLOR["texto"])),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                font=dict(size=13.5, color=COLOR["texto"]),
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor=COLOR["superficie"],
                bordercolor=COLOR["borde"],
                font=dict(family=SANS, size=14, color=COLOR["texto"]),
            ),
            margin=dict(l=10, r=10, t=95, b=10),
            colorway=[GRUPO_COLOR[0], GRUPO_COLOR[1], GRUPO_COLOR[2]],
        )
    )
    pio.templates.default = "cafe"


def encabezado_seccion(indice, titulo, nota=None):
    pie = f'<div class="nota">{nota}</div>' if nota else ""
    st.markdown(
        f'<div class="seccion"><div class="indice">{indice}</div>'
        f"<h2>{titulo}</h2>{pie}</div>",
        unsafe_allow_html=True,
    )


def tarjeta_kpi(etiqueta, valor, unidad="", detalle=""):
    sufijo = f'<span class="unidad">{unidad}</span>' if unidad else ""
    linea = f'<div class="detalle">{detalle}</div>' if detalle else ""
    return (
        f'<div class="tarjeta-base kpi"><div class="etiqueta">{etiqueta}</div>'
        f'<div class="valor">{valor}{sufijo}</div>{linea}</div>'
    )


def leyenda_grupos(df):
    """Leyenda de identidad de los grupos: color acompañado siempre del nombre."""
    partes = []
    for grupo in sorted(df["grupo"].unique()):
        n = int((df["grupo"] == grupo).sum())
        partes.append(
            f'<span><i style="background:{GRUPO_COLOR[grupo]}"></i>'
            f"Grupo {grupo} · {miles(n)} lotes</span>"
        )
    st.markdown(f'<div class="leyenda">{"".join(partes)}</div>', unsafe_allow_html=True)


# --- Filtros ---------------------------------------------------------------


def panel_filtros(df):
    """Dibuja la barra lateral y devuelve el DataFrame filtrado."""
    lado = st.sidebar
    lado.markdown('<div class="sidebar-marca">Filtros</div>', unsafe_allow_html=True)

    # Las opciones se declaran como texto en vez de usar format_func: así el
    # valor que guarda el widget es el mismo que se ve en pantalla.
    etiquetas_grupo = {f"Grupo {g}": g for g in sorted(df["grupo"].unique())}
    elegidos = lado.multiselect(
        "Grupo", options=list(etiquetas_grupo), default=list(etiquetas_grupo)
    )
    grupos = [etiquetas_grupo[e] for e in elegidos]

    puntaje_lo, puntaje_hi = lado.slider(
        "Puntaje total",
        float(df["Total.Cup.Points"].min()),
        float(df["Total.Cup.Points"].max()),
        (float(df["Total.Cup.Points"].min()), float(df["Total.Cup.Points"].max())),
        step=0.25,
    )

    orden_paises = df["Country.of.Origin"].value_counts().index.tolist()
    paises = lado.multiselect(
        "País de origen",
        options=orden_paises,
        default=[],
        placeholder="Todos los países",
    )

    etiquetas_especialidad = {"80 puntos o más": 1, "Menos de 80": 0}
    elegidas = lado.multiselect(
        "Especialidad",
        options=list(etiquetas_especialidad),
        default=list(etiquetas_especialidad),
    )
    especialidad = [etiquetas_especialidad[e] for e in elegidas]

    altitudes = df["altitude_mean_meters"].dropna()
    altitud_lo, altitud_hi = lado.slider(
        "Altitud reportada (m)",
        int(altitudes.min()),
        int(altitudes.max()),
        (int(altitudes.min()), int(altitudes.max())),
        step=50,
    )
    # Un lote sin altitud no es un lote sin calidad: la fase 03 decidió no
    # imputar ese dato. Si el filtro los descartara en silencio se perdería el
    # 21 % de la base sin que nadie lo note.
    sin_altitud = lado.checkbox("Incluir lotes sin altitud reportada", value=True)

    en_rango = df["altitude_mean_meters"].between(altitud_lo, altitud_hi)
    if sin_altitud:
        en_rango = en_rango | df["altitude_mean_meters"].isna()

    filtrado = df[
        df["grupo"].isin(grupos)
        & df["Total.Cup.Points"].between(puntaje_lo, puntaje_hi)
        & df["especialidad"].isin(especialidad)
        & en_rango
    ]
    if paises:
        filtrado = filtrado[filtrado["Country.of.Origin"].isin(paises)]

    lado.markdown(
        f'<div class="sidebar-nota">{miles(len(filtrado))} de {miles(len(df))} lotes '
        "seleccionados.<br>Sin países marcados se muestran todos.</div>",
        unsafe_allow_html=True,
    )
    return filtrado


# --- Secciones -------------------------------------------------------------


def cabecera(df, df_filtrado):
    st.markdown(
        '<div class="masthead">'
        '<div class="marca">Coffee Quality Institute · Arabica</div>'
        "<h1>Coffee Quality Analysis</h1>"
        f'<div class="pie">Clasificación no supervisada de {miles(len(df))} lotes de café '
        "en tres perfiles, a partir de los siete atributos de catación evaluados por "
        "catadores certificados. Origen, altitud y método de beneficio no participaron "
        "del agrupamiento: se usan para describirlo.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def fila_indicadores(df, df_filtrado):
    columnas = st.columns(5, gap="small")
    n = len(df_filtrado)

    with columnas[0]:
        st.markdown(
            tarjeta_kpi(
                "Lotes",
                miles(n),
                detalle=f"de {miles(len(df))} en la base",
            ),
            unsafe_allow_html=True,
        )
    with columnas[1]:
        pct = df_filtrado["especialidad"].mean() * 100
        st.markdown(
            tarjeta_kpi("Especialidad", f"{pct:.1f}", "%", "80 puntos o más"),
            unsafe_allow_html=True,
        )
    with columnas[2]:
        puntaje = df_filtrado["Total.Cup.Points"].mean()
        st.markdown(
            tarjeta_kpi(
                "Puntaje medio",
                f"{puntaje:.2f}",
                detalle=f"desviación {df_filtrado['Total.Cup.Points'].std():.2f}",
            ),
            unsafe_allow_html=True,
        )
    with columnas[3]:
        altitudes = df_filtrado["altitude_mean_meters"].dropna()
        valor = miles(round(altitudes.mean())) if len(altitudes) else "sin dato"
        st.markdown(
            tarjeta_kpi(
                "Altitud media",
                valor,
                "m" if len(altitudes) else "",
                f"{miles(len(altitudes))} lotes la reportan",
            ),
            unsafe_allow_html=True,
        )
    with columnas[4]:
        conteo = df_filtrado["Country.of.Origin"].value_counts()
        st.markdown(
            tarjeta_kpi(
                "Origen principal",
                f'<span style="font-size:23px">{conteo.index[0]}</span>',
                detalle=f"{conteo.iloc[0] / n * 100:.0f} % de la selección",
            ),
            unsafe_allow_html=True,
        )


def seccion_perfiles(df_filtrado):
    encabezado_seccion(
        "01", "Los tres perfiles",
        "Los grupos quedaron ordenados igual en los siete atributos de catación, "
        "sin una sola excepción.",
    )
    resumen = resumen_por_grupo(df_filtrado)
    columnas = st.columns(len(resumen), gap="medium")

    for columna, (_, fila) in zip(columnas, resumen.iterrows()):
        tono = GRUPO_COLOR[fila["grupo"]]
        altitud = (
            f"{miles(round(fila['altitud']))} m" if pd.notna(fila["altitud"]) else "sin dato"
        )
        datos = [
            ("Lotes", miles(fila["n"])),
            ("Puntaje medio", f"{fila['puntaje']:.2f}"),
            ("Especialidad", f"{fila['especialidad']:.1f} %"),
            ("Altitud media", altitud),
            ("Origen principal", fila["pais"]),
            ("Beneficio", fila["proceso"]),
            ("Defectos cat. 2", f"{fila['defectos']:.2f}"),
        ]
        filas = "".join(
            f'<div class="fila"><span class="etiqueta">{etiqueta}</span>'
            f'<span class="valor">{valor}</span></div>'
            for etiqueta, valor in datos
        )
        with columna:
            # El color de grupo se fija con propiedades normales (border-top-color,
            # background), no con una variable CSS (--tono): el sanitizador de
            # Streamlit borra cualquier propiedad "--nombre" de un atributo style,
            # aunque conserve las propiedades estándar del mismo atributo.
            st.markdown(
                f'<div class="tarjeta-base perfil" style="border-top-color:{tono}">'
                f'<div class="cabecera" style="background:{tono}">Grupo {fila["grupo"]}</div>'
                f'<h3>{fila["nombre"]}</h3><div class="datos">{filas}</div></div>',
                unsafe_allow_html=True,
            )


def seccion_sensorial(df_filtrado):
    encabezado_seccion(
        "02", "Espacio sensorial",
        "Cada punto es un lote. Los atributos están tan correlacionados entre sí "
        "que la nube se ordena sobre una sola diagonal de calidad general.",
    )
    leyenda_grupos(df_filtrado)
    izquierda, derecha = st.columns([1.15, 1], gap="medium")

    with izquierda:
        figura = go.Figure()
        for grupo in sorted(df_filtrado["grupo"].unique()):
            subset = df_filtrado[df_filtrado["grupo"] == grupo]
            figura.add_trace(
                go.Scatter(
                    x=subset["Aroma"],
                    y=subset["Flavor"],
                    mode="markers",
                    name=f"Grupo {grupo}",
                    marker=dict(
                        size=9,
                        color=GRUPO_COLOR[grupo],
                        opacity=0.72,
                        line=dict(width=0.8, color=COLOR["fondo"]),
                    ),
                    customdata=np.stack(
                        [subset["Country.of.Origin"], subset["Total.Cup.Points"]], axis=-1
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Aroma %{x:.2f} · "
                        "Flavor %{y:.2f}<br>Total %{customdata[1]:.2f}<extra></extra>"
                    ),
                )
            )
        figura.update_layout(
            title="Flavor vs Aroma",
            xaxis_title="Aroma",
            yaxis_title="Flavor",
            height=470,
        )
        st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)

    with derecha:
        circulo = circulo_correlaciones()
        if circulo is None:
            st.markdown(
                '<div class="tarjeta-base vacio">No se encontró '
                "<code>models/clustering_pca.joblib</code>. Se genera con la fase "
                "05 del proyecto.</div>",
                unsafe_allow_html=True,
            )
        else:
            tabla_circulo, var_ratio = circulo
            figura = go.Figure()

            angulo = np.linspace(0, 2 * np.pi, 100)
            figura.add_trace(
                go.Scatter(
                    x=np.cos(angulo), y=np.sin(angulo), mode="lines",
                    line=dict(color=COLOR["borde"], width=1, dash="dash"),
                    hoverinfo="skip", showlegend=False,
                )
            )
            figura.add_hline(y=0, line=dict(color=COLOR["borde"], width=1))
            figura.add_vline(x=0, line=dict(color=COLOR["borde"], width=1))

            # Los siete atributos apuntan casi todos en la misma dirección (es el
            # hallazgo de esta fase: un solo eje de calidad general), y varios
            # quedan a menos de 0.15 uno de otro en Dim 2: ninguna posición de
            # texto evita que sus etiquetas se pisen. En vez de desplazar el
            # texto lejos de su punto real (lo que engañaría más de lo que
            # ayuda), se etiqueta cada punto solo si hay separación real desde
            # la última etiqueta mostrada; el resto se identifica con el cursor,
            # igual que cualquier otro punto del gráfico.
            tabla_circulo = tabla_circulo.sort_values("dim2", ascending=False).reset_index(
                drop=True
            )
            separacion_minima = 0.15
            ultimo_y_mostrado = None
            texto, ocultos = [], []
            for atributo, y in zip(tabla_circulo["atributo"], tabla_circulo["dim2"]):
                if ultimo_y_mostrado is None or (ultimo_y_mostrado - y) >= separacion_minima:
                    texto.append(atributo)
                    ultimo_y_mostrado = y
                else:
                    texto.append("")
                    ocultos.append(atributo)
            posiciones = ["top center", "bottom center", "middle right"]
            texto_posiciones = [posiciones[i % 3] for i in range(len(tabla_circulo))]

            # Una flecha por atributo, desde el origen hasta su correlación con
            # cada componente. Es la misma cuenta de la fase 04 (notebook 04,
            # celda de "círculo de correlaciones"): como la matriz de entrada del
            # PCA ya estaba estandarizada, corr(atributo, componente) es
            # simplemente el autovector multiplicado por la raíz del autovalor.
            flechas = [
                dict(
                    x=fila["dim1"], y=fila["dim2"], ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowwidth=2, arrowcolor=COLOR["acento_claro"],
                )
                for _, fila in tabla_circulo.iterrows()
            ]
            # El hover usa el nombre del atributo vía customdata, no vía `text`:
            # `text` acá trae "" para los puntos sin etiqueta visible, y si el
            # hover también leyera de `text` esos puntos aparecerían sin nombre
            # al pasarles el cursor.
            figura.add_trace(
                go.Scatter(
                    x=tabla_circulo["dim1"], y=tabla_circulo["dim2"],
                    mode="markers+text",
                    text=texto,
                    textposition=texto_posiciones,
                    textfont=dict(size=13.5, color=COLOR["texto"]),
                    marker=dict(
                        size=11,
                        color=tabla_circulo["cos2"],
                        colorscale=[[0, "#EAF2FE"], [1, "#1B4FA0"]],
                        cmin=0, cmax=1,
                        line=dict(width=1, color=COLOR["superficie"]),
                        colorbar=dict(
                            title=dict(text="cos²", font=dict(size=13)),
                            outlinewidth=0, thickness=12, len=0.75,
                            tickfont=dict(size=12.5, color=COLOR["texto"]),
                        ),
                    ),
                    customdata=np.stack(
                        [tabla_circulo["atributo"], tabla_circulo["cos2"]], axis=-1
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Dim 1: %{x:.2f} · Dim 2: %{y:.2f}"
                        "<br>cos² %{customdata[1]:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            # Sin scaleanchor: forzar la escala 1:1 contra un contenedor mucho más
            # ancho que alto recortaba el eje x a una tajada del círculo. Se
            # prefiere mostrar el rango completo, aunque el círculo se vea algo
            # ovalado, a perder la mitad izquierda del gráfico.
            figura.update_layout(
                title="Círculo de correlaciones",
                height=520,
                margin=dict(l=10, r=10, t=95, b=10),
                xaxis=dict(
                    title=f"Dim 1 ({var_ratio[0] * 100:.1f} %)",
                    range=[-1.2, 1.2], zeroline=False, showgrid=False,
                ),
                yaxis=dict(
                    title=f"Dim 2 ({var_ratio[1] * 100:.1f} %)",
                    range=[-1.2, 1.2], zeroline=False, showgrid=False,
                ),
                annotations=flechas,
            )
            st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)
            if ocultos:
                st.markdown(
                    f'<div class="pie-pagina" style="margin-top:-0.6rem;border:none;'
                    f'padding:0">{", ".join(ocultos)}: vectores casi superpuestos '
                    "con los de un vecino; se identifican al pasar el cursor por "
                    "el punto.</div>",
                    unsafe_allow_html=True,
                )


def seccion_distribucion(df_filtrado):
    encabezado_seccion(
        "03", "Distribución de puntajes",
        "La línea marca los 80 puntos, el umbral con que la industria define un "
        "café de especialidad.",
    )
    izquierda, derecha = st.columns([1, 1.3], gap="medium")

    with izquierda:
        figura = go.Figure()
        for grupo in sorted(df_filtrado["grupo"].unique()):
            figura.add_trace(
                go.Box(
                    y=df_filtrado[df_filtrado["grupo"] == grupo]["Total.Cup.Points"],
                    name=f"Grupo {grupo}",
                    marker_color=GRUPO_COLOR[grupo],
                    line=dict(width=1.6),
                    fillcolor="rgba(0,0,0,0)",
                    boxmean=True,
                    hovertemplate="%{y:.2f}<extra></extra>",
                )
            )
        figura.add_hline(
            y=80,
            line=dict(color=COLOR["acento"], width=1, dash="dot"),
            annotation_text="80 puntos",
            annotation_position="top left",
            annotation_font=dict(size=13.5, color=COLOR["acento"]),
        )
        figura.update_layout(
            title="Puntaje total por grupo", yaxis_title="Total Cup Points",
            height=440, showlegend=False,
        )
        st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)

    with derecha:
        figura = go.Figure()
        rango = (
            df_filtrado["Total.Cup.Points"].min(),
            df_filtrado["Total.Cup.Points"].max(),
        )
        for grupo in sorted(df_filtrado["grupo"].unique()):
            figura.add_trace(
                go.Histogram(
                    x=df_filtrado[df_filtrado["grupo"] == grupo]["Total.Cup.Points"],
                    name=f"Grupo {grupo}",
                    marker=dict(color=GRUPO_COLOR[grupo], line=dict(width=0)),
                    opacity=0.82,
                    xbins=dict(start=rango[0], end=rango[1], size=0.5),
                    hovertemplate="%{x} · %{y} lotes<extra></extra>",
                )
            )
        figura.add_vline(x=80, line=dict(color=COLOR["acento"], width=1, dash="dot"))
        figura.update_layout(
            title="Lotes por tramo de puntaje",
            barmode="stack",
            bargap=0.06,
            xaxis_title="Total Cup Points",
            yaxis_title="Lotes",
            height=440,
        )
        st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)


def seccion_geografia(df_filtrado):
    encabezado_seccion(
        "04", "Origen",
        "Cada círculo es un país; su tamaño refleja cuántos lotes aporta y su color, "
        "el grupo mayoritario dentro de ese país.",
    )
    mapa_col, barras_col = st.columns([1.45, 1], gap="medium")

    por_pais = (
        df_filtrado.groupby("Country.of.Origin")
        .agg(
            lotes=("Total.Cup.Points", "size"),
            puntaje=("Total.Cup.Points", "mean"),
            grupo=("grupo", lambda serie: serie.mode().iloc[0]),
        )
        .sort_values("lotes", ascending=False)
    )

    with mapa_col:
        # Mapa vectorial: dibuja costas y fronteras sin pedir tiles a ningún
        # servidor, así que no depende de una API key ni de la conexión.
        ubicados = por_pais[por_pais.index.isin(COORDENADAS_PAIS)]
        figura = go.Figure()
        for grupo in sorted(ubicados["grupo"].unique()):
            subconjunto = ubicados[ubicados["grupo"] == grupo]
            figura.add_trace(
                go.Scattergeo(
                    lat=[COORDENADAS_PAIS[p][0] for p in subconjunto.index],
                    lon=[COORDENADAS_PAIS[p][1] for p in subconjunto.index],
                    text=subconjunto.index,
                    name=f"Grupo {grupo}",
                    mode="markers",
                    marker=dict(
                        size=np.clip(np.sqrt(subconjunto["lotes"]) * 3.4, 9, 46),
                        color=GRUPO_COLOR[grupo],
                        opacity=0.68,
                        line=dict(width=1, color=COLOR["fondo"]),
                    ),
                    customdata=np.stack(
                        [subconjunto["lotes"], subconjunto["puntaje"]], axis=-1
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>%{customdata[0]} lotes<br>"
                        "Puntaje medio %{customdata[1]:.2f}<extra></extra>"
                    ),
                )
            )
        figura.update_layout(
            title="Lotes por país de origen",
            height=470,
            showlegend=False,
            margin=dict(l=0, r=0, t=58, b=0),
            geo=dict(
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#EFE8DB",
                showcountries=True,
                countrycolor="#D9CFBE",
                coastlinecolor="#D9CFBE",
                coastlinewidth=0.7,
                showocean=False,
                showframe=False,
                lataxis=dict(range=[-36, 38]),
                lonaxis=dict(range=[-112, 158]),
            ),
        )
        st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)

    with barras_col:
        top = por_pais.head(10).iloc[::-1]
        figura = go.Figure(
            go.Bar(
                x=top["lotes"],
                y=top.index,
                orientation="h",
                marker=dict(
                    color=[GRUPO_COLOR[g] for g in top["grupo"]], line=dict(width=0)
                ),
                text=top["lotes"],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=13.5, color=COLOR["texto"]),
                customdata=top["puntaje"],
                hovertemplate="<b>%{y}</b><br>%{x} lotes<br>"
                "Puntaje medio %{customdata:.2f}<extra></extra>",
            )
        )
        figura.update_layout(
            title="Orígenes principales",
            height=470,
            xaxis_title="Lotes",
            xaxis=dict(range=[0, top["lotes"].max() * 1.16]),
            showlegend=False,
            margin=dict(l=10, r=20, t=58, b=10),
        )
        st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)


def seccion_relaciones(df_filtrado):
    encabezado_seccion(
        "05", "Correlaciones",
        "Los siete atributos se mueven casi en bloque. La altitud, en cambio, "
        "apenas se relaciona con ellos.",
    )
    variables = [
        columna
        for columna in ATRIBUTOS_SCA + ["Total.Cup.Points", "altitude_mean_meters"]
        if columna in df_filtrado.columns
    ]
    matriz = df_filtrado[variables].corr()
    etiquetas = [v.replace("altitude_mean_meters", "Altitud") for v in variables]

    # Entre estos atributos las correlaciones son todas positivas, así que una
    # escala de un solo tono de claro a oscuro se lee mejor que una divergente.
    # Si algún filtro llegara a producir una correlación negativa, se cambia a
    # divergente con un gris neutro en el cero, que es lo correcto para ese caso.
    # Los dos polos se detienen en un tono medio, no en el más saturado: así el
    # texto oscuro de las celdas se sigue leyendo en toda la escala.
    hay_negativas = matriz.values.min() < 0
    if hay_negativas:
        escala = [[0.0, "#F6B8B4"], [0.5, COLOR["borde"]], [1.0, "#8FC0F5"]]
        limites = dict(zmin=-1, zmax=1)
    else:
        escala = [[0.0, "#EAF2FE"], [1.0, "#5B9BEA"]]
        limites = dict(zmin=0, zmax=1)
    color_texto = COLOR["texto"]

    figura = go.Figure(
        go.Heatmap(
            z=matriz.values,
            x=etiquetas,
            y=etiquetas,
            colorscale=escala,
            **limites,
            text=np.round(matriz.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=14, family=SANS, color=color_texto),
            hovertemplate="%{y} · %{x}<br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(
                outlinewidth=0,
                tickfont=dict(size=13, color=COLOR["texto"]),
                thickness=12,
                len=0.8,
            ),
            xgap=2,
            ygap=2,
        )
    )
    figura.update_layout(
        title="Correlación entre atributos",
        height=520,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(figura, use_container_width=True, config=CONFIG_PLOTLY)


def seccion_contraste(df_filtrado):
    encabezado_seccion(
        "06", "Contraste estadístico",
        "Prueba de que la diferencia de puntaje entre los grupos no es casualidad "
        "del muestreo.",
    )
    izquierda, derecha = st.columns([1, 1.6], gap="medium")

    muestras = [
        df_filtrado[df_filtrado["grupo"] == grupo]["Total.Cup.Points"].values
        for grupo in sorted(df_filtrado["grupo"].unique())
    ]
    muestras = [m for m in muestras if len(m) > 1]

    with izquierda:
        if len(muestras) > 1:
            f_stat, p_valor = stats.f_oneway(*muestras)
            veredicto = (
                "Las medias difieren" if p_valor < 0.05 else "No hay evidencia de diferencia"
            )
            st.markdown(
                tarjeta_kpi("ANOVA · F", miles(round(f_stat)))
                        + '<div style="height:0.8rem"></div>'
                + tarjeta_kpi(
                    "Valor p",
                    f"{p_valor:.2e}" if p_valor else "< 1e-300",
                    detalle=f"{veredicto} (α = 0.05)",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tarjeta-base vacio">Se necesitan al menos dos grupos con lotes '
                "para comparar medias.</div>",
                unsafe_allow_html=True,
            )

    with derecha:
        tabla = (
            df_filtrado.groupby("grupo")["Total.Cup.Points"]
            .agg(
                Lotes="size",
                Media="mean",
                Mediana="median",
                Desviación="std",
                Mínimo="min",
                Q1=lambda serie: serie.quantile(0.25),
                Q3=lambda serie: serie.quantile(0.75),
                Máximo="max",
            )
            .round(2)
        )
        tabla.index = [f"Grupo {g}" for g in tabla.index]
        st.dataframe(tabla, use_container_width=True)


def seccion_registros(df_filtrado):
    encabezado_seccion("07", "Registros")

    columnas_tabla = [
        "Country.of.Origin",
        "Variety",
        "Processing.Method",
        "Total.Cup.Points",
        "Aroma",
        "Flavor",
        "Aftertaste",
        "Acidity",
        "Body",
        "Balance",
        "altitude_mean_meters",
        "grupo",
    ]
    columnas_tabla = [c for c in columnas_tabla if c in df_filtrado.columns]

    buscador, contador = st.columns([3, 1], gap="medium")
    with buscador:
        consulta = st.text_input("Buscar por país o variedad", "", placeholder="Colombia, Bourbon...")
    with contador:
        cuantas = st.number_input("Filas", value=25, min_value=5, max_value=500, step=25)

    tabla = df_filtrado
    if consulta:
        tabla = tabla[
            tabla["Country.of.Origin"].str.contains(consulta, case=False, na=False)
            | tabla["Variety"].str.contains(consulta, case=False, na=False)
        ]

    # La altitud sin reportar se deja vacía en vez de mostrar "None".
    visible = tabla[columnas_tabla].head(int(cuantas)).copy()
    visible["altitude_mean_meters"] = pd.to_numeric(
        visible["altitude_mean_meters"], errors="coerce"
    )

    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Country.of.Origin": st.column_config.TextColumn("País"),
            "Variety": st.column_config.TextColumn("Variedad"),
            "Processing.Method": st.column_config.TextColumn("Beneficio"),
            "Total.Cup.Points": st.column_config.NumberColumn("Total", format="%.2f"),
            "altitude_mean_meters": st.column_config.NumberColumn("Altitud", format="%.0f"),
            "grupo": st.column_config.NumberColumn("Grupo", format="%d"),
        },
    )
    st.markdown(
        f'<div class="pie-pagina" style="margin-top:0.8rem;border:none;padding:0">'
        f"{miles(min(int(cuantas), len(tabla)))} de {miles(len(tabla))} registros que "
        "cumplen los filtros.</div>",
        unsafe_allow_html=True,
    )


def pie_pagina(df):
    st.markdown(
        '<div class="pie-pagina">'
        "Coffee Quality Institute · muestras de arabica catadas entre 2010 y 2018.<br>"
        "Los tres perfiles provienen de k-means sobre las dos primeras componentes "
        "principales de los siete atributos de catación, con semilla fija. "
        "La base reúne cafés enviados a certificar, no una muestra del café que se "
        "produce en el mundo."
        "</div>",
        unsafe_allow_html=True,
    )


# --- Aplicación ------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Coffee Quality Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    aplicar_estilos()
    registrar_plantilla_plotly()

    df = cargar_datos()
    df_filtrado = panel_filtros(df)

    cabecera(df, df_filtrado)

    if df_filtrado.empty:
        st.markdown(
            '<div class="tarjeta-base vacio">Ningún lote cumple los filtros seleccionados. '
            "Ampliá el rango de puntaje o quitá países de la selección.</div>",
            unsafe_allow_html=True,
        )
        pie_pagina(df)
        return

    fila_indicadores(df, df_filtrado)
    seccion_perfiles(df_filtrado)
    seccion_sensorial(df_filtrado)
    seccion_distribucion(df_filtrado)
    seccion_geografia(df_filtrado)
    seccion_relaciones(df_filtrado)
    seccion_contraste(df_filtrado)
    seccion_registros(df_filtrado)
    pie_pagina(df)


if __name__ == "__main__":
    main()
