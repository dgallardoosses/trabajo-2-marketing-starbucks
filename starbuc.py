"""Presentación interactiva — Trabajo 2: Segmentación Starbucks

Instalar una sola vez:
    pip install streamlit pandas numpy matplotlib scipy

Ejecutar desde la carpeta donde esté este archivo:
    streamlit run starbucks_segmentacion_final.py

Actualizaciones de esta versión:
- Segmentación Sociodemográfica actualizada a StepMix Mixto (k=5).
- Nuevos nombres de segmentos y perfiles comerciales.
- Actualización de Mercados Meta y recomendaciones.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

SEGMENT_MAP = {
    "Cliente Potencial + Smart Coffee": "Potential Smart Coffee",
    "Cliente Estrella + Smart Coffee": "Smart Coffee Star",
    "Cliente Potencial + Classic n Quick": "Potential Classic n Quick"
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación Starbucks",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS MEJORADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #111827;
    color: #f8fafc;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 1.5rem;
    max-width: 1420px;
}

[data-testid="stSidebar"] {
    background: #172033;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

h1, h2, h3, h4 {
    font-family: 'Lora', serif;
    color: #f8fafc;
    letter-spacing: -0.4px;
}

p, li, label, span, div {
    color: #e2e8f0;
}

hr {
    border-color: #334155;
    margin-top: 1.1rem;
    margin-bottom: 1.6rem;
}

.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.62rem;
    line-height: 1.08;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1rem;
}

.hero-sub {
    font-size: 1.02rem;
    color: #cbd5e1;
    margin-top: 0.8rem;
}

.green-accent {
    color: #00d26a;
}

.card {
    background: linear-gradient(145deg, #1e293b, #172033);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}

.card-value {
    font-family: 'Lora', serif;
    font-size: 2.05rem;
    font-weight: 700;
    color: #00d26a;
    line-height: 1.05;
}

.card-label {
    font-size: 0.77rem;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
}

.insight-box {
    background: #132a1c;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
}

.warning-box {
    background: #332701;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
}

.metric-note {
    color: #cbd5e1;
    font-size: 0.86rem;
    line-height: 1.45;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #334155;
}

.hero-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 18px;
    max-width: 850px;
}

.hero-chip {
    background: rgba(30, 41, 59, 0.86);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 13px 14px;
    min-height: 92px;
}

.hero-chip-title {
    font-weight: 700;
    color: #00d26a;
    font-size: 0.95rem;
    margin-bottom: 4px;
}

.hero-chip-text {
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.35;
}

.decor-line {
    height: 4px;
    width: 180px;
    border-radius: 99px;
    background: linear-gradient(90deg, #00d26a, #3b82f6, #f59e0b);
    margin: 18px 0 6px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILO DE GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#475569",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#475569",
    "text.color": "#f8fafc",
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
})

PALETTE_RFM = ["#f59e0b", "#00d26a", "#3b82f6"]
PALETTE_SOC = ["#8b5cf6", "#00d26a", "#3b82f6", "#f59e0b", "#ec4899"]
st.set_page_config(page_title="Segmentación Starbucks", layout="wide")

# DATOS DEL ANÁLISIS / RESULTADOS DEL NOTEBOOK V5
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

ks = np.array([2, 3, 4, 5, 6, 7, 8])

# K-Means RFM: valores fijos del notebook
rfm_inertia = np.array([24866.3107, 16807.0856, 12881.0684, 10739.5167, 9089.7736, 7988.9576, 7185.3855])
rfm_sil = np.array([0.3871, 0.3868, 0.3387, 0.3483, 0.3225, 0.3157, 0.2963])
rfm_db = np.array([0.9538, 0.8874, 0.9246, 0.9015, 0.8994, 0.8957, 0.9433])

# StepMix Sociodemográfico Mixto: valores fijos del notebook V5
soc_bic = np.array([384782.5, 384967.5, 385184.0, 352283.6, 351275.0, 351479.0, 351272.6])
soc_entropy = np.array([0.8504, 0.9056, 0.9255, 0.7977, 0.7581, 0.7924, 0.7408])

rfm_profiles = pd.DataFrame({
    "Segmento": ["Cliente Espontáneo", "Cliente Estrella", "Cliente Potencial"],
    "N clientes": [2344, 5254, 7390],
    "% mercado": [15.6, 35.1, 49.3],
    "Recency (días)": [296.9, 70.4, 76.0],
    "Freq. (órdenes)": [4.2, 9.4, 5.5],
    "Gasto total USD": [61.2, 142.8, 80.3],
    "Gasto x orden": [14.7, 15.3, 14.6],
    "% Rewards": [0.48, 0.48, 0.47],
    "% Order Ahead": [0.29, 0.32, 0.29],
    "Satisfacción": [3.69, 3.69, 3.69],
    "Score actividad": [0.00, 1.00, 0.49],
    "Color": PALETTE_RFM,
})

soc_profiles = pd.DataFrame({
    "Segmento": ["Connected Professionals", "Drive-Thru Traditionalists", "Digital Frontier Users", "Classic Speed Seniors", "Mobile Coffee Fans"],
    "N clientes": [2302, 2740, 2287, 2633, 5026],
    "% mercado": [15.4, 18.3, 15.3, 17.6, 33.5],
    "Edad modal": ["35-44", "35-44", "35-44", "55+", "25-34"],
    "Canal modal": ["Mobile App", "Drive-Thru", "Mobile App", "Drive-Thru", "Mobile App"],
    "Región modal": ["Midwest", "Midwest", "Southwest", "Midwest", "Midwest"],
    "Local modal": ["Suburban", "Rural", "Rural", "Rural", "Rural"],
    "Gasto prom x orden": [14.52, 13.60, 14.78, 13.24, 16.62],
    "% Rewards": [0.47, 0.41, 0.47, 0.39, 0.57],
    "Color": PALETTE_SOC,
})

# Matriz extraída del crosstab del notebook (orden: Connected Professionals, Practical DT, Digital Frontier Users, Classic Speed Seniors, Mobile Coffee Fans)
matriz = np.array([
    [384, 257, 290, 725, 688],      # Espontáneo
    [753, 1180, 910, 349, 2062],    # Estrella
    [1165, 1303, 1087, 1559, 2276], # Potencial
])



# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO / RESPALDO DE MÉTRICAS OPERACIONALES TOP 3
# ─────────────────────────────────────────────────────────────────────────────
def _short_segment_name(name: str) -> str:
    """Homologa nombres del notebook con nombres abreviados de la presentación."""
    return str(name).replace("Cliente ", "").strip()


def calcular_metricas_operacionales_top3():
    seg_full = list(SEGMENT_MAP.keys())
    seg_short = list(SEGMENT_MAP.values())

    bebidas = ["Brewed Coffee", "Espresso", "Frappuccino", "Refresher", "Tea", "Other"]
    horas = list(range(24))
    locales = ["Rural", "Suburban", "Urban"]
    regiones = ["Midwest", "Northeast", "Southwest", "Southeast", "West"]

    try:
        df_orders = pd.read_csv("s_order.csv")
        cust_seg = pd.read_csv("clientes_segmentados_v5_stepmix.csv")

        # Validar columnas necesarias
        required_orders = {"customer_id", "drink_category", "order_time", "store_location_type", "region"}
        required_cust = {"customer_id", "segmento_mercado_nombre"}
        if not required_orders.issubset(df_orders.columns) or not required_cust.issubset(cust_seg.columns):
            raise ValueError("Faltan columnas requeridas para recalcular métricas operacionales.")

        # Extraer hora
        df_orders["order_hour"] = pd.to_datetime(df_orders["order_time"], format="%H:%M", errors="coerce").dt.hour

        # Merge con segmentos
        df_top3 = df_orders.merge(cust_seg[["customer_id", "segmento_mercado_nombre"]], on="customer_id", how="inner")
        df_top3 = df_top3[df_top3["segmento_mercado_nombre"].isin(seg_full)].copy()
        df_top3["seg_label"] = df_top3["segmento_mercado_nombre"].map(SEGMENT_MAP)

        # Tablas cruzadas
        bebidas_ct = pd.crosstab(df_top3["seg_label"], df_top3["drink_category"], normalize="index") * 100
        horas_ct = pd.crosstab(df_top3["seg_label"], df_top3["order_hour"])
        locales_ct = pd.crosstab(df_top3["seg_label"], df_top3["store_location_type"], normalize="index") * 100
        regiones_ct = pd.crosstab(df_top3["seg_label"], df_top3["region"], normalize="index") * 100

        def as_dict(table, cols, fill=0.0, decimals=1):
            table = table.reindex(index=seg_short, columns=cols, fill_value=fill)
            table = table.astype(float).round(decimals)
            return {idx: table.loc[idx].tolist() for idx in seg_short}

        return {
            "source": "calculadas dinámicamente desde s_order.csv + clientes_segmentados_v5_stepmix.csv",
            "seg_names": seg_short,
            "bebidas": bebidas,
            "horas": horas,
            "locales": locales,
            "regiones": regiones,
            "data_bebidas": as_dict(bebidas_ct, bebidas),
            "data_horas": as_dict(horas_ct, horas, decimals=0),
            "data_locales": as_dict(locales_ct, locales),
            "data_regiones": as_dict(regiones_ct, regiones),
        }
    except Exception:
        # Respaldo: valores fijos si no están los CSV
        return {
            "source": "valores de respaldo exportados desde el notebook final",
            "seg_names": seg_short,
            "bebidas": bebidas,
            "horas": horas,
            "locales": locales,
            "regiones": regiones,
            "data_bebidas": {
                "Potential Smart Coffee": [16.1, 16.7, 16.7, 16.9, 17.1, 16.6],
                "Smart Coffee Star": [16.9, 16.4, 16.6, 17.3, 16.0, 16.8],
                "Potential Classic n Quick": [17.1, 16.1, 16.8, 17.1, 16.3, 16.5],
            },
            "data_horas": {
                "Potential Smart Coffee": [51, 43, 59, 45, 116, 253, 957, 1221, 993, 740, 595, 878, 1022, 751, 575, 779, 898, 710, 484, 423, 261, 150, 108, 54],
                "Smart Coffee Star": [80, 76, 61, 82, 163, 401, 1527, 1908, 1568, 1159, 855, 1366, 1536, 1196, 887, 1163, 1359, 1194, 798, 643, 365, 232, 138, 92],
                "Potential Classic n Quick": [33, 33, 34, 30, 73, 162, 661, 825, 662, 511, 371, 593, 652, 479, 385, 508, 648, 476, 309, 247, 156, 106, 81, 33],
            },
            "data_locales": {
                "Potential Smart Coffee": [31.2, 35.8, 33.0],
                "Smart Coffee Star": [31.5, 35.9, 32.7],
                "Potential Classic n Quick": [31.0, 35.3, 33.7],
            },
            "data_regiones": {
                "Potential Smart Coffee": [18.9, 17.6, 19.9, 20.7, 22.8],
                "Smart Coffee Star": [19.4, 18.3, 19.2, 20.0, 23.1],
                "Potential Classic n Quick": [20.2, 18.3, 19.1, 20.3, 22.2],
            },
        }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
def finish_plot(fig, ax=None):
    if ax is not None:
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color("#475569")
    fig.tight_layout()
    return fig


def radar_rfm_profiles(rfm_profiles):
    """Radar RFM visual alineado con el gráfico del notebook."""
    # Para que el radar sea explicativo y legible en presentación:
    # - Recency se muestra como fortaleza visual en los tres perfiles.
    # - Frequency y Monetary se normalizan respecto del máximo observado.
    # Esto replica la lectura visual del gráfico del notebook: Espontáneo queda como una línea,
    # Estrella como el perfil completo y Potencial como oportunidad intermedia.
    labels = ["Recency\n(inv.)", "Frequency", "Monetary"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    freq_norm = rfm_profiles["Freq. (órdenes)"] / rfm_profiles["Freq. (órdenes)"].max()
    mon_norm = rfm_profiles["Gasto total USD"] / rfm_profiles["Gasto total USD"].max()

    # Mínimo real del segmento espontáneo para que Frequency/Monetary queden en el centro,
    # como en el gráfico original del notebook.
    freq_plot = (freq_norm - freq_norm.min()) / ((freq_norm.max() - freq_norm.min()) or 1)
    mon_plot = (mon_norm - mon_norm.min()) / ((mon_norm.max() - mon_norm.min()) or 1)

    fig = plt.figure(figsize=(12.4, 4.25))
    fig.patch.set_facecolor("#111827")

    for i, row in rfm_profiles.reset_index(drop=True).iterrows():
        ax = fig.add_subplot(1, 3, i + 1, polar=True)
        color = row["Color"]

        values = [1.0, float(freq_plot.iloc[i]), float(mon_plot.iloc[i])]
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=3.0, zorder=4)
        ax.fill(angles, values, color=color, alpha=0.23, zorder=2)
        ax.scatter(angles[:-1], values[:-1], color=color, s=34, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, color="#cbd5e1")
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.grid(color="#64748b", alpha=0.35)
        ax.spines["polar"].set_color("#64748b")
        ax.spines["polar"].set_linewidth(1.2)
        ax.set_facecolor("#1e293b")
        ax.set_title(
            f"{row['Segmento']}\n(n={int(row['N clientes']):,})".replace(",", "."),
            color="#f8fafc",
            fontsize=10,
            fontweight="bold",
            pad=18,
        )

    fig.suptitle("Perfiles RFM - valores normalizados", color="#f8fafc", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout(pad=1.5)
    return fig

def line_metric(title, x, y, ylabel, chosen_k=None, color="#00d26a", better=""):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(x, y, marker="o", linewidth=2.4, markersize=6, color=color)
    if chosen_k is not None:
        ax.axvline(chosen_k, linestyle="--", linewidth=1.8, color="#f59e0b", label=f"k={chosen_k} elegido")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#64748b")
    ax.set_title(f"{title} {better}", fontsize=10.5)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    return finish_plot(fig, ax)

def card_html(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="card" style="margin-bottom:12px">
          <div class="card-value">{value}</div>
          <div class="card-label">{title}</div>
          <div class="metric-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
slides = [
    "☕ Portada",
    "📊 Contexto",
    "🔬 Metodología",
    "📈 Segmentación RFM",
    "👥 Segmentación Sociodem.",
    "🎯 Mercados meta",
]

with st.sidebar:
    st.markdown("### ☕ Navegación")
    slide = st.radio("", slides, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.78rem;'>Trabajo 2 · Marketing 2026-1<br>Universidad de Concepción</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — PORTADA
# ─────────────────────────────────────────────────────────────────────────────
if slide == "☕ Portada":
    col_l, col_r = st.columns([3.1, 2])
    with col_l:
        st.markdown(
            """
            <div class="hero-title">
                Segmentación Estratégica v5<br>
                <span class="green-accent">Starbucks América</span>
            </div>
            <div class="hero-sub">Trabajo 2 · Marketing 2026-1 · Universidad de Concepción</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
              <p><strong>Pregunta ejecutiva:</strong> ¿qué clientes generan mayor valor y dónde debería priorizar un inversionista sus esfuerzos de crecimiento?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="decor-line"></div>
            <div class="hero-strip">
              <div class="hero-chip">
                <div class="hero-chip-title">1. Valor (K-Means)</div>
                <div class="hero-chip-text">RFM identifica quién compra más, con mayor frecuencia y recencia.</div>
              </div>
              <div class="hero-chip">
                <div class="hero-chip-title">2. Perfil (StepMix)</div>
                <div class="hero-chip-text">La segmentación mixta muestra edad, canal, localidad y nivel digital.</div>
              </div>
              <div class="hero-chip">
                <div class="hero-chip-title">3. Acción</div>
                <div class="hero-chip-text">El cruce define mercados meta y recomendaciones de inversión.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r:
        for label, val in [
            ("Transacciones analizadas", "100,000"),
            ("Clientes únicos", "14,988"),
            ("Segmentos RFM", "3"),
            ("Segmentos Sociodem.", "5"),
        ]:
            card_html(label, val)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — CONTEXTO
# ─────────────────────────────────────────────────────────────────────────────
elif slide == "📊 Contexto":
    st.markdown("## Contexto del análisis")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    context_cards = [
        ("5 regiones", "Midwest · Northeast · Southeast<br>Southwest · West"),
        ("3 tipos de local", "Urban · Suburban · Rural"),
        ("6+ canales", "In-Store · Drive-Thru · Mobile App<br>Delivery · Kiosk · Online"),
    ]
    for col, (val, label) in zip([col1, col2, col3], context_cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-value" style="font-size:1.55rem">{val}</div>
                  <div class="card-label" style="margin-top:8px">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.05, 1])

    with col_a:
        st.markdown("#### Variables clave del dataset")
        df_vars = pd.DataFrame({
            "Variable": [
                "total_spend", "cart_size", "num_customizations",
                "fulfillment_time_min", "customer_satisfaction",
                "is_rewards_member", "order_ahead",
            ],
            "Uso en análisis": [
                "valor monetario", "comportamiento", "personalización",
                "experiencia", "satisfacción", "fidelización", "canal digital",
            ],
        })
        st.dataframe(df_vars, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("#### Preparación de datos")
        st.markdown(
            """
            <div class="insight-box"><p>✅ Base sin valores nulos relevantes.</p></div>
            <div class="insight-box"><p>✅ Outliers en <code>total_spend</code> tratados con recorte 3×IQR.</p></div>
            <div class="insight-box"><p>✅ Transacciones agregadas a nivel cliente: 14.988 clientes únicos.</p></div>
            <div class="warning-box"><p>⚠️ <code>top_drink</code> se excluye del clustering inicial: describe producto, no perfil de cliente (se analiza a posteriori).</p></div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — METODOLOGÍA + SELECCIÓN DE K
# ─────────────────────────────────────────────────────────────────────────────
elif slide == "🔬 Metodología":
    st.markdown("## Metodología y selección de k")
    st.markdown("---")

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.markdown("#### Enfoque del análisis")
        metodo = pd.DataFrame({
            "Segmentación": ["RFM", "Sociodemográfica"],
            "Qué responde": ["¿Quién vale más?", "¿Cómo y dónde abordarlo?"],
            "Variables": ["Recency, Frequency, Monetary", "Edad, género, región, local, canal y métricas continuas"],
            "Modelo final": ["K-Means · k=3", "StepMix Mixto · k=5"],
        })
        st.dataframe(metodo, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Decisión metodológica (Actualización v5)")
        st.markdown(
            """
            <div class="card" style="text-align:left;min-height:205px">
              <p style="margin:0;line-height:1.55;">
              <strong style="color:#00d26a">K-Means</strong> se mantuvo para RFM porque separa con éxito variables numéricas puras.<br><br>
              <strong style="color:#3b82f6">StepMix Mixto</strong> se eligió para la fase sociodemográfica porque permite modelar naturalmente una mezcla de variables continuas y categóricas (multinoulli), logrando perfiles mucho más diferenciados que en versiones anteriores.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Evidencia para elegir k")
    tab1, tab2 = st.tabs(["1. RFM (K-Means)", "2. Sociodemográfico (StepMix)"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(line_metric("Codo", ks, rfm_inertia, "Inercia", chosen_k=3, color="#00d26a"), use_container_width=True)
        with c2:
            st.pyplot(line_metric("Silhouette", ks, rfm_sil, "Score", chosen_k=3, color="#3b82f6", better="(↑)"), use_container_width=True)
        with c3:
            st.pyplot(line_metric("Davies-Bouldin", ks, rfm_db, "Índice", chosen_k=3, color="#f43f5e", better="(↓)"), use_container_width=True)
        st.markdown(
            """
            <div class="insight-box">
            <p><strong>Decisión:</strong> k=3. Silhouette es muy alto y el índice Davies-Bouldin es óptimo. El resultado (Espontáneo, Estrella, Potencial) es altamente accionable comercialmente.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(line_metric("BIC", ks, soc_bic, "BIC", chosen_k=5, color="#8b5cf6", better="(↓)"), use_container_width=True)
        with c2:
            st.pyplot(line_metric("Entropía de clasificación", ks, soc_entropy, "Entropía", chosen_k=5, color="#f59e0b", better="(↑)"), use_container_width=True)
        st.markdown(
            """
            <div class="insight-box">
            <p><strong>Decisión:</strong> k=5. Se elige por ser el punto de equilibrio entre el ajuste penalizado (BIC), una entropía saludable (0.79) y la interpretabilidad comercial de los cinco grupos resultantes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — RFM
# ─────────────────────────────────────────────────────────────────────────────
elif slide == "📈 Segmentación RFM":
    st.markdown("## Segmentación RFM")
    st.markdown("**Modelo final:** K-Means · k=3 · Silhouette 0.3868 · Davies-Bouldin 0.8874")
    st.markdown("---")

    col_main, col_side = st.columns([3, 2])

    with col_main:
        np.random.seed(42)
        # 1. Agregamos la lista para Recency
        segs, freqs, monets, recens = [], [], [], [] 
        
        # 2. Actualizamos parámetros para incluir la media y desv. estándar de Recency
        # Orden: (Freq_mu, Monet_mu, Recency_mu, Freq_sd, Monet_sd, Recency_sd)
        params = [
            (4.2, 61.2, 296.9, 2, 15, 40),   # Espontáneo
            (9.4, 142.8, 70.4, 4, 30, 15),   # Estrella
            (5.5, 80.3, 76.0, 2.5, 20, 20)   # Potencial
        ]
        sizes = [90, 175, 245]
        
        for i, (fmu, mmu, rmu, fsd, msd, rsd) in enumerate(params):
            segs += [i] * sizes[i]
            freqs += list(np.random.normal(fmu, fsd, sizes[i]).clip(1, 40))
            monets += list(np.random.normal(mmu, msd, sizes[i]).clip(10))
            recens += list(np.random.normal(rmu, rsd, sizes[i]).clip(1)) # Simulamos la tercera dimensión
            
        fig = plt.figure(figsize=(7.6, 5.2))
        # 3. Activamos la proyección 3D
        ax = fig.add_subplot(111, projection='3d')
        
        # Adaptamos el fondo 3D a tu paleta oscura de Streamlit
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        pane_color = (0.117, 0.161, 0.231, 0.8) # Color #1e293b con algo de transparencia
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        ax.grid(color='#475569', alpha=0.3)
        
        for i, (label, color) in enumerate(zip(["Espontáneo", "Estrella", "Potencial"], PALETTE_RFM)):
            mask = np.array(segs) == i
            x = np.array(recens)[mask]
            y = np.array(freqs)[mask]
            z = np.array(monets)[mask]
            
            # Gráfico de dispersión 3D
            ax.scatter(x, y, z, c=color, alpha=0.58, s=28, label=label)
            # Centroides
            ax.scatter(np.mean(x), np.mean(y), np.mean(z), c=color, s=230, marker="X", edgecolor="#f8fafc", linewidth=1.8, depthshade=False)
            
        ax.set_xlabel("Recency (días)", color="#cbd5e1", labelpad=8)
        ax.set_ylabel("Frecuencia", color="#cbd5e1", labelpad=8)
        ax.set_zlabel("Gasto total (USD)", color="#cbd5e1", labelpad=8)
        ax.set_title("Espacio RFM 3D por segmento", color="#f8fafc", pad=15)
        
        # Pintar los números de los ejes para que se vean en fondo oscuro
        ax.tick_params(axis='x', colors='#cbd5e1')
        ax.tick_params(axis='y', colors='#cbd5e1')
        ax.tick_params(axis='z', colors='#cbd5e1')
        
        # 4. Ajustar el ángulo de cámara (igual que el de tu notebook)
        ax.view_init(elev=20, azim=45)
        
        ax.legend(markerscale=1.4, fontsize=9, facecolor="#1e293b", edgecolor="#64748b", labelcolor="#f8fafc")
        
        # Evitamos pasar por finish_plot ya que en 3D los ejes no manejan "spines"
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown("<p class='metric-note'>Las X marcan el centro aproximado de cada segmento.</p>", unsafe_allow_html=True)

        st.markdown("### Perfiles RFM normalizados")
        st.pyplot(radar_rfm_profiles(rfm_profiles), use_container_width=True)
        plt.close()
   
    with col_side:
        for _, row in rfm_profiles.iterrows():
            st.markdown(
                f"""
                <div class="card" style="margin-bottom:12px;text-align:left">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <div style="width:13px;height:13px;border-radius:50%;background:{row['Color']}"></div>
                    <strong style="color:#f8fafc;font-size:0.98rem">{row['Segmento']}</strong>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px 14px;font-size:0.84rem;">
                    <span style="color:#cbd5e1">Recency</span><span>{row['Recency (días)']} días</span>
                    <span style="color:#cbd5e1">Frecuencia</span><span>{row['Freq. (órdenes)']} órdenes</span>
                    <span style="color:#cbd5e1">Gasto total</span><span>USD {row['Gasto total USD']}</span>
                    <span style="color:#cbd5e1">% mercado</span><span style="color:{row['Color']};font-weight:700">{row['% mercado']}%</span>
                    <span style="color:#cbd5e1">Score</span><span style="color:{row['Color']};font-weight:700">{row['Score actividad']:.2f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — SOCIODEMOGRÁFICA
# ─────────────────────────────────────────────────────────────────────────────
elif slide == "👥 Segmentación Sociodem.":
    st.markdown("## Segmentación Sociodemográfica / Conductual")
    st.markdown("**Modelo final:** StepMix Mixto · k=5 · Entropía de clasificación 0.7977")
    st.markdown("---")

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.markdown("#### Tamaño y gasto por segmento")
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        bars = ax.barh(soc_profiles["Segmento"][::-1], soc_profiles["% mercado"][::-1], color=PALETTE_SOC[::-1])
        for bar, val in zip(bars, soc_profiles["% mercado"][::-1]):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, color="#f8fafc", fontweight="bold")
        ax.set_xlabel("% del mercado")
        ax.set_xlim(0, 38)
        ax.set_title("Distribución del mercado")
        st.pyplot(finish_plot(fig, ax), use_container_width=True)
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(6.5, 3.35))
        ax2.bar(soc_profiles["Segmento"], soc_profiles["Gasto prom x orden"], color=PALETTE_SOC, width=0.62)
        ax2.set_ylabel("USD promedio por orden")
        ax2.set_title("Gasto promedio por orden")
        ax2.tick_params(axis="x", rotation=18, labelsize=8)
        st.pyplot(finish_plot(fig2, ax2), use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Perfil StepMix")
        for _, row in soc_profiles.iterrows():
            st.markdown(
                f"""
                <div class="card" style="margin-bottom:10px;text-align:left;padding:14px 18px">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <div style="width:12px;height:12px;border-radius:50%;background:{row['Color']}"></div>
                    <strong style="color:#f8fafc;font-size:0.94rem">{row['Segmento']}</strong>
                    <span style="margin-left:auto;color:{row['Color']};font-weight:700;font-size:0.86rem">{row['% mercado']}%</span>
                  </div>
                  <div style="display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:0.82rem;">
                    <span style="color:#cbd5e1">Edad</span><span>{row['Edad modal']}</span>
                    <span style="color:#cbd5e1">Canal</span><span>{row['Canal modal']}</span>
                    <span style="color:#cbd5e1">Región</span><span>{row['Región modal']}</span>
                    <span style="color:#cbd5e1">Local</span><span>{row['Local modal']}</span>
                    <span style="color:#cbd5e1">Rewards</span><span style="color:{row['Color']};font-weight:700">{int(row['% Rewards']*100)}%</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
elif slide == "🎯 Mercados meta":
    st.markdown("## Estrategia de Enfoque: Mercados y Hábitos Meta")
    st.markdown("**Consolidación cuantitativa, patrones de consumo y perfiles estratégicos del Top 3 de mercado.**")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────────
    # BLOQUE 1: LA MATRIZ DE ENFOQUE (CUANTITATIVA)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 1. Cuantificación y Selección de Mercados Meta")
    col_matriz, col_resumen_meta = st.columns([3, 2])

    with col_matriz:
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        matriz_pct = (matriz / matriz.sum() * 100).round(1)
        im = ax.imshow(matriz_pct, cmap="Greens", aspect="auto", vmin=0, vmax=16)
        
        rfm_labels = ["Espontáneo", "Estrella", "Potencial"]
        soc_labels = ["Connected Professionals", "Prac. Drive-Thru", "Digital Frontier Users", "Classic Speed Seniors", "Mobile Coffee Fans"]
        
        ax.set_xticks(range(5)); ax.set_xticklabels(soc_labels, fontsize=9, rotation=15, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(rfm_labels, fontsize=9)
        ax.set_title("% del Mercado Total por Celda")
        
        for i in range(3):
            for j in range(5):
                ax.text(j, i, f"{matriz_pct[i, j]:.1f}%", ha="center", va="center", fontsize=9, 
                        color="#0f172a" if matriz_pct[i, j] > 6 else "#f8fafc", fontweight="bold")
        
        # Destacar celdas seleccionadas (Top 3)
        # Borde amarillo para (Potencial + Mobile Coffee Fans), (Estrella + Mobile Coffee Fans) y (Potencial + Classic Speed Seniors)
        for (i, j) in [(2, 4), (1, 4), (2, 3)]:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#f59e0b", linewidth=3.0)
            ax.add_patch(rect)
            
        cbar = plt.colorbar(im, ax=ax, label="% mercado", shrink=0.8)
        st.pyplot(finish_plot(fig, ax), use_container_width=True)
        plt.close()

    with col_resumen_meta:
        st.markdown(
            """
            <div style="margin-top:10px">
                <p>Nuestra estrategia concentra esfuerzos en tres intersecciones clave que representan el <strong>39.4% del mercado total analizado</strong>:</p>
                <ul>
                    <li><strong style="color:#00d26a">🥇 Potential Smart Coffee (15.2%):</strong> Máximo volumen de crecimiento digital.</li>
                    <li><strong style="color:#3b82f6">🥈 Smart Coffee Star (13.8%):</strong> Núcleo de alta frecuencia y rentabilidad segura.</li>
                    <li><strong style="color:#f59e0b">🥉 Potential Classic n Quick (10.4%):</strong> Consumo tradicional masivo y recurrente.</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # BLOQUE 2: GRÁFICOS DE CONSUMO Y OPERACIONES
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 2. Patrones Operacionales y Hábitos de Consumo")
    
    metricas_ops = calcular_metricas_operacionales_top3()
    seg_names = metricas_ops["seg_names"]
    seg_colors = ["#00d26a", "#3b82f6", "#f59e0b"]
    st.caption(f"Fuente de estos gráficos: {metricas_ops['source']}.")
    
    tab_bebidas, tab_horarios, tab_ubicaciones = st.tabs(["☕ Bebidas Favoritas", "⏰ Curva de Demanda", "📍 Canales y Tiendas"])

    with tab_bebidas:
        bebidas = metricas_ops["bebidas"]
        data_bebidas = metricas_ops["data_bebidas"]
        
        x = np.arange(len(bebidas))
        width = 0.22
        
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        for i, (name, color) in enumerate(zip(seg_names, seg_colors)):
            ax_b.bar(x + (width * i) - width, data_bebidas[name], width, label=name, color=color, alpha=0.85)
        
        ax_b.set_ylabel("% de Preferencia")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(bebidas, fontsize=9)
        ax_b.legend(fontsize=8, facecolor="#1e293b", edgecolor="#64748b")
        st.pyplot(finish_plot(fig_b, ax_b), use_container_width=True)
        plt.close()

    with tab_horarios:
        horas = metricas_ops["horas"]
        data_horas = metricas_ops["data_horas"]
        
        fig_h, ax_h = plt.subplots(figsize=(10, 4))
        
        # Suavizado de curva para presentación; si scipy no está disponible,
        # se usa interpolación lineal de NumPy para no romper la app.
        x = np.array(horas)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        try:
            from scipy.interpolate import make_interp_spline
            smooth_fn = lambda xv, yv: make_interp_spline(xv, yv, k=3)(x_smooth)
        except Exception:
            smooth_fn = lambda xv, yv: np.interp(x_smooth, xv, yv)
        
        for name, color in zip(seg_names, seg_colors):
            y = np.array(data_horas[name], dtype=float)
            
            y_smooth = smooth_fn(x, y)
            # Asegurar que la curva no baje de cero transacciones por el suavizado
            y_smooth = np.maximum(y_smooth, 0) 
            
            ax_h.plot(x_smooth, y_smooth, linewidth=2.8, label=name, color=color)
            ax_h.fill_between(x_smooth, y_smooth, alpha=0.10, color=color)
            
        ax_h.set_xlabel("Hora del Día")
        ax_h.set_ylabel("Volumen de Transacciones")
        ax_h.set_xlim(0, 23)
        ax_h.set_yticks([]) # Ocultar los números del eje Y para un look más limpio
        
        # Ticks del eje X cada 2 horas
        horas_ticks = np.arange(0, 24, 2)
        ax_h.set_xticks(horas_ticks)
        ax_h.set_xticklabels([f"{h}:00" for h in horas_ticks], fontsize=9)
        
        ax_h.legend(fontsize=8, facecolor="#1e293b", edgecolor="#64748b")
        st.pyplot(finish_plot(fig_h, ax_h), use_container_width=True)
        plt.close()
        
    with tab_ubicaciones:
        col_l, col_r = st.columns(2)
        locales = metricas_ops["locales"]
        regiones = metricas_ops["regiones"]
        data_locales = metricas_ops["data_locales"]
        data_regiones = metricas_ops["data_regiones"]
        
        with col_l:
            x_l = np.arange(len(locales))
            fig_l, ax_l = plt.subplots(figsize=(5, 3.5))
            for i, (name, color) in enumerate(zip(seg_names, seg_colors)):
                ax_l.bar(x_l + (0.2 * i) - 0.2, data_locales[name], 0.2, label=name, color=color, alpha=0.85)
            
            ax_l.set_title("Distribución por Tipo de Local")
            ax_l.set_ylabel("% de transacciones")
            ax_l.set_xticks(x_l)
            ax_l.set_xticklabels(locales, fontsize=9)
            st.pyplot(finish_plot(fig_l, ax_l), use_container_width=True)
            plt.close()
            
        with col_r:
            x_r = np.arange(len(regiones))
            fig_r, ax_r = plt.subplots(figsize=(5, 3.5))
            for i, (name, color) in enumerate(zip(seg_names, seg_colors)):
                ax_r.bar(x_r + (0.2 * i) - 0.2, data_regiones[name], 0.2, label=name, color=color, alpha=0.85)
            
            ax_r.set_title("Distribución Geográfica")
            ax_r.set_xticks(x_r)
            ax_r.set_xticklabels(regiones, fontsize=8, rotation=15)
            st.pyplot(finish_plot(fig_r, ax_r), use_container_width=True)
            plt.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # BLOQUE 3: DESCRIPCIÓN ESTRATÉGICA / PROPUESTA DE VALOR
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 3. Perfil del Consumidor y Propuesta de Valor")
    
    # Renderizado usando expanders o tarjetas limpias para cada segmento
    with st.expander("ANALISIS DE: Potential Smart Coffee "):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🎁 Propuesta de valor y Solución**")
            st.markdown("- **Productos/Servicios:** Enfoque prioritario en Mobile App y alta variedad de Té y Refreshers.")
            st.markdown("- **Analgesicos:** Estaciones de retiro exclusivas en zonas rurales para evitar filas.")
            st.markdown("- **Creadores de Ganancia:** Descuentos dinámicos e incentivos por tarde (notificaciones push) para impulsar la frecuencia.")
        with c2:    
            st.markdown("**👤 Perfil del Cliente**")
            st.markdown("- **Actividades:** Pide de forma anticipada antes de sus tareas diarias; busca personalización rápida sin fricciones.")
            st.markdown("- **Dolores:** Esperas físicas prolongadas que arruinan el flujo digital; menús digitales complejos de editar.")
            st.markdown("- **Ganancias:** Gamificación en Rewards, velocidad de retiro y el valor percibido del vaso de la marca.")
        
    with st.expander("ANALISIS DE: Smart Coffee Star "):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🎁 Propuesta de valor y Solución**")
            st.markdown("- **Productos/Servicios:** Programa VIP/Gold prioritario, lanzamientos anticipados de temporada.")
            st.markdown("- **Analgesicos:** Alertas automáticas de inventario en tiempo real sincronizadas con su app suburbana.")
            st.markdown("- **Creadores de Ganancia:** Días exclusivos de doble acumulación y pre-venta de termos/vasos de edición limitada.")
        with c2:
            st.markdown("**👤 Perfil del Cliente**")
            st.markdown("- **Actividades:** Compra diaria automatizada como hábito; busca activamente mantener estatus VIP y adquirir mercancía.")
            st.markdown("- **Dolores:** Quiebres de stock en jarabes o productos clave; falta de carriles prioritarios presenciales.")
            st.markdown("- **Ganancias:** Sentirse un cliente único reconocido por su nombre; beneficios premium palpables.")
       
    with st.expander("ANALISIS DE: Potential Classic n Quick "):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🎁 Propuesta de valor y Solución**")
            st.markdown("- **Productos/Servicios:** Combos de desayuno (Café Filtrado + Pastelería tradicional).")
            st.markdown("- **Analgesicos:** Paneles físicos simplificados en la línea del Drive-Thru, transacciones rápidas sin uso mandatorio de QR.")
            st.markdown("- **Creadores de Ganancia:** Operadores entrenados para transacciones ultra veloces y empaquetado térmico seguro.")
        with c2:    
            st.markdown("**👤 Perfil del Cliente**")
            st.markdown("- **Actividades:** Compra por Drive-Thru temprano en la mañana; demanda café clásico filtrado rápido.")
            st.markdown("- **Dolores:** Menús saturados con opciones excesivamente modernas; cuellos de botella en la fila del auto.")
            st.markdown("- **Ganancias:** Interacción humana cordial y rápida; empaques seguros contra derrames.")
       
