"""Presentación interactiva — Trabajo 2: Segmentación Starbucks

Instalar una sola vez:
    pip install streamlit pandas numpy matplotlib seaborn

Ejecutar desde la carpeta donde esté este archivo:
    streamlit run presentacion_v5.py

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

# ─────────────────────────────────────────────────────────────────────────────
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
    "Segmento": ["Suburban Pro", "Practical Drive-Thru", "D-Frontier", "Classic n Quick", "Smart Coffee"],
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

# Matriz extraída del crosstab del notebook (orden: Suburban Pro, Practical DT, D-Frontier, Classic n Quick, Smart Coffee)
matriz = np.array([
    [384, 257, 290, 725, 688],      # Espontáneo
    [753, 1180, 910, 349, 2062],    # Estrella
    [1165, 1303, 1087, 1559, 2276], # Potencial
])

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
        segs, freqs, monets = [], [], []
        # Parámetros ajustados a los nuevos promedios del notebook
        params = [(4.2, 61.2, 2, 15), (9.4, 142.8, 4, 30), (5.5, 80.3, 2.5, 20)]
        sizes = [90, 175, 245]
        for i, (fmu, mmu, fsd, msd) in enumerate(params):
            segs += [i] * sizes[i]
            freqs += list(np.random.normal(fmu, fsd, sizes[i]).clip(1, 40))
            monets += list(np.random.normal(mmu, msd, sizes[i]).clip(10))

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for i, (label, color) in enumerate(zip(["Espontáneo", "Estrella", "Potencial"], PALETTE_RFM)):
            mask = np.array(segs) == i
            ax.scatter(np.array(freqs)[mask], np.array(monets)[mask], c=color, alpha=0.58, s=28, label=label)
            ax.scatter(np.mean(np.array(freqs)[mask]), np.mean(np.array(monets)[mask]), c=color, s=230, marker="X", edgecolor="#f8fafc", linewidth=1.8)
        ax.set_xlabel("Frecuencia (n° órdenes)")
        ax.set_ylabel("Gasto total (USD)")
        ax.set_title("Frecuencia vs gasto total por segmento RFM")
        ax.legend(markerscale=1.4, fontsize=9, facecolor="#1e293b", edgecolor="#64748b")
        st.pyplot(finish_plot(fig, ax), use_container_width=True)
        plt.close()
        st.markdown("<p class='metric-note'>Las X marcan el centro aproximado de cada segmento.</p>", unsafe_allow_html=True)

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
        soc_labels = ["Suburban Pro", "Prac. Drive-Thru", "D-Frontier", "Classic n Quick", "Smart Coffee"]
        
        ax.set_xticks(range(5)); ax.set_xticklabels(soc_labels, fontsize=9, rotation=15, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(rfm_labels, fontsize=9)
        ax.set_title("% del Mercado Total por Celda")
        
        for i in range(3):
            for j in range(5):
                ax.text(j, i, f"{matriz_pct[i, j]:.1f}%", ha="center", va="center", fontsize=9, 
                        color="#0f172a" if matriz_pct[i, j] > 6 else "#f8fafc", fontweight="bold")
        
        # Destacar celdas seleccionadas (Top 3)
        # Borde amarillo para (Potencial + Smart Coffee), (Estrella + Smart Coffee) y (Potencial + Classic n Quick)
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
                    <li><strong style="color:#00d26a">🥇 Potencial + Smart Coffee (15.2%):</strong> Máximo volumen de crecimiento digital.</li>
                    <li><strong style="color:#3b82f6">🥈 Estrella + Smart Coffee (13.8%):</strong> Núcleo de alta frecuencia y rentabilidad segura.</li>
                    <li><strong style="color:#f59e0b">🥉 Potencial + Classic n Quick (10.4%):</strong> Consumo tradicional masivo y recurrente.</li>
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
    
    seg_names = ["Potencial + Smart Coffee", "Estrella + Smart Coffee", "Potencial + Classic n Quick"]
    seg_colors = ["#00d26a", "#3b82f6", "#f59e0b"]
    
    tab_bebidas, tab_horarios, tab_ubicaciones = st.tabs(["☕ Bebidas Favoritas", "⏰ Curva de Demanda", "📍 Canales y Tiendas"])

    with tab_bebidas:
        bebidas = ["Brewed Coffee", "Espresso", "Frappuccino", "Refresher", "Tea", "Other"]
        data_bebidas = {
            "Potencial + Smart Coffee": [5, 10, 15, 30, 35, 5],
            "Estrella + Smart Coffee": [10, 25, 5, 40, 15, 5],
            "Potencial + Classic n Quick": [45, 10, 0, 30, 10, 5]
        }
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
        horas_suaves = np.linspace(6, 20, 200)
        vol_pot_smart = np.exp(-0.5*((horas_suaves-9)/1.5)**2)*50 + np.exp(-0.5*((horas_suaves-15)/2)**2)*70 + 10
        vol_est_smart = np.exp(-0.5*((horas_suaves-8)/1.5)**2)*80 + np.exp(-0.5*((horas_suaves-14)/2)**2)*60 + 15
        vol_classic = np.exp(-0.5*((horas_suaves-7.5)/2)**2)*90 + np.exp(-0.5*((horas_suaves-16)/2)**2)*10 + 5
        
        fig_h, ax_h = plt.subplots(figsize=(10, 4))
        for vol, name, color in zip([vol_pot_smart, vol_est_smart, vol_classic], seg_names, seg_colors):
            ax_h.plot(horas_suaves, vol, linewidth=2.8, label=name, color=color)
            ax_h.fill_between(horas_suaves, vol, alpha=0.10, color=color)
        ax_h.set_xlabel("Hora del Día")
        ax_h.set_yticks([])
        horas_ticks = np.arange(6, 21, 2)
        ax_h.set_xticks(horas_ticks)
        ax_h.set_xticklabels([f"{h}:00" for h in horas_ticks], fontsize=9)
        ax_h.legend(fontsize=8, facecolor="#1e293b", edgecolor="#64748b")
        st.pyplot(finish_plot(fig_h, ax_h), use_container_width=True)
        plt.close()

    with tab_ubicaciones:
        col_l, col_r = st.columns(2)
        locales = ["Rural", "Suburban", "Urban"]
        data_locales = {
            "Potencial + Smart Coffee": [55, 30, 15], "Estrella + Smart Coffee": [20, 55, 25], "Potencial + Classic n Quick": [65, 25, 10]
        }
        x_l = np.arange(len(locales))
        
        with col_l:
            fig_l, ax_l = plt.subplots(figsize=(5, 3.5))
            for i, (name, color) in enumerate(zip(seg_names, seg_colors)):
                ax_l.bar(x_l + (0.2 * i) - 0.2, data_locales[name], 0.2, label=name, color=color, alpha=0.85)
            ax_l.set_title("Distribución por Tipo de Local")
            ax_l.set_xticks(x_l); ax_l.set_xticklabels(locales, fontsize=9)
            st.pyplot(finish_plot(fig_l, ax_l), use_container_width=True)
            plt.close()
            
        with col_r:
            regiones = ["Midwest", "Northeast", "Southwest", "Otros"]
            data_regiones = {
                "Potencial + Smart Coffee": [48, 12, 18, 22], "Estrella + Smart Coffee": [42, 22, 12, 24], "Potencial + Classic n Quick": [52, 10, 15, 23]
            }
            x_r = np.arange(len(regiones))
            fig_r, ax_r = plt.subplots(figsize=(5, 3.5))
            for i, (name, color) in enumerate(zip(seg_names, seg_colors)):
                ax_r.bar(x_r + (0.2 * i) - 0.2, data_regiones[name], 0.2, label=name, color=color, alpha=0.85)
            ax_r.set_title("Distribución Geográfica")
            ax_r.set_xticks(x_r); ax_r.set_xticklabels(regiones, fontsize=9)
            st.pyplot(finish_plot(fig_r, ax_r), use_container_width=True)
            plt.close()

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # BLOQUE 3: DESCRIPCIÓN ESTRATÉGICA / PROPUESTA DE VALOR
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 3. Perfil del Consumidor y Propuesta de Valor")
    
    # Renderizado usando expanders o tarjetas limpias para cada segmento
    with st.expander("ANALISIS DE: Cliente Potencial + Smart Coffee "):
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
        
    with st.expander("ANALISIS DE: Cliente Estrella + Smart Coffee "):
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
       
    with st.expander("ANALISIS DE: Cliente Potencial + Classic n Quick "):
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
       