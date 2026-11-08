import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================

st.set_page_config(
    page_title="Forecasting Ventas · Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .dashboard-header h1 { margin: 0; font-size: 1.8rem; }
    .dashboard-header p  { margin: 6px 0 0; opacity: 0.9; font-size: 1rem; }
    .scenario-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        height: 100%;
    }
    .scenario-active { border: 2px solid #667eea; }
    .scenario-inactive { border-left: 4px solid #ccc; }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 700;
        font-size: 1rem;
        padding: 12px;
        border-radius: 8px;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# RUTAS
# =========================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_final.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "inferencia_df_transformado.csv")

# =========================================================
# CARGA (cacheada)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

try:
    model   = load_model()
    df_full = load_data()
    _ok     = True
except Exception as _err:
    _ok      = False
    _err_msg = str(_err)

# =========================================================
# MOTOR DE PREDICCIÓN RECURSIVA
# =========================================================

def predecir_producto(df_prod, modelo, descuento_ajuste_pp, factor_competencia):
    """
    Predice unidades vendidas día a día para noviembre actualizando lags.

    descuento_ajuste_pp : puntos porcentuales a añadir al descuento_porcentaje
    factor_competencia  : multiplicador sobre precio_competencia (0.95 / 1.0 / 1.05)
    """
    df = df_prod.copy().sort_values("fecha").reset_index(drop=True)

    # --- Recalcular variables dependientes de los controles ---
    df["descuento_porcentaje"] = df["descuento_porcentaje"] + descuento_ajuste_pp
    df["precio_venta"]         = df["precio_base"] * (1 + df["descuento_porcentaje"] / 100)
    df["precio_competencia"]   = df["precio_competencia"] * factor_competencia
    df["ratio_precio"]         = df["precio_venta"] / df["precio_competencia"]

    feature_cols = list(modelo.feature_names_in_)

    # Ventana deslizante: los 7 valores más recientes, del más nuevo al más antiguo
    # Para el día 1, se inicializa con lag_1..lag_7 del propio archivo (calculados desde oct.)
    row0         = df.iloc[0]
    recent_vals  = [row0[f"lag_{i}"] for i in range(1, 8)]

    predictions  = []

    for i in range(len(df)):
        if i > 0:
            # Actualizar lags con las predicciones previas
            for j in range(1, 8):
                df.at[i, f"lag_{j}"] = recent_vals[j - 1]
            # Actualizar media móvil 7 días
            df.at[i, "media_movil_7"] = float(np.mean(recent_vals))

        X    = df.loc[[i], feature_cols]
        pred = max(0.0, float(modelo.predict(X)[0]))
        predictions.append(pred)

        # Deslizar la ventana: prepend predicción, descartar el valor más antiguo
        recent_vals = [pred] + recent_vals[:6]

    df["unidades_predichas"]  = predictions
    df["ingresos_predichos"]  = df["unidades_predichas"] * df["precio_venta"]

    return df

# =========================================================
# SIDEBAR
# =========================================================

FACTOR_MAP = {"Actual (0%)": 1.0, "Competencia -5%": 0.95, "Competencia +5%": 1.05}
DIAS_ES    = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
              4: "Viernes", 5: "Sábado", 6: "Domingo"}

if _ok:
    productos = sorted(df_full["nombre"].unique())
else:
    productos = []

with st.sidebar:
    st.markdown("## 🎛️ Controles de Simulación")
    st.markdown("---")

    producto_sel = st.selectbox(
        "📦 Producto",
        productos,
        help="Selecciona el producto a simular",
    )

    st.markdown("**💰 Ajuste de Descuento**")
    st.caption("Puntos porcentuales sobre el descuento actual")
    descuento_ajuste = st.slider(
        "Descuento",
        min_value=-50, max_value=50, value=0, step=5,
        format="%d pp",
        label_visibility="collapsed",
    )

    st.markdown("**🏪 Escenario de Competencia**")
    escenario = st.radio(
        "Competencia",
        list(FACTOR_MAP.keys()),
        label_visibility="collapsed",
        help="Variación porcentual sobre el precio de la competencia",
    )

    st.markdown("---")
    simular = st.button("🚀 Simular Ventas", use_container_width=True)

    st.markdown("---")
    st.caption("📅 Noviembre 2025 · 30 días · 24 productos")
    st.caption("🤖 HistGradientBoostingRegressor")

# =========================================================
# ZONA PRINCIPAL — HEADER
# =========================================================

if not _ok:
    st.error(f"❌ Error cargando archivos: {_err_msg}")
    st.info(
        "Verifica que existen:\n"
        "- `models/modelo_final.joblib`\n"
        "- `data/processed/inferencia_df_transformado.csv`"
    )
    st.stop()

st.markdown(f"""
<div class="dashboard-header">
    <h1>📊 Simulador de Ventas · Noviembre 2025</h1>
    <p>Predicciones recursivas con actualización de lags día a día</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# ESTADO DE SESIÓN
# =========================================================

if simular:
    factor     = FACTOR_MAP[escenario]
    df_prod    = df_full[df_full["nombre"] == producto_sel].copy()

    with st.spinner(f"⏳ Calculando predicciones recursivas para **{producto_sel}**…"):
        df_res  = predecir_producto(df_prod, model, descuento_ajuste, factor)
        df_esc  = {
            nombre: predecir_producto(df_prod, model, descuento_ajuste, f)
            for nombre, f in FACTOR_MAP.items()
        }

    st.session_state["df_res"]           = df_res
    st.session_state["df_esc"]           = df_esc
    st.session_state["producto_activo"]  = producto_sel
    st.session_state["descuento_activo"] = descuento_ajuste
    st.session_state["escenario_activo"] = escenario

if "df_res" not in st.session_state:
    st.info("👈 Selecciona un producto en el panel lateral y pulsa **🚀 Simular Ventas**.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Productos disponibles", "24")
    with col2:
        st.metric("📅 Días a predecir", "30")
    with col3:
        st.metric("🔄 Método", "Recursivo")
    st.stop()

# Recuperar resultados
df_res    = st.session_state["df_res"]
df_esc    = st.session_state["df_esc"]
prod_nom  = st.session_state["producto_activo"]
desc_act  = st.session_state["descuento_activo"]
esc_act   = st.session_state["escenario_activo"]

# =========================================================
# CABECERA DE PRODUCTO
# =========================================================

st.markdown(f"## 📦 {prod_nom}")
cat  = df_res["categoria"].iloc[0]
sub  = df_res["subcategoria"].iloc[0]
desc_label = f"{desc_act:+d} pp" if desc_act != 0 else "sin ajuste"
st.caption(f"🏷️ {cat} › {sub} · Descuento: {desc_label} · Escenario: {esc_act}")

st.markdown("---")

# =========================================================
# KPIs
# =========================================================

st.subheader("📊 Indicadores Clave de Noviembre 2025")

units_total    = int(df_res["unidades_predichas"].sum())
ingresos_total = df_res["ingresos_predichos"].sum()
precio_medio   = df_res["precio_venta"].mean()
desc_medio     = df_res["descuento_porcentaje"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Unidades Totales",      f"{units_total:,}")
c2.metric("💶 Ingresos Proyectados",  f"€{ingresos_total:,.2f}")
c3.metric("🏷️ Precio Promedio",       f"€{precio_medio:.2f}")
c4.metric("🎁 Descuento Promedio",    f"{desc_medio:.1f} pp")

st.markdown("---")

# =========================================================
# GRÁFICO DE PREDICCIÓN DIARIA
# =========================================================

st.subheader("📈 Predicción Diaria de Unidades Vendidas")

BLACK_FRIDAY_DAY = 28

dias     = df_res["fecha"].dt.day.values
unidades = df_res["unidades_predichas"].values
bf_u     = unidades[BLACK_FRIDAY_DAY - 1]
y_max    = unidades.max()
y_margin = max(y_max * 0.12, 1)

sns.set_theme(style="whitegrid", palette="muted")
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#f9f9fb")
ax.set_facecolor("#f9f9fb")

# Línea principal
sns.lineplot(
    x=dias, y=unidades,
    ax=ax,
    color="#667eea",
    linewidth=2.5,
    marker="o",
    markersize=5,
    markerfacecolor="#667eea",
    markeredgewidth=0,
)

# Línea vertical Black Friday
ax.axvline(x=BLACK_FRIDAY_DAY, color="#e74c3c", linestyle="--", linewidth=1.8, alpha=0.75)

# Punto destacado Black Friday
ax.scatter([BLACK_FRIDAY_DAY], [bf_u], color="#e74c3c", s=160, zorder=6, edgecolors="white", linewidths=1.5)

# Anotación Black Friday
xt = BLACK_FRIDAY_DAY - 3 if BLACK_FRIDAY_DAY > 15 else BLACK_FRIDAY_DAY + 2
yt = bf_u + y_margin
ax.annotate(
    f"🖤 Black Friday\n{bf_u:.0f} uds.",
    xy=(BLACK_FRIDAY_DAY, bf_u),
    xytext=(xt, yt),
    fontsize=9,
    fontweight="bold",
    color="#e74c3c",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.85),
)

ax.set_xlabel("Día de Noviembre 2025", fontsize=11, labelpad=8)
ax.set_ylabel("Unidades Vendidas", fontsize=11, labelpad=8)
ax.set_title(f"Predicción de Ventas — {prod_nom}", fontsize=13, fontweight="bold", color="#333", pad=14)
ax.set_xticks(range(1, 31))
ax.set_xlim(0.5, 30.5)
ax.set_ylim(bottom=0)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", labelsize=9)
sns.despine(ax=ax, left=True)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# =========================================================
# TABLA DETALLADA
# =========================================================

st.subheader("📋 Detalle Diario — Noviembre 2025")

df_tabla = (
    df_res[["fecha", "precio_venta", "precio_competencia",
            "descuento_porcentaje", "unidades_predichas", "ingresos_predichos"]]
    .copy()
)
df_tabla["dia_semana"] = df_tabla["fecha"].dt.dayofweek.map(DIAS_ES)
df_tabla["es_bf"]      = df_tabla["fecha"].dt.day == BLACK_FRIDAY_DAY

tabla_display = pd.DataFrame({
    "Fecha":          df_tabla["fecha"].dt.strftime("%d/%m/%Y"),
    "Día":            df_tabla["dia_semana"],
    "Precio Venta":   df_tabla["precio_venta"].map(lambda x: f"€{x:.2f}"),
    "Precio Compet.": df_tabla["precio_competencia"].map(lambda x: f"€{x:.2f}"),
    "Descuento":      df_tabla["descuento_porcentaje"].map(lambda x: f"{x:.1f} pp"),
    "Unidades":       df_tabla["unidades_predichas"].map(lambda x: f"{x:.0f}"),
    "Ingresos":       df_tabla["ingresos_predichos"].map(lambda x: f"€{x:.2f}"),
    "Evento":         df_tabla["es_bf"].map(lambda x: "🖤 BLACK FRIDAY" if x else ""),
})

st.dataframe(tabla_display, use_container_width=True, height=420, hide_index=True)

st.markdown("---")

# =========================================================
# COMPARATIVA DE ESCENARIOS
# =========================================================

st.subheader("🔄 Comparativa de Escenarios de Competencia")
st.caption(f"Descuento fijo: {desc_act:+d} pp · Solo varía el precio de la competencia")

ESC_META = [
    ("Actual (0%)",    "🟡", "#f39c12", "Sin cambios en la competencia"),
    ("Competencia -5%", "🔴", "#e74c3c", "Competencia baja precios un 5%"),
    ("Competencia +5%", "🟢", "#27ae60", "Competencia sube precios un 5%"),
]

cols = st.columns(3)
for col, (nombre, emoji, color, desc_esc) in zip(cols, ESC_META):
    d          = df_esc[nombre]
    u_esc      = int(d["unidades_predichas"].sum())
    i_esc      = d["ingresos_predichos"].sum()
    es_actual  = (nombre == esc_act)
    borde      = f"border: 2.5px solid {color};" if es_actual else f"border-left: 4px solid {color};"

    with col:
        tag_activo = '<p style="color:#667eea; font-weight:700; font-size:0.85rem; margin:8px 0 0;">✓ Escenario seleccionado</p>' if es_actual else ""
        st.markdown(f"""
<div class="scenario-card" style="{borde}">
    <h3 style="color:{color}; margin:0 0 6px;">{emoji} {nombre}</h3>
    <p style="color:#666; font-size:0.82rem; margin:0 0 14px;">{desc_esc}</p>
    <p style="margin:4px 0;"><strong>Unidades totales:</strong> {u_esc:,}</p>
    <p style="margin:4px 0;"><strong>Ingresos:</strong> €{i_esc:,.2f}</p>
    {tag_activo}
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("🤖 Predicciones con HistGradientBoostingRegressor · Método recursivo · Noviembre 2025")
