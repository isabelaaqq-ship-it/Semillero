import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Electrolizador PEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTADO 
if "modelo_seleccionado" not in st.session_state:
    st.session_state.modelo_seleccionado = None

if "datos_validados" not in st.session_state:
    st.session_state.datos_validados = False

if "datos_experimentales" not in st.session_state:
    st.session_state.datos_experimentales = None

if "tabla" not in st.session_state:
    st.session_state.tabla = pd.DataFrame({
        "Voltaje (V)":   [0.0]*15,
        "Corriente (A)": [0.0]*15,
    })

if "estado" not in st.session_state:
    st.session_state.estado = "stopped"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.parameter-box {
    background:#f8fafc; border:2px solid #e2e8f0;
    border-radius:10px; padding:1rem; margin:0.5rem 0;
}
[data-testid="stMetricValue"] { font-size:16px !important; }
[data-testid="stMetricLabel"] { font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("UCO.png", use_container_width=True)
st.sidebar.markdown("### Parámetros de Control")
st.sidebar.markdown("**Ingreso de datos experimentales**")

if st.sidebar.button("🔄 Limpiar tabla"):
    st.session_state.pop("tabla_editor", None)
    st.session_state.tabla = pd.DataFrame({
        "Voltaje (V)":   [0.0]*15,
        "Corriente (A)": [0.0]*15,
    })
    st.rerun()

df_editado = st.sidebar.data_editor(
    st.session_state.tabla,
    key="tabla_editor",
    num_rows="dynamic",
    use_container_width=True,
)
st.session_state.tabla = df_editado

if st.sidebar.button("Validar datos"):
    df_limpio = df_editado[
        (df_editado["Voltaje (V)"] != 0.0) &
        (df_editado["Corriente (A)"] != 0.0)
    ].dropna()

    if len(df_limpio) >= 1:
        st.session_state.datos_experimentales = df_limpio.reset_index(drop=True)
        st.session_state.datos_validados = True
        st.sidebar.success("Datos cargados correctamente")
    else:
        st.sidebar.error(f"Debes ingresar mínimo 15 filas completas (tienes {len(df_limpio)})")

# ── SELECCIÓN DE MODELO ───────────────────────────────────────────────────────
if st.session_state.datos_validados:
    modelo = st.sidebar.selectbox(
        "Seleccione el modelo",
        ["-- Seleccionar --", "Regresión lineal", "Modelo de superficie", "Ley de Faraday"]
    )
    if modelo != "-- Seleccionar --":
        st.session_state.modelo_seleccionado = modelo
    else:
        st.session_state.modelo_seleccionado = None
else:
    st.sidebar.warning("Primero debes ingresar y validar los datos")

# ── PARÁMETROS SOLO PARA LEY DE FARADAY ──────────────────────────────────────
presion     = 1.0
temperatura = 298.0
n_celdas    = 1

if st.session_state.modelo_seleccionado == "Ley de Faraday":
    st.sidebar.markdown("### Parámetros Faraday")
    presion = st.sidebar.number_input(
        "Presión (atm)", value=1.0, step=0.1, format="%.2f"
    )
    temperatura = st.sidebar.number_input(
        "Temperatura (K)", value=298.0, step=1.0, format="%.1f"
    )
    n_celdas = st.sidebar.number_input(
        "Número de celdas", value=1, step=1, min_value=1
    )

# ── FUNCIÓN DE CÁLCULO ────────────────────────────────────────────────────────
def calcular_produccion(modelo, df, n_celdas=1, temperatura=298.0, presion=1.0):
    F    = 96485
    nf   = 0.95
    R    = 0.082057
    V_TN = 1.23  # Voltaje termoneutral mínimo (V)

    V = df["Voltaje (V)"].values
    I = df["Corriente (A)"].values

    V_mean = V.mean()
    I_mean = I.mean()

    # Eficiencia voltaica (común para todos)
    ef_voltaica = min((V_TN / V_mean) * 100, 100) if V_mean > 0 else 0.0

    if modelo == "Regresión lineal":
        reg = LinearRegression()
        reg.fit(I.reshape(-1, 1), V)
        h2 = max(reg.predict([[I_mean]])[0], 0)
        o2 = h2 / 2
        potencia = (V * I).mean()

        # Eficiencia faradaica: H2 real vs H2 teórico por Faraday
        Vm      = (R * 298.0) / 1.0
        h2_teo  = (I_mean) / (2 * F) * Vm * 60
        ef_faradaica = min((h2 / h2_teo) * 100, 100) if h2_teo > 0 else 0.0

    elif modelo == "Modelo de superficie":
        X_train = np.column_stack([V, I, V*I, V**2, I**2])
        y_train = (I) / (2 * F) * 22.4 * 60
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        X_new = np.array([[V_mean, I_mean, V_mean*I_mean, V_mean**2, I_mean**2]])
        h2 = max(reg.predict(X_new)[0], 0)
        o2 = h2 / 2
        potencia = (V * I).mean()

        # Eficiencia faradaica
        Vm      = (R * 298.0) / 1.0
        h2_teo  = (I_mean) / (2 * F) * Vm * 60
        ef_faradaica = min((h2 / h2_teo) * 100, 100) if h2_teo > 0 else 0.0

    elif modelo == "Ley de Faraday":
        Vm   = (R * temperatura) / presion
        h2   = (n_celdas * I_mean) / (2 * F) * Vm * 60 * nf
        o2   = (n_celdas * I_mean) / (4 * F) * Vm * 60 * nf
        potencia = (V * I).mean()

        # Eficiencia faradaica = nf (ya definida como 0.95)
        ef_faradaica = nf * 100

    # Eficiencia global = faradaica × voltaica / 100
    consumo    = (potencia/1000) / (h2*60/1000) if h2 > 0 else None
    eficiencia = (ef_faradaica * ef_voltaica) / 100

    return h2, o2, potencia, consumo, eficiencia

# ── LAYOUT PRINCIPAL ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([0.7, 1.5, 1])

# ── Columna central: SIEMPRE visible ──
with col2:
    st.markdown("""
    <div class="parameter-box" style="text-align:center;">
        <h3>Electrolizador PEM</h3>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    st.markdown("""
    <style>
        div[data-testid="column"]:nth-of-type(1) button {
            background-color: #22c55e !important;
            color: white !important;
        }
        div[data-testid="column"]:nth-of-type(2) button {
            background-color: #3b82f6 !important;
            color: white !important;
        }
        div[data-testid="column"]:nth-of-type(3) button {
            background-color: #ef4444 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if col_a.button("▶️ Iniciar"):
        st.session_state.estado = "running"
    if col_b.button("⏸️ Pausar"):
        st.session_state.estado = "paused"
    if col_c.button("⏹️ Detener"):
        st.session_state.estado = "stopped"

    col_img1, col_img2, col_img3 = st.columns([0.5, 2, 0.5])
    with col_img2:
        if st.session_state.estado == "running":
            st.image("fig.png", caption="▶️ Proceso en ejecución...", use_container_width=True)
        elif st.session_state.estado == "paused":
            st.image("fig.png", caption="⏸️ Proceso en pausa...", use_container_width=True)
        else:
            st.image("fig.png", caption="⏹️ Proceso detenido.", use_container_width=True)

    if st.session_state.estado == "running":
        st.success("▶️ Proceso en ejecución...")
    elif st.session_state.estado == "paused":
        st.info("⏸️ Proceso en pausa...")
    else:
        st.warning("⏹️ Proceso detenido.")

    st.markdown("""
    <div style="background:#f0f9ff;padding:1rem;border-radius:8px;text-align:center;margin-top:1rem;">
        <h4>⚗️ Reacción Química</h4>
        <p><strong>2H₂O → 2H₂ + O₂</strong></p>
        <p><em>Electrólisis del agua usando membrana de intercambio protónico</em></p>
    </div>
    """, unsafe_allow_html=True)

# ── Columnas izquierda y derecha ──────────────────────────────────────────────
if not st.session_state.modelo_seleccionado:
    with col1:
        st.markdown("""
        <div class="parameter-box" style="text-align:center;">
            <h3>Entradas</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("Selecciona un modelo para ver las entradas.")

    with col3:
        st.markdown("""
        <div class="parameter-box" style="text-align:center;">
            <h3>Salidas</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("Selecciona un modelo para ver los resultados.")

else:
    modelo_actual = st.session_state.modelo_seleccionado
    df            = st.session_state.datos_experimentales

    h2, o2, potencia, consumo, eficiencia = calcular_produccion(
        modelo_actual, df, n_celdas, temperatura, presion
    )

    with col1:
        st.markdown("""
        <div class="parameter-box" style="text-align:center;">
            <h3>Entradas</h3>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Modelo",             modelo_actual)
        st.metric("Voltaje promedio",   f"{df['Voltaje (V)'].mean():.3f} V")
        st.metric("Corriente promedio", f"{df['Corriente (A)'].mean():.3f} A")
        st.metric("Nº datos",           str(len(df)))

        if modelo_actual == "Ley de Faraday":
            st.metric("Celdas",      str(int(n_celdas)))
            st.metric("Temperatura", f"{temperatura} K")
            st.metric("Presión",     f"{presion} atm")

    with col3:
        st.markdown("""
        <div class="parameter-box" style="text-align:center;">
            <h3>Salidas</h3>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Producción H₂", f"{h2:.4f} L/min")
        st.metric("Producción O₂", f"{o2:.4f} L/min")
        st.metric("Eficiencia global", f"{eficiencia:.1f} %")

        if potencia is not None:
            st.metric("Potencia", f"{potencia:.2f} W")
        else:
            st.metric("Potencia", "No aplica")

        if consumo is not None:
            st.metric("Consumo", f"{consumo:.4f} kWh/Nm³")
        else:
            st.metric("Consumo", "No aplica")

        if eficiencia >= 70:
            st.success("✅ Eficiencia Óptima")
        elif eficiencia >= 50:
            st.warning("⚠️ Eficiencia Moderada")
        else:
            st.error("❌ Eficiencia Baja")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align:center; color:#666; padding:1rem;">
    <p>🔬 <strong>Gemelo digital de un electrolizador PEM</strong> | Análisis de electrólisis</p>
    <p><em>Universidad Católica de Oriente</em><br>
    Isabela Quintero Arboleda · Johana Andrea Murillo Vergara</p>
</div>
""", unsafe_allow_html=True)