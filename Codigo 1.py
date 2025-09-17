import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Configuración de la página
st.set_page_config(
    page_title="Gemelo Digital de Electrolizador PEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .parameter-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .output-box {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .constant-box {
        background: #fef3c7;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Layout principal
col1, col2, col3 = st.columns([1, 1.5, 1])

# Sidebar - Parámetros de entrada
st.sidebar.markdown("Parámetros de Control")

# --- Diccionario que asocia voltajes con sus corrientes disponibles
opciones = {
    5.3: [1.5, 2.05, 2.6],
    5.75: [3.5, 4.0, 4.5],
    6.2: [5.4, 6.0, 6.6]  
}

# Parámetros principales
voltaje = st.sidebar.selectbox( #st.sidebar.select_slider- para hacerlo en barra que no se mueva
    "Voltaje (V)",
    options=list(opciones.keys()),
    format_func=lambda x: f"{x} V",
    help="Voltaje aplicado al electrolizador"
)

corriente = st.sidebar.selectbox(
    "Corriente (A)",
    options=opciones[voltaje],
    format_func=lambda x: f"{x} A",
    help="Corriente eléctrica del proceso"
)

# CSS para deshabilitar el slider
st.markdown("""
<style>
div[data-baseweb="slider"] input {pointer-events: none;}
</style>
""", unsafe_allow_html=True)


temperatura = st.sidebar.slider(
    "Temperatura (°C)", 
    min_value=22, 
    max_value=80, 
    value=22, 
    step=1,
    help="Temperatura de operación",
    disabled=True  # ← bloquea el slider
)


# Valor fijo
valor_fijo = 1.015

presion = st.sidebar.slider(
    "Presión (hectopascales)", 
    min_value=1.015, 
    max_value=30.000, 
    value=1.015, 
    step=0.001,  # ← permite valores como 1.015
    help="Presión del sistema",
    disabled=True  # ← bloquea el slider
)


# Entrada de agua
st.sidebar.markdown("Entrada de Agua")
flujo_h2o = st.sidebar.number_input(
    "Flujo H2O (L/min)", 
    min_value=0.1, 
    max_value=5.0, 
    value=1.5, 
    step=0.1
)


eficiencia = st.sidebar.selectbox(
    "Eficiencia (%)", 
    options=[75, 80, 85, 90, 95], 
    index=2
)

area_electrodo = st.sidebar.number_input(
    "Área del electrodo (cm²)", 
    min_value=10, 
    max_value=500, 
    value=100, 
    step=10
)

# Cálculos del electrolizador
def calcular_produccion(voltaje, corriente, eficiencia, temperatura, presion):
    # Constantes
    F = 96485  # Constante de Faraday (C/mol)
    R = 8.314  # Constante de los gases (J/mol·K)
    
    # Cálculo teórico de producción de hidrógeno (mol/s)
    # H2O → H2 + 1/2 O2 (2 electrones por molécula de H2)
    produccion_h2_mol_s = (corriente * eficiencia / 100) / (2 * F)
    
    # Conversión a L/min (condiciones estándar)
    produccion_h2_l_min = produccion_h2_mol_s * 22.4 * 60 / 1000
    
    # Producción de oxígeno (la mitad que el hidrógeno)
    produccion_o2_l_min = produccion_h2_l_min / 2
    
    # Potencia consumida
    potencia = voltaje * corriente
    
    # Consumo específico de energía (kWh/Nm³ H2)
    if produccion_h2_l_min > 0:
        consumo_especifico = (potencia / 1000) / (produccion_h2_l_min / 1000 * 60)
    else:
        consumo_especifico = 0
    
    return {
        'h2_production': produccion_h2_l_min,
        'o2_production': produccion_o2_l_min,
        'power': potencia,
        'specific_consumption': consumo_especifico,
        'efficiency': eficiencia
    }


# Realizar cálculos
resultados = calcular_produccion(voltaje, corriente, eficiencia, temperatura, presion)

# Obtener la potencia calculada
potencia = round(resultados['power'], 2)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        ...
</style>
""", unsafe_allow_html=True)

# ↓ Cambiar tamaño del texto de las métricas
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# Columna izquierda - Parámetros
with col1:
    st.markdown("""
    <div class="parameter-box">
        <h3>Parámetros Actuales</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Voltaje", f"{voltaje} V")
    st.metric("Corriente", f"{corriente} A")
    st.metric("Temperatura", f"{temperatura} °C")
    st.metric("Presión", f"{presion} hPa")
    st.metric("Potencia", f"{potencia} kW")
    st.metric("Flujo H₂O", f"{flujo_h2o} L/min")
 

    




    # Animación de flujo (burbujas)
# Inicializar estado si no existe
with col2:
    # --- Título centrado ---
    st.markdown("""
    <div class="parameter-box" style="text-align: center;">
        <h3>Electrolizador PEM</h3>
    </div>
    """, unsafe_allow_html=True)

    # --- Botones alineados horizontalmente ---
    col_a, col_b, col_c = st.columns(3)

    # CSS para colorear los botones
    st.markdown("""
    <style>
        div[data-testid="column"]:nth-of-type(1) button {
            background-color: #22c55e !important; /* verde */
            color: white !important;
        }
        div[data-testid="column"]:nth-of-type(2) button {
            background-color: #3b82f6 !important; /* azul */
            color: white !important;
        }
        div[data-testid="column"]:nth-of-type(3) button {
            background-color: #ef4444 !important; /* rojo */
            color: white !important;
        }
        button[kind="secondary"] {
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Inicializar estado si no existe
    if "estado" not in st.session_state:
        st.session_state.estado = "stopped"

    # Botón INICIAR
    if col_a.button("▶️ Iniciar"):
        st.session_state.estado = "running"

    # Botón PAUSAR
    if col_b.button("⏸️ Pausar"):
        st.session_state.estado = "paused"

    # Botón DETENER
    if col_c.button("⏹️ Detener"):
        st.session_state.estado = "stopped"

    # --- Mostrar imagen debajo de los botones ---
    
    st.image("fig.png", caption="Esquema del electrolizador PEM", width=300)

    # --- Estado actual ---
    if st.session_state.estado == "running":
        st.success("Proceso en ejecución...")
    elif st.session_state.estado == "paused":
        st.info("Proceso en pausa...")
    elif st.session_state.estado == "stopped":
        st.warning("Proceso detenido.")


    # Ecuación química
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; text-align: center;">
        <h4>⚗️ Reacción Química</h4>
        <p><strong>2H₂O → 2H₂ + O₂</strong></p>
        <p><em>Electrólisis del agua usando membrana de intercambio protónico</em></p>
    </div>
    """, unsafe_allow_html=True)

# Columna derecha - Salidas y resultados
with col3:
    st.markdown("""
    <div class="output-box">
        <h3>Salidas del Sistema</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric(
        "Producción H₂", 
        f"{resultados['h2_production']:.2f} L/min",
        delta=f"{resultados['h2_production']*60:.1f} L/h"
    )
    
    st.metric(
        "Producción O₂", 
        f"{resultados['o2_production']:.2f} L/min",
        delta=f"{resultados['o2_production']*60:.1f} L/h"
    )
    
    # Indicadores de estado
    #st.markdown("Estado del Sistema")
    
    #if resultados['power'] > 0:
      #  st.success("Sistema Operativo")
    #else:
        #st.error("Sistema Detenido")
    
    # Eficiencia
    if eficiencia >= 85:
        st.success(f"✅ Eficiencia Óptima: {eficiencia}%")
    elif eficiencia >= 75:
        st.warning(f"⚠️ Eficiencia Moderada: {eficiencia}%")
    else:
        st.error(f"❌ Eficiencia Baja: {eficiencia}%")


# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 <strong>Gemelo digital de un electrolizador PEM</strong> | Desarrollado para análisis de electrólisis</p>
    <p><em>Hecho por:</em></p>
    <p><em>Isabela Quintero</em></p>
    <p><em>Johana Andrea Murillo</em></p>
</div>
""", unsafe_allow_html=True)