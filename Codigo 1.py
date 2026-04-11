import streamlit as st
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Electrolizador PEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

#ESTADO
if "modelo_seleccionado" not in st.session_state:
    st.session_state.modelo_seleccionado = None

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
            
                /* Centrar imagen */
    .stImage {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Layout principal
col1, col2, col3 = st.columns([0.7, 1.5, 1])


st.sidebar.image("UCO.png", use_container_width=True)
# Sidebar - Parámetros de entrada
st.sidebar.markdown("Parámetros de Control")

#Paso 1: elegir modelo
modelo = st.sidebar.selectbox(
    "Seleccione el modelo",
    ["-- Seleccionar --", "Regresión lineal", "Modelo de superficie", "Ley de Faraday"]
)

if modelo != "-- Seleccionar --":
    st.session_state.modelo_seleccionado = modelo


voltaje = 0.0
corriente = 0.0
n_celdas = 1

if st.session_state.modelo_seleccionado:

    modelo = st.session_state.modelo_seleccionado

    if modelo == "Regresión lineal":
        corriente = st.sidebar.number_input(
            "Corriente (A)",
            value=1.0,
            step=0.1,
            format="%.3f"
        )

    elif modelo == "Modelo de superficie":
        voltaje = st.sidebar.number_input(
            "Voltaje (V)",
            value=5.5,
            step=0.1,
            format="%.3f"
        )

        corriente = st.sidebar.number_input(
            "Corriente (A)",
            value=1.0,
            step=0.1,
            format="%.3f"
        )

        st.sidebar.caption("Rango recomendado: 5V - 6.5V")

    elif modelo == "Ley de Faraday":
        voltaje = st.sidebar.number_input(
            "Voltaje (V)",
            value=5.5,
            step=0.1,
            format="%.3f"
        )

        corriente = st.sidebar.number_input(
            "Corriente (A)",
            value=1.0,
            step=0.1,
            format="%.3f"
        )

        n_celdas = st.sidebar.number_input(
            "Número de celdas",
            value=1,
            step=1
        )

# FUNCIÓN DE CÁLCULO MODELOS
def calcular_produccion(modelo, voltaje, corriente, n_celdas):
    F = 96485

    if modelo == "Regresión lineal":
        B0 = -2.0361339880739 
        B1 = 30.4014883456403000
        h2 = B0 + B1 * corriente

    elif modelo == "Modelo de superficie":
        V = voltaje
        A = corriente

        h2 = (
            3098.97 
            - 1261.98 * V
            + 277.01 * A
            - 52.6 * V * A
            + 129.11 * V**2
            + 6.69 * A**2
        ) 

    elif modelo == "Ley de Faraday":
        h2 = (corriente * n_celdas) / (2 * F)
        h2 = h2 * 22.4 * 60 
    o2 = h2 / 2

    if modelo == "Regresión lineal":
        potencia = None
        consumo = None

    elif modelo == "Modelo de superficie":
        potencia = voltaje * corriente

    elif modelo == "Ley de Faraday":
        potencia = voltaje * corriente

    if potencia and h2 > 0:
        consumo = (potencia / 1000) / (h2 * 60 / 1000)
    else:
        consumo = None

    eficiencia = 70
    return h2, o2, potencia, consumo, eficiencia

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

# 🔹 Pantalla inicial
if not st.session_state.modelo_seleccionado:
    st.markdown("""
    <div style="text-align:center; padding:60px;">
        <h2>🔬 Gemelo Digital - Electrolizador PEM</h2>
        <p>Selecciona un modelo en el panel izquierdo para comenzar</p>
    </div>
    """, unsafe_allow_html=True)

# 🔹 Cuando ya eligió modelo
else:
    modelo = st.session_state.modelo_seleccionado
    h2, o2, potencia, consumo, eficiencia = calcular_produccion(
        modelo, voltaje, corriente, n_celdas
    )
    
# Columna izquierda - Parámetros
with col1:
        st.markdown("""
        <div class="parameter-box" style="text-align: center;">
            <h3>Entradas</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Modelo", modelo)

        if modelo == "Regresión lineal":
            st.metric("Corriente", f"{corriente} A")

        elif modelo == "Modelo de superficie":
            st.metric("Voltaje", f"{voltaje} V")
            st.metric("Corriente", f"{corriente} A")

        elif modelo == "Ley de Faraday":
            st.metric("Voltaje", f"{voltaje} V")
            st.metric("Corriente", f"{corriente} A")
            st.metric("Celdas", n_celdas)

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
        
        col_img1, col_img2, col_img3 = st.columns([0.5, 2, 0.5])
with col_img2:
        st.image("fig.png", caption="Esquema del electrolizador PEM", use_container_width=True)

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
if st.session_state.modelo_seleccionado:
    with col3:
        st.markdown("""
        <div class="parameter-box" style="text-align: center;">
            <h3>Salidas</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Producción H₂", f"{h2:.4f} L/min")
        st.metric("Producción O₂", f"{o2:.4f} L/min")
        st.metric("Eficiencia", f"{eficiencia} %")

        if potencia is not None:
            st.metric("Potencia", f"{potencia:.2f} W")
        else:
            st.metric("Potencia", "No aplica")

        if consumo is not None:
            st.metric("Consumo", f"{consumo:.2f} kWh/Nm³")
        else:
            st.metric("Consumo", "No aplica")

        if eficiencia >= 70:
            st.success("Eficiencia Óptima")
        elif eficiencia >= 50:
            st.warning("Eficiencia Moderada")
        else:
            st.error("Eficiencia Baja")
else:
    with col3:
        st.markdown("""
        <div class="parameter-box" style="text-align: center;">
            <h3>Salidas</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("Selecciona un modelo en el panel izquierdo para ver los resultados.")


# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 <strong>Gemelo digital de un electrolizador PEM</strong> | Desarrollado para análisis de electrólisis</p>
    <p><em>Hecho por:</em></p>
    <p><em>Universidad Católica de Oriente</em></p>
    <p><em>Isabela Quintero Arboleda</em></p>
    <p><em>Johana Andrea Murillo Vergara</em></p>
</div>
""", unsafe_allow_html=True)