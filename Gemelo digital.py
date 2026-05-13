import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Gemelo digital electrolizador PEM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background-color: #008b50;
        border-color: #024426;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #024426;
        border-color: #024426;
    }
    .stProgress > div > div > div > div {
        background-color: #008b50 !important;
    }
    .seccion-titulo {
        background: linear-gradient(90deg, #f0fdf4, #ffffff);
        border-left: 4px solid #008b50;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.4rem 0 0.8rem 0;
        font-weight: 600;
        color: #14532d;
        font-size: 1.05rem;
    }
    .stAlert[data-baseweb="notification"] p {
        font-weight: normal;
    }
    .r2-verde {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        color: #14532d;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .ef-card {
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .ef-alta   { background:#f0fdf4; border:1px solid #86efac; color:#14532d; }
    .ef-media  { background:#fffbeb; border:1px solid #fde68a; color:#92400e; }
    .ef-baja   { background:#fff1f2; border:1px solid #fecdd3; color:#9f1239; }
</style>
""", unsafe_allow_html=True)

defaults = {
    'paso_actual': 1,
    'estado': "stopped",
    'data': pd.DataFrame(columns=["voltaje", "corriente", "tasa"]),
    'celdas': 1,
    'temperatura': 298.0,
    'presion': 0.99,
    'popt': None,
    'modelo_seleccionado': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def siguiente_paso():
    st.session_state.paso_actual += 1


def paso_anterior():
    st.session_state.paso_actual -= 1


def reiniciar():
    st.session_state.clear()
    st.session_state.paso_actual = 1


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════

def _reg_lineal(I_arr, tasa_arr):
    n = len(I_arr)
    SX, SY = np.sum(I_arr), np.sum(tasa_arr)
    SXY, SX2 = np.sum(I_arr * tasa_arr), np.sum(I_arr ** 2)
    denom = n * SX2 - SX ** 2
    b1 = (n * SXY - SX * SY) / denom if denom != 0 else 0.0
    b0 = SY / n - b1 * SX / n
    return b0, b1


def _modelo_superficie(V_arr, I_arr, tasa_arr):
    ones = np.ones(len(V_arr))
    X = np.column_stack([ones, V_arr, I_arr, V_arr * I_arr, V_arr*2, I_arr*2])
    Y = tasa_arr.reshape(-1, 1)
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta.flatten(), X


def _nf_faraday(I_arr, tasa_arr, n_celdas, T, P):
    """
    Calcula la eficiencia faradaica comparando la tasa real (mL/min)
    con la tasa teórica (mL/min).

    Unidades:
        R  = 0.08314  L·atm / (mol·K)
        Vm = R·T/P    L/mol   (volumen molar del gas a T,P)
        tasa_teo [mL/min] = n_celdas · (I/2F) [mol/s] · Vm [L/mol] · 60 [s/min] · 1000 [mL/L]
        tasa_real [mL/min] = tasa_arr  (dato experimental)
    """
    R, F = 0.08314, 96485
    Vm = (R * T) / P                            # L/mol  ← sin ×1000
    # CORRECCIÓN: tasa_teo en mL/min (mismas unidades que tasa_arr)
    tasa_teo = n_celdas * (I_arr / (2 * F)) * Vm * 60 * 1000   # mL/min
    tasa_real = tasa_arr                                          # mL/min

    nf_arr = np.where(tasa_teo > 0, tasa_real / tasa_teo, 0.0)
    nf_arr = np.clip(nf_arr, 0.0, 1.0)
    return float(nf_arr.mean()), nf_arr, tasa_teo, tasa_real, Vm


def _accuracy(pred, real):
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.where(real > 0, np.abs(pred - real) / real, 0.0)
    acc = np.clip(1.0 - err, 0.0, 1.0)
    return err, acc, float(err.mean()), float(acc.mean())


def calcular_produccion(modelo, data_limpia, n_celdas, T, P):
    F, V_TN = 96485, 1.23
    I    = data_limpia["corriente"].values.astype(float)
    V    = data_limpia["voltaje"].values.astype(float)
    tasa = data_limpia["tasa"].values.astype(float)
    h2     = 0.0
    o2     = 0.0
    ef_far = 0.0   # inicializado aquí para evitar UnboundLocalError si un bloque interno falla

    potencia = float(np.mean(V * I))
    V_mean   = V.mean()
    ef_volt  = min((V_TN / V_mean) * 100, 100) if V_mean > 0 else 0.0
    coefs    = {}

    if modelo == "Regresión lineal":
        b0, b1 = _reg_lineal(I, tasa)
        pred   = b0 + b1 * I
        h2     = max(float(pred.mean()), 0.0)
        o2     = h2 / 2
        err, acc, err_m, acc_m = _accuracy(pred, tasa)
        ef_far = acc_m * 100
        coefs  = {"β₀": b0, "β₁": b1, "pred": pred,
                  "err": err, "acc": acc, "err_m": err_m, "acc_m": acc_m}

    elif modelo == "Modelo de superficie":
        beta, X = _modelo_superficie(V, I, tasa)
        pred    = X @ beta
        h2      = max(float(pred.mean()), 0.0)
        o2      = h2 / 2
        err, acc, err_m, acc_m = _accuracy(pred, tasa)
        ef_far  = acc_m * 100
        coefs   = {"β": beta, "X": X, "pred": pred,
                   "err": err, "acc": acc, "err_m": err_m, "acc_m": acc_m}

    elif modelo == "Ley de Faraday":
        nf, nf_arr, tasa_teo, tasa_real, Vm = _nf_faraday(I, tasa, n_celdas, T, P)
        # tasa_teo y tasa_real ya están en mL/min → h2_arr y o2_arr en mL/min
        h2_arr = n_celdas * nf_arr * (I / (2 * F)) * Vm * 60 * 1000   # mL/min
        o2_arr = n_celdas * nf_arr * (I / (4 * F)) * Vm * 60 * 1000   # mL/min

        # CORRECCIÓN: asignar h2 y o2 para que consumo se calcule correctamente
        h2 = float(np.mean(h2_arr))
        o2 = float(np.mean(o2_arr))

        err, acc, err_m, acc_m = _accuracy(h2_arr, tasa_real)
        ef_far = acc_m * 100          # eficiencia del modelo (accuracy)

        # CORRECCIÓN: ef_far para eficiencia global usa la faradaica (nf), no el accuracy
        ef_far_global = float(np.mean(nf_arr)) * 100

        coefs = {
            "nf":          nf,
            "nf_arr":      nf_arr,
            "Vm":          Vm,
            "tasa_teo":    tasa_teo,    # mL/min
            "tasa_real":   tasa_real,   # mL/min
            "h2_arr":      h2_arr,      # mL/min
            "o2_arr":      o2_arr,      # mL/min
            "ef_nf_pct":   nf_arr * 100,
            "ef_modelo":   ef_far,
            "ef_far_global": ef_far_global,
        }
        # Usar la eficiencia faradaica (no la del modelo) en el cálculo global
        ef_far = ef_far_global

    else:
        h2 = o2 = ef_far = 0.0

    # Asegurar que h2, o2 y ef_far sean siempre escalares Python finitos
    h2     = float(h2)     if (h2 is not None and np.isfinite(h2))     else 0.0
    o2     = float(o2)     if (o2 is not None and np.isfinite(o2))     else 0.0
    ef_far = float(ef_far) if (ef_far is not None and np.isfinite(ef_far)) else 0.0

    # consumo en kWh/L_H2: potencia[W]/1000→kW, h2[mL/min]·60/1000→L/h
    # Umbral mínimo de 1e-10 para evitar ZeroDivisionError por underflow numérico
    consumo    = (potencia / 1000) / (h2 * 60 / 1000) if h2 > 1e-10 else None
    eficiencia = (ef_far * ef_volt) / 100

    return h2, o2, potencia, consumo, eficiencia, ef_far, ef_volt, coefs


def h2_por_punto(modelo, I_val, V_val, coefs, n_celdas, T, P):
    F = 96485
    if modelo == "Regresión lineal":
        return max((coefs["β₀"] + coefs["β₁"] * I_val), 0.0)
    elif modelo == "Modelo de superficie":
        x = np.array([1, V_val, I_val, V_val * I_val, V_val*2, I_val*2])
        return max(float(x @ coefs["β"]), 0.0)
    elif modelo == "Ley de Faraday":
        nf = coefs.get("nf", 0.0)
        Vm = coefs.get("Vm", 0.0)
        # Vm en L/mol → resultado en mL/min
        return n_celdas * nf * (I_val / (2 * F)) * Vm * 60 * 1000
    return 0.0


def modelo_pem_base(I, E_rev, A, I_0, R_ohm):
    return E_rev + A * np.log(I / I_0) + I * R_ohm


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS: ecuaciones en LaTeX
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(val):
    return f"{val:.2f}"


def _fmt_sci(val):
    base, exp_str = f"{val:.2e}".split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp_str)}}}"


def latex_regresion_lineal(coefs):
    b0 = coefs["β₀"]
    b1 = coefs["β₁"]
    signo = "+" if b1 >= 0 else "-"
    return (
        r"\text{Producción}_{H_2} \, \text{(mL/min)} = " +
        f"{_fmt(b0)} {signo} {_fmt(abs(b1))} \\cdot I"
    )


def latex_superficie(coefs):
    beta = coefs["β"]
    b0, b1, b2, b3, b4, b5 = beta

    def s(v, var):
        sign = "+" if v >= 0 else "-"
        return f"{sign} {_fmt(abs(v))} \\cdot {var}"

    return (
        r"\text{Producción}_{H_2} \, \text{(mL/min)} = " +
        f"{_fmt(b0)} {s(b1,'V')} {s(b2,'I')} {s(b3,'V \\cdot I')} {s(b4,'V^2')} {s(b5,'I^2')}"
    )


def latex_faraday(coefs, n_celdas, T, P):
    F = 96485
    nf = coefs["nf"]
    Vm = coefs["Vm"]
    return (
        r"\text{Producción}_{H_2} \, \text{(mL/min)} = " +
        f"{n_celdas} \\cdot {_fmt(nf)} \\cdot \\frac{{I}}{{2 \\cdot {F}}} \\cdot {_fmt(Vm)} \\cdot 60 \\cdot 1000"
    )


def latex_faraday_simplificada(coefs, n_celdas):
    F = 96485
    nf = coefs["nf"]
    Vm = coefs["Vm"]
    # Vm en L/mol → ×1000 para mL/min
    K = n_celdas * nf * Vm * 60 * 1000 / (2 * F)
    return (
        r"\text{Producción}_{H_2} \, \text{(mL/min)} = " +
        f"{_fmt(K)} \\cdot I"
    )


def latex_pem(popt):
    E_rev, A, I_0, R_ohm = popt
    return (
        r"V = " +
        f"{_fmt(E_rev)} + {_fmt(A)} \\cdot \\ln\\!\\left(\\frac{{I}}{{{_fmt_sci(I_0)}}}\\right) + {_fmt(R_ohm)} \\cdot I"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: tarjeta de eficiencia del modelo
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_eficiencia_modelo(modelo_actual, coefs, ef_far):
    st.markdown('<div class="seccion-titulo">Coeficientes del modelo de producción</div>', unsafe_allow_html=True)

    if modelo_actual == "Regresión lineal":
        b0, b1 = coefs["β₀"], coefs["β₁"]
        c1, c2 = st.columns(2)
        c1.metric("β₀ (intercepto)", f"{_fmt(b0)}")
        c2.metric("β₁ (pendiente)", f"{_fmt(b1)}")

    elif modelo_actual == "Modelo de superficie":
        beta = coefs["β"]
        labels = ["β₀", "β₁ (V)", "β₂ (I)", "β₃ (V·I)", "β₄ (V²)", "β₅ (I²)"]
        cols = st.columns(3)
        for idx, (lbl, val) in enumerate(zip(labels, beta)):
            cols[idx % 3].metric(lbl, f"{_fmt(val)}")

    elif modelo_actual == "Ley de Faraday":
        nf = coefs["nf"]
        Vm = coefs["Vm"]
        st.markdown("*Parámetro ajustado a los datos:*")
        c1, c2 = st.columns(2)
        c1.metric("Eficiencia faradaica η_F", f"{float(np.mean(coefs['ef_nf_pct'])):.2f} %")
        c2.metric("Volumen molar del gas, Vm (L/mol)", f"{_fmt(Vm)}")
        st.markdown("*Constantes físicas fijas:*")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Constante de Faraday, F (C/mol)", "96,485")
        k2.metric("Constante de gas ideal, R (L·atm/mol·K)", "0.08314")
        k3.metric("Electrones por mol H₂ (n)", "2")
        k4.metric("Electrones por mol O₂ (n)", "4")

    st.markdown('<div class="seccion-titulo">Eficiencia del modelo de producción</div>', unsafe_allow_html=True)

    # Para Faraday, mostrar tanto la eficiencia faradaica como la del modelo
    if modelo_actual == "Ley de Faraday":
        ef_modelo = coefs["ef_modelo"]
        ef_far_global = coefs["ef_far_global"]
        col_ef1, col_ef2, col_ef3 = st.columns(3)
        col_ef1.metric("Eficiencia faradaica (%)", f"{_fmt(ef_far_global)} %")
        col_ef2.metric("Eficiencia del modelo (%)", f"{_fmt(ef_modelo)} %")
        ef_mostrar = ef_modelo
    else:
        col_ef1, col_ef2 = st.columns([1, 2])
        col_ef1.metric("Eficiencia del modelo (%)", f"{_fmt(ef_far)} %")
        ef_mostrar = ef_far
        col_ef2 = col_ef2  # alias para el mensaje

    if ef_mostrar >= 85:
        clase = "ef-alta"
        icono = "✅"
        titulo = "Eficiencia óptima"
        mensaje = (
            f"La eficiencia del modelo es <strong>{_fmt(ef_mostrar)} %</strong>, lo que indica un excelente "
            "acuerdo entre las predicciones y los datos experimentales. "
            "El electrolizador opera en condiciones cercanas al ideal teórico."
        )
    elif ef_mostrar >= 60:
        clase = "ef-media"
        icono = "⚠️"
        titulo = "Eficiencia moderada"
        mensaje = (
            f"La eficiencia del modelo es <strong>{_fmt(ef_mostrar)} %</strong>. Existe un ajuste aceptable, "
            "pero hay margen de mejora. Revisa la distribución de los datos experimentales "
            "y considera ampliar el rango de corrientes medidas."
        )
    else:
        clase = "ef-baja"
        icono = "❌"
        titulo = "Eficiencia baja"
        mensaje = (
            f"La eficiencia del modelo es <strong>{_fmt(ef_mostrar)} %</strong>, lo que sugiere que el modelo "
            "no captura bien el comportamiento del electrolizador. "
            "Verifica la calidad de los datos, descarta valores atípicos o prueba un modelo diferente."
        )

    if modelo_actual == "Ley de Faraday":
        col_ef3.markdown(
            f'<div class="ef-card {clase}">'
            f'<strong>{icono} {titulo}:</strong> {mensaje}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        col_ef2.markdown(
            f'<div class="ef-card {clase}">'
            f'<strong>{icono} {titulo}:</strong> {mensaje}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
try:
    # CORRECCIÓN: use_column_width deprecado → use_container_width
    st.sidebar.image("UCO.png", use_container_width=True)
except Exception:
    pass

st.sidebar.title("Estado de la Simulación")
pasos_nombres = ["Ingreso de datos", "Optimización de cinéticas", "Simulación de H₂", "Resumen"]

for i, nombre in enumerate(pasos_nombres):
    n = i + 1
    if n < st.session_state.paso_actual:
        st.sidebar.markdown(f"✅ *Paso {n}:* {nombre}")
    elif n == st.session_state.paso_actual:
        st.sidebar.markdown(
            f"<span style='color:#008b50;font-weight:bold;'>▶️ Paso {n}: {nombre}</span>",
            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            f"<span style='color:#9ca3af;'>🔒 Paso {n}: {nombre}</span>",
            unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.progress(st.session_state.paso_actual / len(pasos_nombres))

st.title("Gemelo digital electrolizador PEM")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1 – Ingreso de datos
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.paso_actual == 1:
    st.header("Paso 1: Ingreso de datos")
    st.write("Ingresa los parámetros de la celda, las condiciones de experimentación y los datos experimentales.")

    col1, col2, col3 = st.columns(3)
    celdas      = col1.number_input("Número de celdas", min_value=1, value=st.session_state.celdas, format="%d", step=1)
    temperatura = col2.number_input("Temperatura (K)", min_value=0.0, value=st.session_state.temperatura, max_value=313.0, format="%.2f", step=1.0)
    presion     = col3.number_input("Presión atmosférica (atm)", min_value=-0.4, value=st.session_state.presion, max_value=40.0, format="%.2f", step=1.0)

    st.info("⚠️ Se deben proporcionar mínimo 15 puntos medidos, bien distribuidos a lo largo del rango operativo.")

    st.session_state.data.reset_index(drop=True, inplace=True)
    data = st.data_editor(
        st.session_state.data,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "voltaje":   st.column_config.NumberColumn("Voltaje (V)", format="%.2f"),
            "corriente": st.column_config.NumberColumn("Corriente (A)", format="%.2f"),
            "tasa":      st.column_config.NumberColumn("Tasa de producción H₂ (mL/min)", format="%.2f"),
        },
    )


    def ingreso_datos():
        if len(data.dropna()) < 15:
            st.warning("Ingresa al menos 15 mediciones válidas para continuar.")
        else:
            st.session_state.celdas = celdas
            st.session_state.temperatura = temperatura
            st.session_state.presion = presion
            st.session_state.data = data.copy()
            st.session_state.paso_actual += 1

    col1, col2, col3 = st.columns(3)
    col3.button("Guardar y Continuar ▶️", on_click=ingreso_datos, type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 – Optimización de cinéticas
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.paso_actual == 2:
    st.header("Paso 2: Optimización de cinéticas")
    st.write("Ajuste de parámetros cinéticos mediante el modelo de Butler-Volmer / Tafel con resistencia óhmica.")

    data_limpia = st.session_state.data.dropna()
    data_limpia = data_limpia[(data_limpia["corriente"] > 0) & (data_limpia["voltaje"] > 0)]
    I_dat = data_limpia["corriente"].values
    V_dat = data_limpia["voltaje"].values

    li = [1.1, 0.001, 1e-10, 0.0001]
    ls = [3.0, 2.0,   1.0,   10.0]
    seed_E = np.clip(np.min(V_dat) * 0.9, li[0] + 1e-5, ls[0] - 1e-5)
    dV = V_dat[-1] - V_dat[-3] if len(V_dat) >= 3 else 0.1
    dI = I_dat[-1] - I_dat[-3] if len(I_dat) >= 3 else 1.0
    seed_R = np.clip(abs(dV / dI) if dI != 0 else 0.1, li[3] + 1e-5, ls[3] - 1e-5)
    seeds  = [seed_E, 0.1, 0.01, seed_R]

    try:
        popt, _ = curve_fit(modelo_pem_base, I_dat, V_dat, p0=seeds, bounds=(li, ls),
                            maxfev=10000, method='trf', x_scale='jac')
        E_rev_opt, A_opt, I_0_opt, R_ohm_opt = popt
        st.session_state.popt = popt

        res = V_dat - modelo_pem_base(I_dat, *popt)
        r2  = 1 - np.sum(res*2) / np.sum((V_dat - V_dat.mean())*2)

        st.success("✅ El algoritmo convergió correctamente.")

        st.subheader("Parámetros ajustados")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Voltaje Reversible (V)", f"{_fmt(E_rev_opt)}")
        c2.metric("Pendiente Tafel, A",     f"{_fmt(A_opt)}")
        c3.metric("I. de Intercambio (A)",  f"{I_0_opt:.2e}")
        c4.metric("R. Óhmica (Ω)",          f"{_fmt(R_ohm_opt)}")

        st.markdown(
            f'<div class="r2-verde">✅ <strong>Coeficiente de determinación (R²):</strong> {_fmt(r2)}</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Ecuación del modelo")
        st.latex(r"\LARGE " + latex_pem(popt))

        st.divider()
        st.subheader("Curva de polarización")
        I_lin = np.linspace(I_dat.min() * 0.95, I_dat.max() * 1.05, 200)
        fig = go.Figure([
            go.Scatter(x=I_dat, y=V_dat, mode='markers', name='Datos experimentales',
                       marker=dict(color='royalblue', size=8)),
            go.Scatter(x=I_lin, y=modelo_pem_base(I_lin, *popt), mode='lines', name='Modelo ajustado',
                       line=dict(color='crimson', width=2)),
        ])
        fig.update_layout(xaxis_title="Corriente (A)", yaxis_title="Voltaje (V)",
                          margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        # CORRECCIÓN: use_container_width en lugar de width='stretch' (deprecado)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("❌ El algoritmo no logró converger.")
        st.warning(f"Detalle: {e}")
        st.info("Verifica que los datos tengan sentido físico y estén bien distribuidos en el rango operativo.")

    c1, c2, c3 = st.columns(3)
    c1.button("◀️ Atrás",     on_click=paso_anterior)
    c2.button("🔄 Reiniciar", on_click=reiniciar)
    c3.button("Continuar ▶️", on_click=siguiente_paso, type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3 – Simulación
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.paso_actual == 3:
    st.header("Paso 3: Simulación de generación de H₂ y O₂")
    st.write("Selecciona el modelo de producción. Los resultados se calculan a partir de los datos experimentales del Paso 1.")

    opciones_modelo = ["-- Seleccionar --", "Regresión lineal", "Modelo de superficie", "Ley de Faraday"]
    idx_previo = (
        opciones_modelo.index(st.session_state.modelo_seleccionado)
        if st.session_state.modelo_seleccionado in opciones_modelo
        else 0
    )

    modelo = st.selectbox(
        "Modelo de producción de H₂:",
        opciones_modelo,
        index=idx_previo,
    )

    if modelo != "-- Seleccionar --":
        st.session_state.modelo_seleccionado = modelo
    else:
        st.session_state.modelo_seleccionado = None

    if not st.session_state.modelo_seleccionado:
        st.info("👆 Selecciona un modelo para ver los resultados.")

    else:
        modelo_actual = st.session_state.modelo_seleccionado
        data_limpia   = st.session_state.data.dropna()
        data_limpia   = data_limpia[(data_limpia["corriente"] >= 0) & (data_limpia["voltaje"] >= 0)]

        h2, o2, potencia, consumo, eficiencia, ef_far, ef_volt, coefs = calcular_produccion(
            modelo_actual, data_limpia,
            st.session_state.celdas, st.session_state.temperatura, st.session_state.presion,
        )

        st.session_state.update({
            "h2": h2, "o2": o2, "potencia": potencia, "consumo": consumo,
            "eficiencia": eficiencia, "ef_faradaica": ef_far,
            "ef_voltaica": ef_volt, "coeficientes": coefs,
        })

        I_arr    = data_limpia["corriente"].values.astype(float)
        V_arr    = data_limpia["voltaje"].values.astype(float)
        tasa_arr = data_limpia["tasa"].values.astype(float)

        st.divider()

        mostrar_eficiencia_modelo(modelo_actual, coefs, ef_far)

        st.markdown('<div class="seccion-titulo">Ecuación del modelo de producción</div>', unsafe_allow_html=True)

        if modelo_actual == "Regresión lineal":
            st.latex(r"\Large " + latex_regresion_lineal(coefs))
        elif modelo_actual == "Modelo de superficie":
            st.latex(r"\normalsize " + latex_superficie(coefs))
        elif modelo_actual == "Ley de Faraday":
            st.latex(r"\Large " + latex_faraday(coefs, st.session_state.celdas, st.session_state.temperatura, st.session_state.presion))

        st.divider()

        st.markdown('<div class="seccion-titulo">Visualización del modelo de producción</div>', unsafe_allow_html=True)

        idx_s  = np.argsort(I_arr)
        I_s    = I_arr[idx_s]
        V_s    = V_arr[idx_s]
        tasa_s = tasa_arr[idx_s]

        # ── SUPERFICIE: gráfica 3D ─────────────────────────────────────────────
        if modelo_actual == "Modelo de superficie":
            beta = coefs["β"]
            I_grid = np.linspace(I_arr.min(), I_arr.max(), 40)
            V_grid = np.linspace(V_arr.min(), V_arr.max(), 40)
            II, VV = np.meshgrid(I_grid, V_grid)
            ZZ = (beta[0] + beta[1]*VV + beta[2]*II + beta[3]VV*II + beta[4]*VV2 + beta[5]*II*2)

            fig3d = go.Figure()
            fig3d.add_trace(go.Surface(
                x=II, y=VV, z=ZZ,
                colorscale='Greens', opacity=0.85,
                name='Superficie ajustada', showscale=True,
                colorbar=dict(title="mL/min"),
            ))
            fig3d.add_trace(go.Scatter3d(
                x=I_arr, y=V_arr, z=tasa_arr,
                mode='markers',
                marker=dict(size=5, color='darkorange', symbol='circle'),
                name='Datos experimentales',
            ))
            fig3d.update_layout(
                scene=dict(
                    xaxis_title="Corriente (A)",
                    yaxis_title="Voltaje (V)",
                    zaxis_title="Producción H₂ (mL/min)",
                ),
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            # CORRECCIÓN: use_container_width en lugar de width='stretch'
            st.plotly_chart(fig3d, use_container_width=True)

        # ── REGRESIÓN LINEAL: ajuste + residuos ───────────────────────────────
        elif modelo_actual == "Regresión lineal":
            b0, b1 = coefs["β₀"], coefs["β₁"]
            I_lin    = np.linspace(I_arr.min() * 0.95, I_arr.max() * 1.05, 200)
            pred_lin = b0 + b1 * I_lin
            pred_pts = b0 + b1 * I_s
            residuos = tasa_s - pred_pts
            std_res  = np.std(residuos)

            fig_rl = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Regresión lineal con banda ±1σ", "Residuos del modelo")
            )
            fig_rl.add_trace(go.Scatter(
                x=np.concatenate([I_lin, I_lin[::-1]]),
                y=np.concatenate([pred_lin + std_res, (pred_lin - std_res)[::-1]]),
                fill='toself', fillcolor='rgba(0,139,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Banda ±1σ', showlegend=True,
            ), row=1, col=1)
            fig_rl.add_trace(go.Scatter(
                x=I_lin, y=pred_lin, mode='lines',
                name='Modelo lineal', line=dict(color='#008b50', width=2.5),
            ), row=1, col=1)
            fig_rl.add_trace(go.Scatter(
                x=I_s, y=tasa_s, mode='markers',
                name='Datos medidos', marker=dict(color='darkorange', size=8, symbol='square'),
            ), row=1, col=1)
            fig_rl.add_trace(go.Bar(
                x=I_s, y=residuos,
                name='Residuo',
                marker_color=['#008b50' if r >= 0 else '#e74c3c' for r in residuos],
                showlegend=True,
            ), row=1, col=2)
            fig_rl.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
            fig_rl.update_xaxes(title_text="Corriente (A)", row=1, col=1)
            fig_rl.update_xaxes(title_text="Corriente (A)", row=1, col=2)
            fig_rl.update_yaxes(title_text="Producción H₂ (mL/min)", row=1, col=1)
            fig_rl.update_yaxes(title_text="Residuo (mL/min)", row=1, col=2)
            fig_rl.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            # CORRECCIÓN: use_container_width en lugar de width='stretch'
            st.plotly_chart(fig_rl, use_container_width=True)

        # ── LEY DE FARADAY: múltiples gráficas con selector ──────────────────
        elif modelo_actual == "Ley de Faraday":
            F_const    = 96485
            nf_mean    = coefs["nf"]
            nf_arr_exp = coefs["nf_arr"]
            Vm         = coefs["Vm"]
            n_cel      = st.session_state.celdas
            tasa_teo   = coefs["tasa_teo"]    # mL/min
            tasa_real  = coefs["tasa_real"]   # mL/min
            h2_arr     = coefs["h2_arr"]      # mL/min
            o2_arr     = coefs["o2_arr"]      # mL/min

            tipo_graf = st.radio(
                "Selecciona la visualización:",
                [
                    "Producción H₂ vs Corriente",
                    "Tasa teórica vs Tasa experimental",
                    "Producción H₂ y O₂ vs Corriente",
                ],
                horizontal=True,
            )

            # 1. Producción H₂ vs Corriente + banda de ±5 %
            if tipo_graf == "Producción H₂ vs Corriente":
                I_lin  = np.linspace(I_s.min() * 0.9, I_s.max() * 1.1, 200)
                # CORRECCIÓN: Vm en L/mol → ×1000 para mL/min
                h2_lin = n_cel * nf_mean * (I_lin / (2 * F_const)) * Vm * 60 * 1000
                delta  = 0.05 * h2_lin

                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(
                    x=np.concatenate([I_lin, I_lin[::-1]]),
                    y=np.concatenate([h2_lin + delta, (h2_lin - delta)[::-1]]),
                    fill='toself', fillcolor='rgba(0,100,200,0.12)',
                    line=dict(color='rgba(0,0,0,0)'), name='Banda ±5 %',
                ))
                fig_f1.add_trace(go.Scatter(
                    x=I_lin, y=h2_lin, mode='lines',
                    name='Modelo Faraday', line=dict(color='royalblue', width=2.5),
                ))
                fig_f1.add_trace(go.Scatter(
                    x=I_s, y=tasa_real[idx_s], mode='markers',
                    name='Datos experimentales',
                    marker=dict(color='darkorange', size=8, symbol='circle'),
                ))
                fig_f1.update_layout(
                    xaxis_title="Corriente (A)",
                    yaxis_title="Producción H₂ (mL/min)",
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                # CORRECCIÓN: use_container_width en lugar de width='stretch'
                st.plotly_chart(fig_f1, use_container_width=True)

            # 2. Tasa teórica vs Tasa experimental (paridad)
            elif tipo_graf == "Tasa teórica vs Tasa experimental":
                # CORRECCIÓN: tasa_teo y tasa_real ya están en mL/min → sin ×1000
                teo_ml  = tasa_teo
                real_ml = tasa_real
                lim_min = min(teo_ml.min(), real_ml.min()) * 0.95
                lim_max = max(teo_ml.max(), real_ml.max()) * 1.05

                fig_f2 = go.Figure()
                fig_f2.add_trace(go.Scatter(
                    x=[lim_min, lim_max], y=[lim_min, lim_max],
                    mode='lines', name='Paridad (y = x)',
                    line=dict(color='gray', dash='dash'),
                ))
                fig_f2.add_trace(go.Scatter(
                    x=teo_ml, y=real_ml, mode='markers',
                    name='Puntos experimentales',
                    marker=dict(color='royalblue', size=8),
                ))
                fig_f2.update_layout(
                    xaxis_title="Tasa teórica (mL/min)",
                    yaxis_title="Tasa real (mL/min)",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                # CORRECCIÓN: use_container_width en lugar de width='stretch'
                st.plotly_chart(fig_f2, use_container_width=True)

            # 3. H₂ y O₂ vs Corriente (doble eje)
            elif tipo_graf == "Producción H₂ y O₂ vs Corriente":
                I_lin  = np.linspace(I_s.min() * 0.9, I_s.max() * 1.1, 200)
                # CORRECCIÓN: Vm en L/mol → ×1000 para mL/min
                h2_lin = n_cel * nf_mean * (I_lin / (2 * F_const)) * Vm * 60 * 1000
                o2_lin = n_cel * nf_mean * (I_lin / (4 * F_const)) * Vm * 60 * 1000

                fig_f4 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_f4.add_trace(go.Scatter(
                    x=I_lin, y=h2_lin, mode='lines',
                    name='H₂ — modelo', line=dict(color='royalblue', width=2.5),
                ), secondary_y=False)
                fig_f4.add_trace(go.Scatter(
                    x=I_s, y=tasa_real[idx_s], mode='markers',
                    name='H₂ — experimental',
                    marker=dict(color='royalblue', size=8, symbol='circle-open'),
                ), secondary_y=False)
                fig_f4.add_trace(go.Scatter(
                    x=I_lin, y=o2_lin, mode='lines',
                    name='O₂ — modelo', line=dict(color='#e74c3c', width=2.5, dash='dot'),
                ), secondary_y=True)
                fig_f4.add_trace(go.Scatter(
                    x=I_s, y=o2_arr[idx_s], mode='markers',
                    name='O₂ — experimental',
                    marker=dict(color='#e74c3c', size=8, symbol='circle-open'),
                ), secondary_y=True)
                fig_f4.update_xaxes(title_text="Corriente (A)")
                fig_f4.update_yaxes(title_text="Producción H₂ (mL/min)", secondary_y=False)
                fig_f4.update_yaxes(title_text="Producción O₂ (mL/min)", secondary_y=True)
                fig_f4.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                # CORRECCIÓN: use_container_width en lugar de width='stretch'
                st.plotly_chart(fig_f4, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.button("◀️ Atrás",     on_click=paso_anterior)
    c2.button("🔄 Reiniciar", on_click=reiniciar)
    c3.button("Continuar ▶️", on_click=siguiente_paso, type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4 – Resumen final
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.paso_actual == 4:
    st.header("Paso 4: Resumen Final")
    st.success("✅ Simulación completada con éxito.")

    popt          = st.session_state.get("popt", None)
    coefs         = st.session_state.get("coeficientes", {})
    data_limpia   = st.session_state.data.dropna()
    data_limpia   = data_limpia[(data_limpia["corriente"] > 0) & (data_limpia["voltaje"] > 0)]
    I_exp         = data_limpia["corriente"].values.astype(float)
    V_exp         = data_limpia["voltaje"].values.astype(float)
    modelo_actual = st.session_state.get("modelo_seleccionado", "No seleccionado")
    h2            = st.session_state.get("h2",           0.0)
    o2            = st.session_state.get("o2",           0.0)
    potencia      = st.session_state.get("potencia",     0.0)
    consumo       = st.session_state.get("consumo",      None)
    ef_far        = st.session_state.get("ef_faradaica", 0.0)
    ef_volt       = st.session_state.get("ef_voltaica",  0.0)

    st.markdown('<div class="seccion-titulo">Ecuación obtenida en el Paso 2 — Modelo cinético (Butler-Volmer / Tafel)</div>', unsafe_allow_html=True)

    if popt is not None:
        st.latex(r"\Large " + latex_pem(popt))
    else:
        st.info("No se obtuvieron parámetros cinéticos en el Paso 2.")

    st.markdown('<div class="seccion-titulo">Ecuación obtenida en el Paso 3 — Modelo de producción de H₂</div>', unsafe_allow_html=True)
    st.write(f"*Modelo seleccionado:* {modelo_actual}")

    if modelo_actual == "Regresión lineal" and coefs:
        st.latex(r"\Large " + latex_regresion_lineal(coefs))
    elif modelo_actual == "Modelo de superficie" and coefs:
        st.latex(r"\Large " + latex_superficie(coefs))
    elif modelo_actual == "Ley de Faraday" and coefs:
        # CORRECCIÓN: latex_faraday_simplificada ya incluye ×1000 en la constante K
        st.latex(r"\Large " + latex_faraday_simplificada(coefs, st.session_state.celdas))

    st.divider()

    st.markdown('<div class="seccion-titulo">Calculadora de punto de operación</div>', unsafe_allow_html=True)
    st.write(
        "Ingresa un valor de *corriente* o de *voltaje* para predecir el otro y "
        "estimar la producción de H₂ con el modelo seleccionado."
    )

    modo_calc = st.radio(
        "¿Qué variable quieres ingresar?",
        ["Corriente (A) → obtener Voltaje y H₂",
         "Voltaje (V) → obtener Corriente y H₂"],
        horizontal=True,
    )

    h2_calc = 0.0
    col_inp, col_out = st.columns(2)

    if modo_calc.startswith("Corriente"):
        I_usuario = col_inp.number_input(
            "Corriente (A)", min_value=0.001,
            value=float(np.mean(I_exp)), format="%.4f",
        )

        if popt is not None:
            V_calc = modelo_pem_base(I_usuario, *popt)
            col_out.metric("Voltaje optimizado (V) — modelo Paso 2", f"{_fmt(V_calc)}")
        else:
            V_calc = float(np.mean(V_exp))
            col_out.info("Sin parámetros cinéticos; se usa voltaje promedio experimental.")

        if coefs:
            h2_calc = h2_por_punto(
                modelo_actual, I_usuario, V_calc, coefs,
                st.session_state.celdas, st.session_state.temperatura, st.session_state.presion,
            )

        st.metric("Producción de H₂ estimada (mL/min)", f"{h2_calc:.2f}")

    else:  # Voltaje → Corriente
        V_usuario = col_inp.number_input(
            "Voltaje (V)", min_value=0.001,
            value=float(np.mean(V_exp)), format="%.4f",
        )

        if popt is not None:
            def ecuacion_a_resolver(I_var):
                if I_var <= 0:
                    return 1e6
                return modelo_pem_base(I_var, *popt) - V_usuario

            try:
                I_seed = float(np.mean(I_exp))
                I_calc_arr = fsolve(ecuacion_a_resolver, I_seed, full_output=True)
                I_calc = float(I_calc_arr[0][0])
                convergio = abs(ecuacion_a_resolver(I_calc)) < 1e-4

                if convergio and I_calc > 0:
                    col_out.metric("Corriente estimada (A) — modelo Paso 2", f"{_fmt(I_calc)}")
                else:
                    st.warning("No se pudo despejar la corriente para ese voltaje. Verifica que esté en rango.")
                    I_calc = float(np.mean(I_exp))
            except Exception:
                st.warning("Error al despejar la corriente. Se usa valor medio experimental.")
                I_calc = float(np.mean(I_exp))
        else:
            I_calc = float(np.mean(I_exp))
            col_out.info("Sin parámetros cinéticos; se usa corriente promedio experimental.")

        if coefs:
            h2_calc = h2_por_punto(
                modelo_actual, I_calc, V_usuario, coefs,
                st.session_state.celdas, st.session_state.temperatura, st.session_state.presion,
            )

        st.metric("Producción de H₂ estimada (mL/min)", f"{h2_calc:.2f}")

    st.divider()
    c1, c2 = st.columns(2)
    c1.button("◀️ Atrás",             on_click=paso_anterior)
    c2.button("🔄 Nueva Simulación", on_click=reiniciar, type="primary")