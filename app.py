import warnings
warnings.filterwarnings('ignore')
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline, _logistic_map

# ═══════════════════════════════════════════
# CONFIGURACIÓN Y ESTILO
# ═══════════════════════════════════════════
P = {'bg': '#0d1117', 'surface': '#161b22', 'accent': '#58a6ff', 'purple': '#bc8cff', 'text': '#e6edf3'}
plt.rcParams.update({
    'figure.facecolor': P['bg'], 
    'axes.facecolor': P['surface'], 
    'text.color': P['text'], 
    'axes.labelcolor': P['text'],
    'xtick.color': P['text'],
    'ytick.color': P['text']
})
AUTHOR = "Investigador: Emanuel Duarte · Pergamino, 2026"

# ═══════════════════════════════════════════
# LECTOR PHYSIONET (CON CORRECCIÓN DE ÍNDICE)
# ═══════════════════════════════════════════
def read_mitbih(record_id):
    """Lector robusto para evitar 'list index out of range'."""
    bases = [
        f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}',
        f'mitbih/{record_id}',
        str(record_id)
    ]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"Falta registro {record_id} (.hea). Verifica la carpeta 'mitbih'.")

    # 1. Leer Header
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    fs, n = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    # 2. Leer Datos (.dat)
    dat_file = path_base + '.dat'
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {record_id}.dat")

    with open(dat_file, 'rb') as f:
        raw = f.read()
    
    samples = []
    i = 0
    # Protección: Validamos que haya al menos 3 bytes para procesar
    while i + 2 < len(raw):
        try:
            b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
            # Lógica de bits original de Emanuel Duarte
            s1 = b0 | ((b1 & 0x0F) << 8)
            s1 = s1 - 4096 if s1 >= 2048 else s1
            
            s2 = b2 | ((b1 & 0xF0) << 4)
            s2 = s2 - 4096 if s2 >= 2048 else s2
            
            samples.extend([s1, s2])
            i += 3
        except (IndexError, ValueError):
            break
            
    if not samples:
        raise ValueError(f"Error crítico: No se pudieron extraer muestras de {record_id}.dat")

    # Ajustamos a la longitud declarada en el header
    mlii = np.array(samples[::2][:n])
    return (mlii - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.markdown(f"<h1>🌀 MODE · Attractor Explorer</h1>", unsafe_allow_html=True)
st.caption(AUTHOR)

with st.sidebar:
    st.header("📂 Entrada de Datos")
    fuente = st.radio("Seleccionar origen:", ["Señal Demo", "Cargar mi CSV", "PhysioNet (MIT-BIH)"])
    
    x_data, label, fs_actual = None, "", 1.0

    if fuente == "Señal Demo":
        tipo = st.selectbox("Modelo", ["Lorenz", "Seno + Ruido"])
        x_data = np.random.normal(0, 1, 1000) if tipo == "Seno + Ruido" else np.sin(np.linspace(0, 50, 1000))
        label = f"Demo {tipo}"
    
    elif fuente == "Cargar mi CSV":
        up = st.file_uploader("Subir CSV", type=['csv'])
        if up:
            x_data = pd.read_csv(up, header=None).iloc[:, 0].values
            label = up.name
            
    elif fuente == "PhysioNet (MIT-BIH)":
        rec = st.selectbox("Elegir Registro", ["100", "208", "214"])
        try:
            x_full, fs_actual = read_mitbih(rec)
            x_data = x_full[:1000] # Primeros 1000 puntos para la vista previa
            label = f"MIT-BIH {rec}"
            
            st.divider()
            st.subheader("🛠️ Diagnóstico Estructural")
            if st.button("Ejecutar Test (5 Ventanas)"):
                pipe_t = AttractorPipeline(m=3, max_tau=40, verbose=False)
                logs = []
                # El test de 5 ventanas sobre el registro cargado
                for j in range(5):
                    inicio = j * 500
                    ventana = x_full[inicio : inicio + 1000]
                    res_v = pipe_t.run(ventana)
                    logs.append({
                        "t(s)": f"{inicio/fs_actual:.1f}", 
                        "R³": f"{res_v['R3']['R3_score']:.3f}", 
                        "Coh": '✔' if res_v['R3']['coherent'] else '✘', 
                        "Régimen": res_v['R3']['regime']
                    })
                st.table(logs)
        except Exception as e:
            st.error(f"Error de lectura: {e}")

    st.divider()
    m_dim = st.slider("Dimensión m", 2, 5, 3)
    tau_m = st.slider("τ máximo", 10, 80, 40)
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)

# ═══════════════════════════════════════════
# ÁREA DE RESULTADOS
# ═══════════════════════════════════════════
if ejecutar and x_data is not None:
    with st.spinner("Procesando señal..."):
        pipe = AttractorPipeline(m=m_dim, max_tau=tau_m, verbose=False)
        res = pipe.run(x_data, label=label)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("τ AMI", res['tau'])
        c3.metric("Régimen", res['R3']['regime'])

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(x_data, color=P['accent'], lw=0.8)
        ax[0].set_title(f"Señal: {label}")
        ax[1].psd(x_data, Fs=fs_actual, color=P['purple'])
        ax[1].set_title("Espectro de Potencia")
        st.pyplot(fig)
        st.success(f"Análisis finalizado con éxito para {label}.")
elif not ejecutar:
    st.info("Configurá la entrada y presioná 'Analizar Pipeline'.")