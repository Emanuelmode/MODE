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
plt.rcParams.update({'figure.facecolor': P['bg'], 'axes.facecolor': P['surface'], 'text.color': P['text'], 'axes.labelcolor': P['text']})
AUTHOR = "Investigador: Emanuel Duarte · 2026"

# ═══════════════════════════════════════════
# LECTOR UNIVERSAL PHYSIONET
# ═══════════════════════════════════════════
def read_mitbih(record_id):
    # Busca el archivo en cualquier subcarpeta común o en la raíz[cite: 2]
    bases = [
        f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}',
        f'mitbih/{record_id}',
        str(record_id)
    ]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"Falta registro {record_id}. Subí los archivos .hea y .dat a una carpeta 'mitbih'.")

    with open(path_base + '.hea') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    fs, n = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    with open(path_base + '.dat', 'rb') as f:
        raw = f.read()
    
    samples = []
    i = 0
    while i + 2 < len(raw):
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        s1 = b0 | ((b1 & 0x0F) << 8); s1 = s1 - 4096 if s1 >= 2048 else s1
        s2 = b2 | ((b1 & 0xF0) << 4); s2 = s2 - 4096 if s2 >= 2048 else s2
        samples.extend([s1, s2]); i += 3
    
    return (np.array(samples[::2][:n]) - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Attractor Explorer", layout="wide")
st.markdown(f"<h1>🌀 MODE · Attractor Explorer</h1>", unsafe_allow_html=True)
st.caption(AUTHOR)

with st.sidebar:
    st.header("📂 Entrada de Datos")
    fuente = st.radio("Seleccionar origen:", ["Señal Demo", "Cargar mi CSV", "PhysioNet (MIT-BIH)"])
    
    x_data, label, fs_actual = None, "", 1.0

    if fuente == "Señal Demo":
        tipo = st.selectbox("Modelo", ["Lorenz", "Ruido Blanco"])
        x_data = np.random.normal(0, 1, 1000) if tipo == "Ruido Blanco" else np.sin(np.linspace(0, 50, 1000))
        label = f"Demo {tipo}"
    
    elif fuente == "Cargar mi CSV":
        up = st.file_uploader("Subir CSV (1 columna)", type=['csv'])
        if up:
            x_data = pd.read_csv(up, header=None).iloc[:, 0].values
            label = up.name
            
    elif fuente == "PhysioNet (MIT-BIH)":
        rec = st.selectbox("Elegir Registro", ["100", "208", "214"])
        try:
            x_full, fs_actual = read_mitbih(rec)
            # Para el análisis principal usamos los primeros 2 segundos por defecto
            x_data = x_full[:int(fs_actual * 2.8)] 
            label = f"MIT-BIH {rec}"
            
            st.divider()
            st.subheader("🛠️ Diagnóstico Estructural")
            if st.button("Ejecutar Test (5 Ventanas)"):
                pipe_t = AttractorPipeline(m=3, max_tau=40, verbose=False)
                logs = []
                for i in range(5):
                    w = x_full[i*500 : i*500 + 1000]
                    r = pipe_t.run(w)
                    logs.append({"t(s)": f"{i*500/fs_actual:.1f}", "R³": f"{r['R3']['R3_score']:.3f}", 
                                 "Coh": '✔' if r['R3']['coherent'] else '✘', "Régimen": r['R3']['regime']})
                st.table(logs)
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    m_dim = st.slider("Dimensión m", 2, 5, 3)
    tau_m = st.slider("τ máximo", 10, 80, 40)
    ejecutar = st.button("▶ ANALIZAR CON PIPELINE", type="primary", use_container_width=True)

# ═══════════════════════════════════════════
# ÁREA DE RESULTADOS
# ═══════════════════════════════════════════
if ejecutar and x_data is not None:
    pipe = AttractorPipeline(m=m_dim, max_tau=tau_m, verbose=False)
    res = pipe.run(x_data, label=label)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    col2.metric("τ AMI", res['tau'])
    col3.metric("Régimen", res['R3']['regime'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x_data, color=P['accent'], lw=0.8); ax[0].set_title(f"Señal: {label}")
    ax[1].psd(x_data, Fs=fs_actual, color=P['purple']); ax[1].set_title("Densidad Espectral")
    st.pyplot(fig)
    
    st.success("Análisis estructural de Emanuel Duarte finalizado.")
elif not ejecutar:
    st.info("Seleccioná los datos a la izquierda y presioná el botón azul para iniciar el Pipeline.")