import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR DE SEÑAL OPTIMIZADO
# ═══════════════════════════════════════════
def read_mitbih_clean(record_id):
    """Lee PhysioNet asegurando un solo canal para evitar ruido de fase."""
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No se encontró el header del registro {record_id}")

    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs = int(header[2])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    dat_path = path_base + '.dat'
    with open(dat_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    n_groups = len(raw) // 3
    b = raw[:n_groups*3].reshape(-1, 3)
    
    # Extraemos canal único (MLII) para limpiar el atractor
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    
    # Normalización: (Muestras - Baseline) / Gain
    return (c1 - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ DE INVESTIGACIÓN
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE Explorer v3", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption(f"Investigador Responsable: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("📂 Fuente de Datos")
    fuente = st.radio("Seleccionar:", ["PhysioNet (MIT-BIH)", "Archivo Local (CSV)"])
    
    data_signal, fs_actual = None, 360.0

    if fuente == "PhysioNet (MIT-BIH)":
        reg = st.selectbox("Registro:", ["100", "208", "214"])
        try:
            data_signal, fs_actual = read_mitbih_clean(reg)
            st.success(f"Cargado: {reg} ({len(data_signal)} pts)")
        except Exception as e:
            st.error(f"Error: {e}")
            
    else:
        # Carga desde gemini csv.csv si existe en el repo
        if os.path.exists('gemini csv.csv'):
            df = pd.read_csv('gemini csv.csv')
            data_signal = df.iloc[:, 0].values
            st.success("Cargado: gemini csv.csv")
        else:
            st.warning("No se encontró 'gemini csv.csv' en el repositorio.")

    st.divider()
    # Ajuste de sensibilidad para la Verdad Estructural
    m_dim = st.slider("Dimensión de Inmersión (m)", 2, 4, 3)
    analizar = st.button("▶ EJECUTAR PIPELINE", type="primary", use_container_width=True)

# ═══════════════════════════════════════════
# PROCESAMIENTO Y VISUALIZACIÓN
# ═══════════════════════════════════════════
if analizar and data_signal is not None:
    # Seleccionamos una ventana significativa (1500 muestras)
    # Si la señal es corta, tomamos lo que haya
    idx_limit = min(2500, len(data_signal))
    segmento = data_signal[1000:idx_limit] if len(data_signal) > 1000 else data_signal
    
    # El corazón del pipeline MODE
    pipe = AttractorPipeline(m=m_dim, max_tau=40)
    res = pipe.run(segmento)
    
    # Métricas de Verdad Estructural
    c1, c2, c3 = st.columns(3)
    c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    c2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
    c3.metric("Coherencia", "SÍ (Validado)" if res['R3']['coherent'] else "NO (Ruido)")

    # Gráfico de la señal para validación visual
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:1000], color='#58a6ff', lw=0.8)
    ax.set_title(f"Señal Analizada - Dinámica Temporal")
    ax.set_ylabel("Amplitud (mV)")
    st.pyplot(fig)
    
    # Debug de muestras para Emanuel
    with st.expander("Ver muestras procesadas (Debug)"):
        st.write(segmento[:15])
else:
    st.info("Configurá la fuente y presioná 'Ejecutar' para ver la dinámica del atractor.")
