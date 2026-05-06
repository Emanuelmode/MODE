import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR CALIBRADO (VERSIÓN EMANUEL)
# ═══════════════════════════════════════════
def read_mitbih_calibrated(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    if not path_base: return None, 360
    
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs = int(header[2])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    with open(path_base + '.dat', 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    n_groups = len(raw) // 3
    b = raw[:n_groups*3].reshape(-1, 3)
    # Reconstrucción de un solo canal (MLII) para máxima coherencia fractal
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    return (c1 - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ FINAL
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer v5", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · Calibración Exitosa")

with st.sidebar:
    st.header("Control")
    registro = st.selectbox("Registro:", ["100", "208", "214"])
    st.divider()
    m_dim = st.slider("Dimensión m", 2, 4, 3)
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)

data, fs = read_mitbih_calibrated(registro)

if data is not None:
    # Mostramos siempre la señal calibrada
    segmento = data[1000:3000]
    
    if ejecutar:
        pipe = AttractorPipeline(m=m_dim, max_tau=40)
        res = pipe.run(segmento)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
        c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")

    # Gráfico de monitoreo constante
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:1000], color='#58a6ff', lw=0.8)
    ax.set_title(f"Señal Calibrada - {registro}")
    ax.set_ylabel("mV")
    st.pyplot(fig)
