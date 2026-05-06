import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # Para suavizar la curva
from pipeline import AttractorPipeline

def read_mitbih_calibrated(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    if not path_base: return None, 360
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    with open(path_base + '.dat', 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    n_groups = len(raw) // 3
    b = raw[:n_groups*3].reshape(-1, 3)
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    return (c1 - baseline) / gain, 360

st.set_page_config(page_title="MODE v5.3 · Estructura Fina", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · Super-resolución de Atractor")

registro = st.sidebar.selectbox("Registro:", ["100", "208", "214"])
m_dim = st.sidebar.slider("Dimensión m", 2, 4, 3)
ejecutar = st.sidebar.button("▶ ANALIZAR PIPELINE", type="primary")

data, fs = read_mitbih_calibrated(registro)

if data is not None:
    # 1. Ventana de análisis
    seg_raw = data[1000:2500]
    
    # 2. SUAVIZADO POR INTERPOLACIÓN (Anti-Escalones)
    # Creamos 4 puntos nuevos por cada punto original para suavizar el atractor
    x_old = np.linspace(0, 1, len(seg_raw))
    x_new = np.linspace(0, 1, len(seg_raw) * 4)
    f_interp = interp1d(x_old, seg_raw, kind='cubic')
    seg_smooth = f_interp(x_new)
    
    # 3. Normalización Local final
    segmento = (seg_smooth - np.mean(seg_smooth)) / np.std(seg_smooth)
    
    if ejecutar:
        pipe = AttractorPipeline(m=m_dim, max_tau=40)
        res = pipe.run(segmento)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
        c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:400], color='#00d4ff', lw=1) # Azul eléctrico para alta resolución
    ax.set_title("Señal con Super-Resolución (Suavizado Cúbico)")
    st.pyplot(fig)
