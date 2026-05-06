import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR PHYSIONET (CANAL ÚNICO)
# ═══════════════════════════════════════════
def read_mitbih_final(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"Falta {record_id}.hea")

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
    
    # Extraemos solo el Canal 1 (MLII) para evitar el "zig-zag"
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    
    return (c1 - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · 2026")

with st.sidebar:
    registro = st.selectbox("Registro:", ["100", "208", "214"])
    analizar = st.button("▶ EJECUTAR PIPELINE", type="primary")

try:
    data, fs = read_mitbih_final(registro)
    # Tomamos un segmento limpio para el análisis
    segmento = data[1000:2500] 

    if analizar:
        st.subheader(f"Análisis Registro {registro}")
        
        # PIPELINE
        pipe = AttractorPipeline(m=3, max_tau=40)
        res = pipe.run(segmento)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("Estado", res['R3']['regime'])
        c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
        
        # GRÁFICO
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(segmento[:1000], color='#58a6ff', lw=0.8)
        ax.set_title("Señal ECG Recuperada")
        st.pyplot(fig)
        
        # Monitor de valores (Diagnóstico Emanuel)
        st.write("**Muestras de control:**", segmento[:10])
    else:
        st.info("Presioná el botón para iniciar el análisis estructural.")

except Exception as e:
    st.error(f"Error en pantalla: {e}")
