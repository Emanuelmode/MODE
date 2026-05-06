import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR PHYSIONET PROTEGIDO
# ═══════════════════════════════════════════
def read_mitbih_safe(record_id):
    # Definimos rutas posibles dentro de tu repo
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No encontré el archivo {record_id}.hea en /mitbih/")

    # 1. Leer metadatos del Header
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs, n_total = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    # 2. Leer binario .dat
    dat_path = path_base + '.dat'
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Falta el archivo {record_id}.dat")

    with open(dat_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    samples = []
    # Usamos un rango seguro para evitar el "index out of range"
    max_idx = (len(raw) // 3) * 3
    for i in range(0, max_idx, 3):
        # Lógica de desempaquetado de bits (Formato 212)
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        
        s1 = b0 | ((b1 & 0x0F) << 8)
        if s1 >= 2048: s1 -= 4096
        
        s2 = b2 | ((b1 & 0xF0) << 4)
        if s2 >= 2048: s2 -= 4096
        
        samples.extend([s1, s2])
    
    # Convertimos y normalizamos según el investigador Emanuel Duarte[cite: 1, 2]
    signal = (np.array(samples) - baseline) / gain
    return signal[:n_total], fs

# ═══════════════════════════════════════════
# INTERFAZ
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador Responsable: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("Configuración")
    fuente = st.radio("Origen:", ["Demo", "PhysioNet (MIT-BIH)"])
    
    x_data, fs_actual = None, 360.0

    if fuente == "PhysioNet (MIT-BIH)":
        rec = st.selectbox("Registro:", ["100", "208", "214"])
        try:
            x_full, fs_actual = read_mitbih_safe(rec)
            x_data = x_full[:2000] # Ventana inicial
            
            if st.button("Ejecutar Test 5 Ventanas"):
                pipe = AttractorPipeline(m=3, max_tau=40, verbose=False)
                res_test = []
                for i in range(5):
                    # Ventanas de 1000 muestras[cite: 2]
                    w = x_full[i*500 : i*500 + 1000]
                    r = pipe.run(w)
                    res_test.append({
                        "Ventana": i+1, "R³": f"{r['R3']['R3_score']:.3f}",
                        "Coh": "SÍ" if r['R3']['coherent'] else "NO",
                        "Régimen": r['R3']['regime']
                    })
                st.table(res_test)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        x_data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)

    st.divider()
    btn = st.button("▶ ANALIZAR PIPELINE", type="primary")

if btn and x_data is not None:
    p = AttractorPipeline(m=3, max_tau=40, verbose=False)
    res = p.run(x_data)
    
    c1, c2 = st.columns(2)
    c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    c2.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_data[:1000], color='#58a6ff', lw=0.7)
    st.pyplot(fig)
