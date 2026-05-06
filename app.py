import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR PHYSIONET SEGURO (Anti-Index Error)
# ═══════════════════════════════════════════
def read_mitbih_safe(record_id):
    # Definimos rutas posibles en el repositorio
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No encontré el archivo {record_id}.hea en la carpeta /mitbih/")

    # 1. Leer metadatos del Header
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs, n_total = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    # 2. Leer binario .dat usando NumPy para mayor seguridad
    dat_path = path_base + '.dat'
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Falta el archivo binario: {record_id}.dat")

    with open(dat_path, 'rb') as f:
        # Cargamos todo como un array de bytes sin signo
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    samples = []
    # Calculamos el límite máximo múltiplo de 3 para no salirnos de rango
    max_idx = (len(raw) // 3) * 3
    
    for i in range(0, max_idx, 3):
        try:
            b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
            
            # Desempaquetado de bits Formato 212
            s1 = b0 | ((b1 & 0x0F) << 8)
            if s1 >= 2048: s1 -= 4096
            
            s2 = b2 | ((b1 & 0xF0) << 4)
            if s2 >= 2048: s2 -= 4096
            
            samples.extend([s1, s2])
        except IndexError:
            break # Si algo falla, salimos del bucle y usamos lo recolectado
    
    if not samples:
        raise ValueError("El archivo .dat está vacío o no se pudo leer correctamente.")

    # Convertimos y normalizamos según tus parámetros de investigación[cite: 1]
    signal = (np.array(samples) - baseline) / gain
    return signal[:n_total], fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador Responsable: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("📂 Entrada de Datos")
    fuente = st.radio("Origen:", ["Señal Demo", "PhysioNet (MIT-BIH)"])
    
    x_data, fs_actual = None, 360.0

    if fuente == "PhysioNet (MIT-BIH)":
        rec = st.selectbox("Registro:", ["100", "208", "214"])
        try:
            x_full, fs_actual = read_mitbih_safe(rec)
            x_data = x_full[:2000] # Vista previa inicial
            
            st.divider()
            if st.button("Ejecutar Test 5 Ventanas"):
                pipe_t = AttractorPipeline(m=3, max_tau=40, verbose=False)
                logs = []
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
            st.error(f"Error: {e}")
    else:
        x_data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)

    st.divider()
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)

if ejecutar and x_data is not None:
    pipe = AttractorPipeline(m=3, max_tau=40, verbose=False)
    res = pipe.run(x_data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    c2.metric("τ AMI", res['tau'])
    c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_data, color='#58a6ff', lw=0.7)
    ax.set_title("Fragmento de Señal Analizada")
    st.pyplot(fig)
