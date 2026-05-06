import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR ATÓMICO (Anti-Index Error)
# ═══════════════════════════════════════════
def read_mitbih_safe(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No se encontró {record_id}.hea")

    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs = int(header[2])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    dat_path = path_base + '.dat'
    with open(dat_path, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    
    n_groups = len(raw_bytes) // 3 
    if n_groups == 0:
        raise ValueError("Archivo .dat vacío o corrupto.")

    # Desempaquetado Formato 212 Vectorizado
    b = raw_bytes[:n_groups*3].reshape(-1, 3)
    s1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    s1[s1 >= 2048] -= 4096
    s2 = b[:, 2].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0xF0) << 4)
    s2[s2 >= 2048] -= 4096
    
    signal = np.empty(n_groups * 2)
    signal[0::2] = s1
    signal[1::2] = s2
    
    return (signal - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · Pergamino, Argentina")

with st.sidebar:
    st.header("Configuración")
    registro = st.selectbox("Registro MIT-BIH:", ["100", "208", "214"])
    
    data_signal = None
    try:
        data_signal, freq = read_mitbih_safe(registro)
        st.success(f"Datos cargados: {len(data_signal)} muestras.")
    except Exception as e:
        st.error(f"Error de lectura: {e}")

    st.divider()
    btn_test = st.button("Ejecutar Test 5 Ventanas")
    btn_analizar = st.button("▶ ANALIZAR PIPELINE", type="primary")

# ═══════════════════════════════════════════
# LÓGICA DE DIAGNÓSTICO Y ANÁLISIS
# ═══════════════════════════════════════════
if data_signal is not None:
    # Seleccionamos el segmento a mostrar/analizar
    segmento = data_signal[1000:3000]

    if btn_test:
        pipe_t = AttractorPipeline(m=3, max_tau=40)
        logs = []
        for i in range(5):
            v = data_signal[i*500 : i*500 + 1000]
            r = pipe_t.run(v)
            logs.append({"T": i, "R³": f"{r['R3']['R3_score']:.4f}", "Estado": r['R3']['regime']})
        st.table(logs)

    if btn_analizar:
        st.subheader("📊 Resultados del Pipeline")
        
        # MONITOR DE DATOS (Para Emanuel)
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.write("**Muestras crudas (Primeras 10):**")
            st.code(segmento[:10])
        with col_diag2:
            st.write("**Estadísticas:**")
            st.write(f"Varianza: {np.var(segmento):.6f}")
            st.write(f"Rango: {np.ptp(segmento):.4f}")

        st.divider()

        # EJECUCIÓN
        pipe = AttractorPipeline(m=3, max_tau=40)
        res = pipe.run(segmento)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        m2.metric("τ AMI", res['tau'])
        m3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")

        # GRÁFICO DE SEÑAL
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(segmento, color='#58a6ff', lw=0.8)
        ax.set_title(f"Señal del Registro {registro}")
        st.pyplot(fig)
        
        if res['R3']['R3_score'] == 0:
            st.warning("⚠️ Atención: El Score 0 indica que la señal es constante o tiene varianza nula.")

else:
    st.info("Esperando carga de datos...")
