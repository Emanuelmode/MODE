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
    # Rutas dinámicas para tu repositorio
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No se encontró {record_id}.hea en /mitbih/")

    # 1. Leer metadatos del Header
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs, n_total = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    # 2. Leer binario .dat usando NumPy para evitar errores de índice
    dat_path = path_base + '.dat'
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Falta el archivo binario: {record_id}.dat")

    with open(dat_path, 'rb') as f:
        # Cargamos todo el archivo como un array de bytes
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 3. Procesamiento en bloque (Formato 212)
    # Calculamos el límite máximo que es múltiplo de 3 para no salirnos de rango
    max_groups = len(raw) // 3
    samples = []
    
    # Extraemos los bytes de forma segura
    b0 = raw[0:max_groups*3:3].astype(np.int16)
    b1 = raw[1:max_groups*3:3].astype(np.int16)
    b2 = raw[2:max_groups*3:3].astype(np.int16)
    
    # Reconstrucción de la señal (Lógica de Emanuel Duarte)
    s1 = b0 | ((b1 & 0x0F) << 8)
    s1[s1 >= 2048] -= 4096
    
    s2 = b2 | ((b1 & 0xF0) << 4)
    s2[s2 >= 2048] -= 4096
    
    # Combinamos las dos señales (intercaladas)
    signal_interleaved = np.empty(max_groups * 2)
    signal_interleaved[0::2] = s1
    signal_interleaved[1::2] = s2
    
    # Normalización final con los parámetros del header
    final_signal = (signal_interleaved - baseline) / gain
    return final_signal[:n_total], fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador Responsable: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("📂 Configuración")
    fuente = st.radio("Origen de datos:", ["Demo", "PhysioNet (MIT-BIH)"])
    
    x_data, fs_actual = None, 360.0

    if fuente == "PhysioNet (MIT-BIH)":
        rec = st.selectbox("Registro MIT-BIH:", ["100", "208", "214"])
        try:
            x_full, fs_actual = read_mitbih_safe(rec)
            x_data = x_full[:2000] # Ventana para análisis rápido
            st.success(f"Registro {rec} cargado correctamente.")
            
            st.divider()
            if st.button("Ejecutar Test 5 Ventanas"):
                pipe_t = AttractorPipeline(m=3, max_tau=40, verbose=False)
                logs = []
                for j in range(5):
                    inicio = j * 500
                    ventana = x_full[inicio : inicio + 1000]
                    res_v = pipe_t.run(ventana)
                    logs.append({
                        "Ventana": j+1,
                        "R³": f"{res_v['R3']['R3_score']:.3f}", 
                        "Coherente": 'SÍ' if res_v['R3']['coherent'] else 'NO', 
                        "Régimen": res_v['R3']['regime']
                    })
                st.table(logs)
        except Exception as e:
            st.error(f"Error técnico: {e}")
    else:
        # Señal de prueba
        x_data = np.sin(np.linspace(0, 50, 1000)) + np.random.normal(0, 0.05, 1000)

    st.divider()
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)

# ═══════════════════════════════════════════
# ÁREA DE RESULTADOS
# ═══════════════════════════════════════════
if ejecutar and x_data is not None:
    pipe = AttractorPipeline(m=3, max_tau=40, verbose=False)
    res = pipe.run(x_data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    c2.metric("τ (Lag)", res['tau'])
    c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x_data, color='#58a6ff', lw=0.8)
    ax.set_title("Señal en el Dominio del Tiempo")
    st.pyplot(fig)
