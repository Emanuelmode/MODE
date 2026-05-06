import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR PHYSIONET VECTORIZADO (Anti-Index Error)
# ═══════════════════════════════════════════
def read_mitbih_atomic(record_id):
    # Buscamos la ruta del archivo
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        raise FileNotFoundError(f"No se encontró {record_id}.hea en la carpeta /mitbih/")

    # 1. Leer Header para obtener ganancia y frecuencia
    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs, n_total = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    # 2. Leer .dat de forma atómica
    dat_path = path_base + '.dat'
    with open(dat_path, 'rb') as f:
        # Cargamos el archivo completo como array de bytes
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 3. Lógica de Seguridad Atómica: 
    # Calculamos cuántos grupos de 3 bytes existen realmente
    num_groups = len(raw) // 3 
    
    if num_groups == 0:
        raise ValueError("El archivo .dat está vacío o corrupto.")

    # Tomamos solo los bytes que completan grupos de 3 (adiós Index Error)
    b0 = raw[0:num_groups*3:3].astype(np.int16)
    b1 = raw[1:num_groups*3:3].astype(np.int16)
    b2 = raw[2:num_groups*3:3].astype(np.int16)
    
    # Desempaquetado de bits Formato 212
    # s1 usa b0 y los 4 bits bajos de b1
    s1 = b0 | ((b1 & 0x0F) << 8)
    s1[s1 >= 2048] -= 4096
    
    # s2 usa b2 y los 4 bits altos de b1
    s2 = b2 | ((b1 & 0xF0) << 4)
    s2[s2 >= 2048] -= 4096
    
    # Intercalamos para obtener la señal completa
    interleaved = np.empty(num_groups * 2)
    interleaved[0::2] = s1
    interleaved[1::2] = s2
    
    # Normalizamos y cortamos al total declarado
    final_signal = (interleaved - baseline) / gain
    return final_signal[:n_total], fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Explorer", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Responsable Intelectual: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("📂 Configuración")
    rec = st.selectbox("Registro MIT-BIH:", ["100", "208", "214"])
    
    try:
        x_full, fs = read_mitbih_atomic(rec)
        st.success(f"Registro {rec} cargado.")
        
        st.divider()
        test_btn = st.button("Ejecutar Test 5 Ventanas")
        if test_btn:
            pipe_t = AttractorPipeline(m=3, max_tau=40)
            resultados = []
            for i in range(5):
                ventana = x_full[i*500 : i*500 + 1000]
                r = pipe_t.run(ventana)
                resultados.append({
                    "Ventana": i+1, 
                    "R³": f"{r['R3']['R3_score']:.3f}", 
                    "Coherencia": "✔" if r['R3']['coherent'] else "✘"
                })
            st.table(resultados)
    except Exception as e:
        st.error(f"Error: {e}")

    analizar = st.button("▶ ANALIZAR PIPELINE", type="primary")

if analizar and 'x_full' in locals():
    pipe = AttractorPipeline(m=3, max_tau=40)
    res = pipe.run(x_full[:1000])
    
    col1, col2 = st.columns(2)
    col1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
    col2.metric("Régimen", res['R3']['regime'])
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_full[:1000], color='#58a6ff')
    st.pyplot(fig)
