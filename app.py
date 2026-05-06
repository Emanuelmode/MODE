import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR CON NORMALIZACIÓN Z-SCORE (MODE v5.1)
# ═══════════════════════════════════════════
def read_mitbih_zscore(record_id):
    """
    Lee PhysioNet y aplica normalización estadística para 
    maximizar el relieve del atractor extraño.
    """
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
    
    # Reconstrucción de canal único (MLII)
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    
    # 1. Señal en Voltaje Real
    raw_signal = (c1 - baseline) / gain
    
    # 2. CALIBRACIÓN DE RELIEVE (Z-Score)
    # Centramos la señal en 0 y escalamos por su propia varianza
    signal_final = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
    
    return signal_final, fs

# ═══════════════════════════════════════════
# INTERFAZ DE INVESTIGACIÓN
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE v5.1 · Relieve Fractal", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · Calibración de Relieve Estructural")

with st.sidebar:
    st.header("Configuración")
    registro = st.selectbox("Registro MIT-BIH:", ["100", "208", "214"])
    st.divider()
    m_dim = st.slider("Dimensión de Inmersión (m)", 2, 4, 3)
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)
    
    if ejecutar:
        st.info("Procesando dinámica del atractor...")

# Carga de datos
data, fs = read_mitbih_zscore(registro)

if data is not None:
    # Ventana de análisis (2000 muestras para estabilidad estadística)
    segmento = data[1000:3000]
    
    if ejecutar:
        # Ejecución del Pipeline con señal normalizada
        pipe = AttractorPipeline(m=m_dim, max_tau=40)
        res = pipe.run(segmento)
        
        # Panel de Resultados de Verdad Estructural
        col1, col2, col3 = st.columns(3)
        col1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        col2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
        col3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
        
        if res['R3']['R3_score'] > 0.5:
            st.balloons()
            st.success("¡Mejora en la resolución del atractor detectada!")

    # Gráfico de monitoreo de la señal normalizada
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:1000], color='#ff4b4b', lw=0.8)
    ax.set_title(f"Señal con Relieve Z-Score - Registro {registro}")
    ax.set_ylabel("Amplitud Estándar (σ)")
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)
    
    # Debug de valores para Emanuel
    with st.expander("Verificar flujo de datos (Z-Score)"):
        st.write("La media debería tender a 0:")
        st.code(f"Media: {np.mean(segmento):.6f}")
        st.write("Muestras:")
        st.code(segmento[:10])

else:
    st.error("No se pudieron cargar los datos del repositorio.")
