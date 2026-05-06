import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# LECTOR DE SEÑAL CALIBRADO
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
    # Reconstrucción de un solo canal (MLII) para evitar ruido de fase
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    return (c1 - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE v5.2 · Local Normalization", layout="wide")
st.title("🌀 MODE · Attractor Explorer")
st.caption("Investigador: Emanuel Duarte · 2026")

with st.sidebar:
    st.header("Configuración de Análisis")
    registro = st.selectbox("Registro:", ["100", "208", "214"])
    st.divider()
    m_dim = st.slider("Dimensión m", 2, 4, 3)
    ejecutar = st.button("▶ ANALIZAR PIPELINE", type="primary", use_container_width=True)

# 1. Carga inicial de datos
data, fs = read_mitbih_calibrated(registro)

if data is not None:
    # 2. SELECCIÓN Y NORMALIZACIÓN LOCAL (CORRECCIÓN EMANUEL)
    # Tomamos la ventana y la forzamos a Media 0 y STD 1
    segmento_raw = data[1000:3000]
    segmento = (segmento_raw - np.mean(segmento_raw)) / np.std(segmento_raw)
    
    if ejecutar:
        # Ejecución del Pipeline con señal centrada
        pipe = AttractorPipeline(m=m_dim, max_tau=40)
        res = pipe.run(segmento)
        
        # Panel de Resultados
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
        c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
        
        # Monitor de Calibración
        st.write(f"**Media Local:** {np.mean(segmento):.10e} (Objetivo: 0)")
        st.write(f"**Desviación Estándar:** {np.std(segmento):.4f} (Objetivo: 1)")
        st.write("**Muestras de Entrada (Normalizadas):**")
        st.code(segmento[:10])

    # 3. Gráfico de Monitoreo (Verde para indicar señal centrada)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:1000], color='#00ff41', lw=0.8)
    ax.axhline(0, color='white', linestyle='--', alpha=0.3) # Línea de referencia en cero
    ax.set_title(f"Señal con Normalización Local Forzada - Registro {registro}")
    ax.set_ylabel("Amplitud (Z-Score)")
    st.pyplot(fig)

else:
    st.error("No se pudieron cargar los datos de PhysioNet.")
