import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    return (c1 - baseline) / gain, fs

# ═══════════════════════════════════════════
# INTERFAZ DE INVESTIGACIÓN (MODE ε–τ)
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE v5.6 · High Fidelity", layout="wide")
st.title("🌀 MODE · Extractor de Verdad Estructural")
st.caption("Investigador: Emanuel Duarte · Configuración de Alta Resolución")

# Sidebar de control individual
with st.sidebar:
    st.header("Selección de Datos")
    registro = st.selectbox("Registro para analizar:", ["100", "208", "214"])
    st.divider()
    m_dim = st.slider("Dimensión de Inmersión (m)", 2, 4, 3)
    ejecutar = st.button("▶ ANALIZAR REGISTRO", type="primary", use_container_width=True)
    st.info("Configuración: Ventana 2000, Interp x4, Z-Score Local.")

# Carga y procesamiento
data, fs = read_mitbih_calibrated(registro)

if data is not None:
    # 1. Ventana de 2000 muestras (Estabilidad estructural)
    seg_raw = data[1000:3000]
    
    # 2. Super-resolución x4 (Cúbica) para eliminar escalones de cuantización
    x_old = np.linspace(0, 1, len(seg_raw))
    x_new = np.linspace(0, 1, len(seg_raw) * 4)
    f_interp = interp1d(x_old, seg_raw, kind='cubic')
    seg_smooth = f_interp(x_new)
    
    # 3. Normalización Local Absoluta (Media 0, Desviación 1)
    segmento = (seg_smooth - np.mean(seg_smooth)) / np.std(seg_smooth)
    
    if ejecutar:
        # Ejecución del Pipeline MODE
        pipe = AttractorPipeline(m=m_dim, max_tau=40)
        res = pipe.run(segmento)
        
        # Dashboard de resultados
        col1, col2, col3 = st.columns(3)
        col1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        col2.metric("Régimen", res['R3']['regime'].replace('_', ' ').title())
        col3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
        
        # Tabla resumida para copiar
        st.subheader("Datos para Reporte Técnico")
        st.code(f"ID: {registro} | R3: {res['R3']['R3_score']:.4f} | Régimen: {res['R3']['regime']} | Coh: {res['R3']['coherent']}")

    # Gráfico de monitoreo en azul (Alta Fidelidad)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(segmento[:800], color='#007acc', lw=1)
    ax.set_title(f"Visualización de Estructura - Registro {registro} (Procesamiento x4)")
    ax.set_ylabel("Z-Score")
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

else:
    st.error("Error al cargar el registro.")
