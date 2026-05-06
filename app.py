import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pipeline import AttractorPipeline

# Lector optimizado (mismo de la v5.3)
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

st.set_page_config(page_title="MODE v5.4 · Comparativo", layout="wide")
st.title("🌀 MODE · Análisis Comparativo de Atractores")
st.caption(f"Investigador: Emanuel Duarte · Pergamino, {2026}")

# Sidebar con selección múltiple
with st.sidebar:
    st.header("Parámetros")
    registros_inst = st.multiselect("Seleccionar Registros:", ["100", "208", "214"], default=["100"])
    m_dim = st.slider("Dimensión m", 2, 5, 4) # Actualizado a 4 por tu prueba
    ejecutar = st.button("▶ EJECUTAR COMPARATIVA", type="primary", use_container_width=True)

if ejecutar:
    resultados = []
    
    # Procesamos cada registro seleccionado
    for reg in registros_inst:
        data, _ = read_mitbih_calibrated(reg)
        if data is not None:
            # Ventana + Super-resolución + Normalización
            seg_raw = data[1000:2500]
            x_old = np.linspace(0, 1, len(seg_raw))
            x_new = np.linspace(0, 1, len(seg_raw) * 4)
            f_interp = interp1d(x_old, seg_raw, kind='cubic')
            seg_smooth = f_interp(x_new)
            segmento = (seg_smooth - np.mean(seg_smooth)) / np.std(seg_smooth)
            
            # Pipeline
            pipe = AttractorPipeline(m=m_dim, max_tau=40)
            res = pipe.run(segmento)
            
            # Guardamos para la tabla
            resultados.append({
                "Registro": reg,
                "R³ Score": f"{res['R3']['R3_score']:.4f}",
                "Régimen": res['R3']['regime'].replace('_', ' ').title(),
                "Coherencia": "SÍ" if res['R3']['coherent'] else "NO"
            })
    
    # Mostrar tabla de Verdad Estructural
    st.subheader("📊 Resultados Comparativos")
    st.table(resultados)
    
    if len(resultados) > 1:
        st.info("💡 Compará los R³ Scores. Una diferencia > 0.1 indica un cambio significativo en la dinámica estructural.")
else:
    st.info("Seleccioná al menos dos registros (ej. 100 y 208) para ver la sensibilidad del sistema.")
