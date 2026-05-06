import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import pandas as pd # Para manejar la tabla mejor
from scipy.interpolate import interp1d
from pipeline import AttractorPipeline

# Lector Estándar
def read_mitbih_calibrated(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    if not path_base: return None
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
    return (c1 - baseline) / gain

st.set_page_config(page_title="MODE v5.5 · Reporte Técnico", layout="wide")
st.title("🌀 MODE · Reporte de Inteligencia Estructural")
st.caption("Investigador Responsable: Emanuel Duarte")

registros_lista = ["100", "208", "214"]

if st.button("🚀 GENERAR TABLA COMPARATIVA (S3)", type="primary", use_container_width=True):
    res_final = []
    progreso = st.progress(0)
    
    for i, reg in enumerate(registros_lista):
        st.write(f"Procesando Registro {reg}...")
        data = read_mitbih_calibrated(reg)
        
        if data is not None:
            # Ventana reducida para evitar el bug de memoria (1000 muestras originales)
            seg_raw = data[1000:2000]
            
            # Interpolación controlada (x2 en lugar de x4 para fluidez sin lag)
            x_old = np.linspace(0, 1, len(seg_raw))
            x_new = np.linspace(0, 1, len(seg_raw) * 2)
            f_interp = interp1d(x_old, seg_raw, kind='cubic')
            seg_smooth = f_interp(x_new)
            
            # Normalización Local
            segmento = (seg_smooth - np.mean(seg_smooth)) / np.std(seg_smooth)
            
            # Pipeline (m=3 para eficiencia)
            pipe = AttractorPipeline(m=3, max_tau=40)
            res = pipe.run(segmento)
            
            res_final.append({
                "ID": reg,
                "R³ Score": res['R3']['R3_score'],
                "Régimen": res['R3']['regime'].replace('_', ' ').title(),
                "Coherencia": "SÍ" if res['R3']['coherent'] else "NO"
            })
        progreso.progress((i + 1) / len(registros_lista))

    # Presentación final
    st.divider()
    df_res = pd.DataFrame(res_final)
    st.table(df_res)
    
    # Análisis Automático para la Patente
    st.success("✅ Análisis Completo")
    st.markdown("""
    ### Notas de Investigación:
    * El **R³ Score** mide la fidelidad de la estructura fractal.
    * El **Régimen** identifica la firma dinámica (Estabilidad vs Caos).
    * La **Coherencia** valida si el atractor es un sistema físico real.
    """)
