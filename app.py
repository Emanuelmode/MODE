import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════
# LECTOR DE CALIBRACIÓN (Filtro Anti-Ruido)
# ═══════════════════════════════════════════
def read_calibration(record_id):
    bases = [f'mitbih/{record_id}', f'mitbih/mit-bih-arrhythmia-database-1.0.0/{record_id}', str(record_id)]
    path_base = next((p for p in bases if os.path.exists(p + '.hea')), None)
    
    if not path_base:
        return None, "Error: No se encontró el archivo .hea"

    with open(path_base + '.hea', 'r') as f:
        lines = f.readlines()
    
    # Extraer parámetros de calibración
    header = lines[0].strip().split()
    sig_info = lines[1].strip().split()
    gain = float(sig_info[2])
    baseline = int(sig_info[4])
    
    dat_path = path_base + '.dat'
    with open(dat_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Calibración de empaquetado:
    # El error anterior mezclaba canales. Aquí separamos estrictamente el Canal 1.
    n_groups = len(raw) // 3
    b = raw[:n_groups*3].reshape(-1, 3)
    
    # Reconstrucción limpia del canal MLII
    c1 = b[:, 0].astype(np.int16) | ((b[:, 1].astype(np.int16) & 0x0F) << 8)
    c1[c1 >= 2048] -= 4096
    
    # Aplicar ganancia y baseline oficial
    signal_calibrada = (c1 - baseline) / gain
    return signal_calibrada, "Lectura exitosa"

# ═══════════════════════════════════════════
# INTERFAZ DE MONITOREO
# ═══════════════════════════════════════════
st.set_page_config(page_title="MODE · Calibrador", layout="wide")
st.title("🛠️ MODE · Calibración de Señal")
st.caption("Investigador: Emanuel Duarte · Objetivo: Eliminar ruido de lectura")

registro = st.selectbox("Seleccionar Registro para Calibrar:", ["100", "208", "214"])

signal, msg = read_calibration(registro)

if signal is not None:
    st.success(msg)
    
    # Gráfico de Alta Resolución para ver el ruido
    fig, ax = plt.subplots(figsize=(12, 4))
    # Mostramos 1000 muestras para ver si hay "zig-zag"
    ax.plot(signal[1000:2000], color='#ff4b4b', lw=1, label="Señal Calibrada")
    ax.set_title(f"Monitoreo de Ruido - Registro {registro}")
    ax.set_ylabel("Voltaje (mV)")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    
    # MONITOR DE VALORES (Crucial para ver si hay saltos artificiales)
    st.subheader("📋 Valores Crudos de Salida")
    st.write("Si estos valores no saltan bruscamente de -0.3 a -0.1 en cada línea, la calibración es correcta:")
    st.code(signal[1000:1015])

else:
    st.error(msg)
