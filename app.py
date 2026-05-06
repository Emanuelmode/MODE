import warnings
warnings.filterwarnings('ignore')

import traceback
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import streamlit as st
from pipeline import AttractorPipeline, _logistic_map

# ═══════════════════════════════════════════
# CONFIGURACIÓN Y PALETA
# ═══════════════════════════════════════════

P = {
    'bg':      '#0d1117',
    'surface': '#161b22',
    'border':  '#30363d',
    'text':    '#e6edf3',
    'accent':  '#58a6ff',
    'green':   '#3fb950',
    'orange':  '#d29922',
    'red':     '#f85149',
    'purple':  '#bc8cff',
}

plt.rcParams.update({
    'figure.facecolor': P['bg'],   'axes.facecolor':  P['surface'],
    'axes.edgecolor':   P['border'],'axes.labelcolor': P['text'],
    'xtick.color':      P['border'],'ytick.color':     P['border'],
    'text.color':       P['text'],  'grid.color':      P['border'],
    'grid.alpha':       0.4,        'font.family':     'monospace',
    'font.size':        9,
})

WATERMARK = "Investigador: Emanuel Duarte"

def add_watermark(fig):
    fig.text(0.5, 0.5, WATERMARK, fontsize=11, color='white', alpha=0.10,
             ha='center', va='center', rotation=30, transform=fig.transFigure)
    fig.text(0.99, 0.01, WATERMARK, fontsize=7, color='white', alpha=0.30,
             ha='right', va='bottom', transform=fig.transFigure)
    return fig

# ═══════════════════════════════════════════
# LECTOR PHYSIONET (MIT-BIH)
# ═══════════════════════════════════════════

def read_mitbih(record_id):
    """Lee registros MIT-BIH usando la lógica de bits de Emanuel Duarte."""
    BASE = 'mitbih/mit-bih-arrhythmia-database-1.0.0/'
    hea_path = os.path.join(BASE, f"{record_id}.hea")
    dat_path = os.path.join(BASE, f"{record_id}.dat")
    
    if not os.path.exists(hea_path):
        raise FileNotFoundError(f"No se encontró el header: {hea_path}")

    with open(hea_path) as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    fs, n = int(header[2]), int(header[3])
    sig_info = lines[1].strip().split()
    gain, baseline = float(sig_info[2]), int(sig_info[4])
    
    with open(dat_path, 'rb') as f:
        raw = f.read()
    
    samples = []
    i = 0
    while i + 2 < len(raw):
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        s1 = b0 | ((b1 & 0x0F) << 8); s1 = s1 - 4096 if s1 >= 2048 else s1
        s2 = b2 | ((b1 & 0xF0) << 4); s2 = s2 - 4096 if s2 >= 2048 else s2
        samples.extend([s1, s2]); i += 3
    
    mlii = np.array(samples[::2][:n])
    return (mlii - baseline) / gain, fs

# ═══════════════════════════════════════════
# FUNCIONES DE APOYO (DEMOS Y FIGURAS)
# ═══════════════════════════════════════════

def lorenz_ts(n=1000):
    x, y, z = 1.0, 1.0, 1.05
    out = []
    for _ in range(n):
        dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-(8/3)*z
        x+=dx*.01; y+=dy*.01; z+=dz*.01
        out.append(x)
    return np.array(out)

DEMOS = {
    'Lorenz (caótico clásico)': lorenz_ts,
    'Mapa Logístico r=3.9': lambda n=1000: _logistic_map(n, r=3.9),
    'Ruido blanco': lambda n=1000: np.random.default_rng(1).normal(size=n),
}

def to_png(fig):
    add_watermark(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return buf.read()

def fig_signal(x, label):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))
    axes[0].plot(x[:500], lw=0.7, color=P['accent']); axes[0].set_title(label[:30])
    fv = np.abs(np.fft.rfft(x - x.mean()))
    axes[1].semilogy(fv, color=P['purple'], lw=0.7); axes[1].set_title('Espectro FFT')
    fig.tight_layout()
    return fig

# (Aquí podrías incluir fig_attractor y fig_metrics si las necesitas, simplificado por brevedad)

# ═══════════════════════════════════════════
# APP PRINCIPAL
# ═══════════════════════════════════════════

def main():
    st.set_page_config(page_title="MODE · Attractor Pipeline", page_icon="🌀", layout="wide")
    st.markdown("# 🌀 MODE · Attractor Pipeline")
    st.caption(f"{WATERMARK} · Pergamino, Argentina · 2026")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Configuración")
        fuente = st.radio("Fuente de datos", ["Señal demo", "PhysioNet (MIT-BIH)", "Cargar CSV"])

        if fuente == "Señal demo":
            demo_name = st.selectbox("Señal", list(DEMOS.keys()))
            x_data = DEMOS[demo_name](1000)
            label = demo_name
        elif fuente == "PhysioNet (MIT-BIH)":
            record_id = st.selectbox("Registro", ["100", "208", "214"])
            try:
                x_data, fs = read_mitbih(record_id)
                x_data = x_data[:1000] # Ventana inicial
                label = f"MIT-BIH Rec {record_id}"
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        else:
            uploaded = st.file_uploader("Subir CSV", type=['csv'])
            if not uploaded: st.stop()
            x_data = pd.read_csv(uploaded, header=None).iloc[:, 0].values
            label = uploaded.name

        st.divider()
        m_dim = st.slider("Dimensión m", 2, 5, 3)
        max_tau = st.slider("τ máximo", 10, 80, 40)
        run_btn = st.button("▶ Ejecutar pipeline", type="primary", use_container_width=True)

        st.divider()
        st.subheader("🛠️ Diagnóstico Rápido")
        if st.button("Ejecutar Test 5 Ventanas (Rec 100)"):
            try:
                sig, fs_test = read_mitbih("100")
                test_pipe = AttractorPipeline(m=3, max_tau=40, verbose=False)
                rows = []
                for i in range(5):
                    start = i * 500
                    w = sig[start:start+1000]
                    r = test_pipe.run(w)
                    rows.append({
                        "t(s)": f"{start/fs_test:.1f}", "τ": r['tau'],
                        "R³": f"{r['R3']['R3_score']:.3f}",
                        "Coh": '✔' if r['R3']['coherent'] else '✘',
                        "Régimen": r['R3']['regime']
                    })
                st.table(rows)
            except Exception as e:
                st.error(f"Error en diagnóstico: {e}")

    # Ejecución
    if run_btn:
        with st.spinner("Analizando..."):
            pipe = AttractorPipeline(m=m_dim, max_tau=max_tau, verbose=False)
            st.session_state['res'] = pipe.run(x_data, label=label)
            st.session_state['x'] = x_data

    if 'res' in st.session_state:
        res = st.session_state['res']
        st.subheader(f"Resultados: {label}")
        c1, c2, c3 = st.columns(3)
        c1.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c2.metric("τ AMI", res['tau'])
        c3.metric("Coherencia", "SÍ" if res['R3']['coherent'] else "NO")
        
        st.image(to_png(fig_signal(st.session_state['x'], label)), use_container_width=True)

if __name__ == '__main__':
    main()