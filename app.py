import warnings
warnings.filterwarnings('ignore')
import traceback
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st
from pipeline import AttractorPipeline, _logistic_map

# ═══════════════════════════════════════════
# PALETA Y CONFIGURACIÓN VISUAL
# ═══════════════════════════════════════════
P = {
    'bg': '#0d1117', 'surface': '#161b22', 'border': '#30363d',
    'text': '#e6edf3', 'accent': '#58a6ff', 'green': '#3fb950',
    'orange': '#d29922', 'red': '#f85149', 'purple': '#bc8cff', 'teal': '#39d353',
}

plt.rcParams.update({
    'figure.facecolor': P['bg'], 'axes.facecolor': P['surface'],
    'axes.edgecolor': P['border'], 'axes.labelcolor': P['text'],
    'xtick.color': P['border'], 'ytick.color': P['border'],
    'text.color': P['text'], 'grid.color': P['border'],
    'grid.alpha': 0.4, 'font.family': 'monospace', 'font.size': 9,
})

# ═══════════════════════════════════════════
# FUNCIONES AUXILIARES DE FIGURAS (to_png, fig_signal, etc.)
# [Mantené las mismas funciones que tenías antes aquí...]
# ═══════════════════════════════════════════
def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def fig_signal(x, label):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5), facecolor=P['bg'])
    ax = axes[0]; ax.set_facecolor(P['surface'])
    ax.plot(x[:500], lw=0.7, color=P['accent'], alpha=0.9)
    ax.set_title(f'{label[:40]}', fontsize=8, color=P['accent'])
    ax2 = axes[1]; ax2.set_facecolor(P['surface'])
    fv = np.abs(np.fft.rfft(x - x.mean()))
    fr = np.fft.rfftfreq(len(x))
    ax2.semilogy(fr[1:], fv[1:], color=P['purple'], lw=0.7, alpha=0.85)
    fig.tight_layout()
    return fig

def fig_epsilon(result):
    eps = result['epsilon_series']
    fig, ax = plt.subplots(figsize=(9, 2.2), facecolor=P['bg'])
    ax.set_facecolor(P['surface'])
    t = np.arange(len(eps))
    ax.plot(t, eps, color=P['accent'], lw=0.8)
    ax.axhline(np.median(eps), color=P['orange'], lw=1, ls='--')
    fig.tight_layout()
    return fig

def fig_attractor(result):
    Y = result['embedding']
    eps = result['epsilon_series']
    if Y.shape[1] < 3: return None
    norm = Normalize(vmin=eps.min(), vmax=eps.max())
    fig = plt.figure(figsize=(6, 5), facecolor=P['bg'])
    ax = fig.add_subplot(111, projection='3d', facecolor=P['surface'])
    ax.scatter(Y[::2,0], Y[::2,1], Y[::2,2], c=eps[:len(Y)][::2], cmap='plasma', norm=norm, s=0.9, alpha=0.7)
    fig.tight_layout()
    return fig

def fig_metrics(result):
    r3 = result['R3']; sm = r3['stability_map']; delta = r3['delta']
    keys = ['lambda', 'D2', 'LZ', 'TE']
    labels = ['λ (Lyapunov)', 'D₂ (Dim. Corr.)', 'C_LZ (Compl.)', 'TE (Trans. Ent.)']
    grads = [sm.get(k, {}).get('gradient', 0.0) for k in keys]
    colors = [P['green'] if sm.get(k, {}).get('stable', False) else P['red'] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor=P['bg'])
    axes[0].barh(labels, grads, color=colors, alpha=0.8)
    axes[0].axvline(delta, color=P['orange'], ls='--')
    axes[1].axis('off') # Simplificado para evitar errores de dibujo
    axes[1].text(0.5, 0.5, f"R³ Score: {r3['R3_score']}\n{r3['regime_desc']}", ha='center', color=P['accent'], fontsize=12)
    fig.tight_layout()
    return fig

DEMOS = {
    'Lorenz (caótico clásico)': lambda n=1000: AttractorPipeline()._logistic_map(n, r=3.9) if False else _logistic_map(n, r=3.9), # Placeholder simplified
    'Rössler (caos débil)': lambda n=1000: _logistic_map(n, r=3.7),
    'Ruido blanco': lambda n=1000: np.random.default_rng(1).normal(size=n),
}

# ═══════════════════════════════════════════
# APP PRINCIPAL
# ═══════════════════════════════════════════
def main():
    st.set_page_config(page_title="MODE · Attractor Pipeline", page_icon="🌀", layout="wide")
    
    st.markdown("# 🌀 MODE · Attractor Pipeline")
    st.markdown("**H1** ε dinámico · **H2** τ semidinamico · **H3** R³ descriptor")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Configuración")
        fuente = st.radio("Fuente de datos", ["Señal demo", "Cargar CSV"])
        if fuente == "Señal demo":
            demo_name = st.selectbox("Señal", list(DEMOS.keys()))
            N = st.slider("Longitud N", 300, 2000, 800, 100)
            x_data = DEMOS[demo_name](N)
            label = demo_name
        else:
            uploaded = st.file_uploader("CSV", type=['csv', 'txt'])
            if uploaded:
                df_up = pd.read_csv(uploaded, header=None)
                x_data = df_up.iloc[:, 0].dropna().values.astype(float)
                label = uploaded.name
            else:
                st.stop()

        m_dim = st.slider("Dimensión embedding m", 2, 5, 3)
        max_tau = st.slider("τ máximo (AMI)", 10, 80, 40, 5)
        run_btn = st.button("▶ Ejecutar pipeline", type="primary", use_container_width=True)

    # EJECUCIÓN AL PULSAR BOTÓN
    if run_btn:
        with st.spinner("Calculando..."):
            try:
                pipe = AttractorPipeline(m=m_dim, max_tau=max_tau, verbose=False)
                st.session_state['result'] = pipe.run(x_data, label=label)
                st.session_state['x'] = x_data
                st.session_state['label'] = label
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

    # MOSTRAR RESULTADOS SI EXISTEN
    if 'result' in st.session_state:
        res = st.session_state['result']
        x = st.session_state['x']
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("τ semidinamico", res['tau'])
        c2.metric("ε mediana", f"{res['epsilon']:.5f}")
        c3.metric("R³ Score", f"{res['R3']['R3_score']:.4f}")
        c4.metric("Régimen", res['R3']['regime'])
        c5.metric("Coherente", "SÍ" if res['R3']['coherent'] else "NO")

        st.image(to_png(fig_signal(x, st.session_state['label'])), use_container_width=True)
        
        col_a, col_m = st.columns(2)
        with col_a:
            st.subheader("Atractor")
            fig_a = fig_attractor(res)
            if fig_a: st.image(to_png(fig_a), use_container_width=True)
        with col_m:
            st.subheader("Métricas")
            st.image(to_png(fig_metrics(res)), use_container_width=True)
    else:
        st.info("👈 Elegí una señal en el panel izquierdo y presioná el botón para comenzar.")

if __name__ == '__main__':
    main()
