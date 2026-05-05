import warnings
warnings.filterwarnings('ignore')

import traceback
import io

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
# PALETA
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
    'teal':    '#39d353',
}

plt.rcParams.update({
    'figure.facecolor': P['bg'],   'axes.facecolor':  P['surface'],
    'axes.edgecolor':   P['border'],'axes.labelcolor': P['text'],
    'xtick.color':      P['border'],'ytick.color':     P['border'],
    'text.color':       P['text'],  'grid.color':      P['border'],
    'grid.alpha':       0.4,        'font.family':     'monospace',
    'font.size':        9,
})


# ═══════════════════════════════════════════
# SEÑALES DEMO
# ═══════════════════════════════════════════

def lorenz_ts(n=1000):
    x, y, z = 1.0, 1.0, 1.05
    out = []
    for _ in range(n):
        dx=10*(y-x); dy=x*(28-z)-y; dz=x*y-(8/3)*z
        x+=dx*.01; y+=dy*.01; z+=dz*.01
        out.append(x)
    return np.array(out)

def rossler_ts(n=1000):
    x, y, z = 1.0, 0.0, 0.0
    out = []
    for _ in range(n):
        dx=-y-z; dy=x+0.2*y; dz=0.2+z*(x-5.7)
        x+=dx*.05; y+=dy*.05; z+=dz*.05
        out.append(x)
    return np.array(out)

DEMOS = {
    'Lorenz (caótico clásico)':          lorenz_ts,
    'Rössler (caos débil)':              rossler_ts,
    'Mapa Logístico r=3.9 (caótico)':   lambda n=1000: _logistic_map(n, r=3.9),
    'Mapa Logístico r=3.5 (periódico)': lambda n=1000: _logistic_map(n, r=3.5),
    'Senoidal + ruido':                  lambda n=1000: (
        np.sin(2*np.pi*0.05*np.arange(n)) +
        0.1*np.random.default_rng(0).normal(size=n)
    ),
    'Ruido blanco':                      lambda n=1000: np.random.default_rng(1).normal(size=n),
}


# ═══════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════

def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def fig_signal(x, label):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5), facecolor=P['bg'])
    ax = axes[0]; ax.set_facecolor(P['surface'])
    ax.plot(x[:500], lw=0.7, color=P['accent'], alpha=0.9)
    ax.set_xlabel('t', fontsize=8); ax.set_ylabel('x(t)', fontsize=8)
    ax.set_title(f'{label[:40]}', fontsize=8, color=P['accent'])
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]; ax2.set_facecolor(P['surface'])
    fv = np.abs(np.fft.rfft(x - x.mean()))
    fr = np.fft.rfftfreq(len(x))
    ax2.semilogy(fr[1:], fv[1:], color=P['purple'], lw=0.7, alpha=0.85)
    ax2.set_xlabel('Frecuencia', fontsize=8); ax2.set_ylabel('|FFT|', fontsize=8)
    ax2.set_title('Espectro de potencia', fontsize=8, color=P['accent'])
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_epsilon(result):
    eps = result['epsilon_series']
    fig, ax = plt.subplots(figsize=(9, 2.2), facecolor=P['bg'])
    ax.set_facecolor(P['surface'])
    t = np.arange(len(eps))
    ax.fill_between(t, eps, alpha=0.2, color=P['accent'])
    ax.plot(t, eps, color=P['accent'], lw=0.8)
    ax.axhline(np.median(eps), color=P['orange'], lw=1, ls='--',
               label=f"ε̃={np.median(eps):.4f}")
    ax.set_xlabel('t (embedding)', fontsize=8); ax.set_ylabel('ε(t)', fontsize=8)
    ax.set_title('ε dinámico · escala local', fontsize=9, color=P['accent'])
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_attractor(result):
    Y   = result['embedding']
    eps = result['epsilon_series']
    if Y.shape[1] < 3:
        return None
    norm = Normalize(vmin=eps.min(), vmax=eps.max())
    cmap = plt.cm.plasma
    fig  = plt.figure(figsize=(6, 5), facecolor=P['bg'])
    ax   = fig.add_subplot(111, projection='3d', facecolor=P['surface'])
    n    = len(Y); step = max(1, n // 2000)
    Ys   = Y[::step]; es = eps[:n][::step]
    ax.scatter(Ys[:,0], Ys[:,1], Ys[:,2], c=es, cmap=cmap,
               norm=norm, s=0.9, alpha=0.7)
    ax.set_xlabel('y(t)', fontsize=7, color=P['text'])
    ax.set_ylabel('y(t-τ)', fontsize=7, color=P['text'])
    ax.set_zlabel('y(t-2τ)', fontsize=7, color=P['text'])
    ax.set_title(f"τ={result['tau']} · ε={result['epsilon']:.4f}",
                 color=P['accent'], fontsize=9, pad=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor(P['border'])
    ax.tick_params(colors=P['border'], labelsize=6)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cb.ax.tick_params(labelsize=6, colors=P['text'])
    cb.set_label('ε(t)', color=P['text'], fontsize=7)
    fig.tight_layout()
    return fig


def fig_metrics(result):
    r3    = result['R3']
    sm    = r3['stability_map']
    delta = r3['delta']
    keys   = ['lambda', 'D2', 'LZ', 'TE']
    labels = ['λ (Lyapunov)', 'D₂ (Dim. Corr.)', 'C_LZ (Compl.)', 'TE (Trans. Ent.)']
    grads  = [sm.get(k, {}).get('gradient', 0.0) for k in keys]
    stable = [sm.get(k, {}).get('stable', False) for k in keys]
    colors = [P['green'] if s else P['red'] for s in stable]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor=P['bg'])

    ax = axes[0]; ax.set_facecolor(P['surface'])
    bars = ax.barh(labels, grads, color=colors, alpha=0.8, height=0.5)
    ax.axvline(delta, color=P['orange'], lw=1.5, ls='--',
               label=f"δ={delta} ({r3['regime']})")
    ax.set_xlabel('|∂μ/∂τ| / max|μ|', fontsize=7)
    ax.set_title('Sensibilidad a τ por métrica', fontsize=9, color=P['accent'])
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, axis='x', alpha=0.3)
    mx = max(grads) if max(grads) > 0 else 0.01
    for bar, g in zip(bars, grads):
        ax.text(g + mx*0.02, bar.get_y() + bar.get_height()/2,
                f'{g:.4f}', va='center', ha='left', fontsize=7, color=P['text'])

    ax2 = axes[1]; ax2.set_facecolor(P['surface'])
    theta = np.linspace(np.pi, 0, 300)
    for i in range(len(theta)-1):
        c = plt.cm.RdYlGn(i/(len(theta)-1))
        ax2.fill_between(
            [np.cos(theta[i]), np.cos(theta[i+1])],
            [np.sin(theta[i])*0.7, np.sin(theta[i+1])*0.7],
            [np.sin(theta[i])*1.0, np.sin(theta[i+1])*1.0],
            color=c, alpha=0.85)
    score = r3['R3_score']
    angle = np.pi * (1 - score)
    ax2.annotate('', xy=(0.82*np.cos(angle), 0.82*np.sin(angle)),
                 xytext=(0,0),
                 arrowprops=dict(arrowstyle='->', color=P['text'], lw=2.5),
                 zorder=5)
    ax2.plot(0, 0, 'o', color=P['text'], ms=6, zorder=6)
    coh_color = P['green'] if r3['coherent'] else P['red']
    coh_text  = '✔  COHERENTE' if r3['coherent'] else '✘  NO COHERENTE'
    ax2.text(0, -0.22, f"R³ = {score:.3f}", ha='center',
             fontsize=16, fontweight='bold', color=P['accent'])
    ax2.text(0, -0.46, coh_text, ha='center', fontsize=10, color=coh_color)
    ax2.text(0, -0.66, r3['regime_desc'], ha='center', fontsize=8, color=P['orange'])
    ax2.text(-1.02, -0.1, '0', ha='center', fontsize=8, color=P['red'])
    ax2.text( 1.02, -0.1, '1', ha='center', fontsize=8, color=P['green'])
    ax2.text( 0,    1.05, '0.5', ha='center', fontsize=8, color=P['orange'])
    ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-0.9, 1.2)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('R³ Score · co-estabilización', fontsize=9, color=P['accent'])
    fig.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════
# APP
# ═══════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MODE · Attractor Pipeline",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stSidebar"]          { background-color: #161b22; }
    h1,h2,h3 { color: #58a6ff; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🌀 MODE · Attractor Pipeline")
    st.markdown(
        "**H1** ε dinámico &nbsp;·&nbsp; "
        "**H2** τ semidinamico &nbsp;·&nbsp; "
        "**H3** R³ descriptor de co-estabilización"
    )
    st.divider()

    # ── SIDEBAR ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")
        fuente = st.radio("Fuente de datos", ["Señal demo", "Cargar CSV"])

        if fuente == "Señal demo":
            demo_name = st.selectbox("Señal", list(DEMOS.keys()))
            N         = st.slider("Longitud N", 300, 2000, 800, 100)
            x_data    = DEMOS[demo_name](N)
            label     = demo_name
        else:
            uploaded = st.file_uploader("CSV — una columna numérica",
                                        type=['csv', 'txt'])
            if uploaded is None:
                st.info("Esperando archivo CSV…")
                st.stop()
            df_up  = pd.read_csv(uploaded, header=None)
            x_data = df_up.iloc[:, 0].dropna().values.astype(float)
            label  = uploaded.name

        st.divider()
        st.subheader("Parámetros")
        m_dim   = st.slider("Dimensión embedding m", 2, 5, 3)
        max_tau = st.slider("τ máximo (AMI)", 10, 80, 40, 5)
        run_btn = st.button("▶ Ejecutar pipeline", type="primary",
                            use_container_width=True)
        st.divider()
        st.markdown("""
**δ por régimen**
| Régimen | δ |
|---|---|
| Estable | 0.06 |
| Caos débil | 0.05 |
| Caótico | 0.08 |
| Hipercaótico | 0.15 |
| Ruidoso | 0.20 |
""")

    # ── EJECUCIÓN ────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("Calculando ε · τ · métricas · R³…"):
            try:
                pipe   = AttractorPipeline(m=m_dim, max_tau=max_tau, verbose=False)
                result = pipe.run(x_data, label=label)
                st.session_state['result'] = result
                st.session_state['x']      = x_data
                st.session_state['label']  = label
            except Exception as e:
                st.error(f"Error en pipeline: {e}")
                st.code(traceback.format_exc())
                st.stop()

    if 'result' not in st.session_state:
        st.info("👈 Elegí una señal y presioná **▶ Ejecutar pipeline**")
        st.stop()

    result = st.session_state['result']
    x      = st.session_state['x']
    label  = st.session_state['label']
    r3     = result['R3']
    mvals  = result['metrics']

    # ── MÉTRICAS ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("τ semidinamico", result['tau'])
    c2.metric("ε mediana",      f"{result['epsilon']:.5f}")
    c3.metric("R³ Score",       f"{r3['R3_score']:.4f}",
              delta="✔ coherente" if r3['coherent'] else "✘ no coherente")
    c4.metric("Régimen",        r3['regime'])
    c5.metric("δ activo",       r3['delta'])
    st.divider()

    # ── SEÑAL + ε ────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Serie temporal")
        st.image(to_png(fig_signal(x, label)), use_container_width=True)
    with col_r:
        st.subheader("ε(t) dinámico")
        st.image(to_png(fig_epsilon(result)), use_container_width=True)
    st.divider()

    # ── ATRACTOR + MÉTRICAS ──────────────────────────────────────────
    col_a, col_m = st.columns([2, 3])
    with col_a:
        st.subheader("Atractor reconstruido")
        fa = fig_attractor(result)
        if fa:
            st.image(to_png(fa), use_container_width=True)
    with col_m:
        st.subheader("Métricas + R³ gauge")
        st.image(to_png(fig_metrics(result)), use_container_width=True)
    st.divider()

    # ── TABLA ────────────────────────────────────────────────────────
    st.subheader("📊 Resultados")
    rows = [
        ('τ semidinamico',       result['tau'],              '—'),
        ('ε mediana',            f"{result['epsilon']:.5f}", '—'),
        ('Régimen',              r3['regime_desc'],           '—'),
        ('δ activo',             r3['delta'],                 '—'),
        ('R³ Score',             r3['R3_score'],              '≥0.75 → coherente'),
        ('λ (Lyapunov)',         mvals.get('lambda'),         '<0 estable · >0 caos'),
        ('D₂ (dim. corr.)',      mvals.get('D2'),             'Lorenz ≈ 2.05'),
        ('C_LZ (complejidad)',   mvals.get('LZ'),             '0=ord · 1=máx compl.'),
        ('TE (trans. entropía)', mvals.get('TE'),             'Flujo inf. interno'),
    ]
    df_res = pd.DataFrame(rows, columns=['Variable', 'Valor', 'Referencia'])
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    st.divider()

    # ── BASELINES ────────────────────────────────────────────────────
    st.subheader("📐 Baselines de comparación")
    st.caption("Métodos estándar — para contrastar con R³")

    x_bl = st.session_state['x']
    x_bl = (x_bl - x_bl.mean()) / (x_bl.std() + 1e-12)

    # 1. FOURIER
    fft_vals  = np.abs(np.fft.rfft(x_bl))
    freqs     = np.fft.rfftfreq(len(x_bl))
    psd       = fft_vals**2
    psd_norm  = psd / (psd.sum() + 1e-12)
    # Entropía espectral (Shannon sobre PSD)
    h_spectral = float(-np.sum(psd_norm[psd_norm > 0] *
                               np.log2(psd_norm[psd_norm > 0])))
    h_spec_max = np.log2(len(psd_norm))
    h_spec_norm = h_spectral / (h_spec_max + 1e-12)
    peak_freq   = float(freqs[1:][np.argmax(fft_vals[1:])])
    peak_amp    = float(fft_vals[1:].max())
    flat_ratio  = float(np.std(fft_vals[1:]) / (np.mean(fft_vals[1:]) + 1e-12))
    espectro_plano = "SÍ (ruido/caos)" if flat_ratio < 1.5 else "NO (estructura)"

    # 2. SHANNON sobre señal
    bins_sh   = 32
    hist, _   = np.histogram(x_bl, bins=bins_sh, density=True)
    hist_norm = hist / (hist.sum() + 1e-12)
    h_shannon = float(-np.sum(hist_norm[hist_norm > 0] *
                              np.log2(hist_norm[hist_norm > 0])))
    h_sh_max  = np.log2(bins_sh)
    h_sh_norm = h_shannon / (h_sh_max + 1e-12)

    # 3. D2 clásico (ya calculado en pipeline)
    d2_val    = mvals.get('D2')

    # Tabla comparativa
    bl_rows = [
        ('── FOURIER ──',              '',         ''),
        ('Frecuencia pico',            f"{peak_freq:.4f}",   'Hz normalizado'),
        ('Amplitud pico FFT',          f"{peak_amp:.3f}",    ''),
        ('Entropía espectral',         f"{h_spectral:.3f}",  'bits'),
        ('Entropía espectral norm.',   f"{h_spec_norm:.4f}", '0=orden · 1=caos/ruido'),
        ('¿Espectro plano?',           espectro_plano,       'std/mean FFT'),
        ('── SHANNON ──',              '',         ''),
        ('H Shannon (señal)',          f"{h_shannon:.4f}",   'bits'),
        ('H Shannon normalizada',      f"{h_sh_norm:.4f}",   '0=orden · 1=máx desorden'),
        ('── D₂ CLÁSICO ──',          '',         ''),
        ('D₂ (dim. correlación)',      f"{d2_val:.4f}" if d2_val else 'nan',
                                                              'Lorenz≈2.05 · Ruido→alto'),
        ('── R³ (nuestro) ──',        '',         ''),
        ('R³ Score',                   f"{r3['R3_score']:.4f}", '≥0.75 → coherente'),
        ('Coherente',                  '✔' if r3['coherent'] else '✘', ''),
        ('Régimen detectado',          r3['regime_desc'],    'Con δ semidinamico'),
    ]

    df_bl = pd.DataFrame(bl_rows, columns=['Métrica', 'Valor', 'Referencia'])
    st.dataframe(df_bl, use_container_width=True, hide_index=True)

    # Export baseline
    st.download_button("⬇ Baselines CSV",
                       df_bl.to_csv(index=False).encode(),
                       f"baselines_{label[:20]}.csv", mime="text/csv")
    st.divider()

    # ── EXPORT ───────────────────────────────────────────────────────
    ce1, ce2 = st.columns(2)
    with ce1:
        st.download_button("⬇ Resultados CSV",
                           df_res.to_csv(index=False).encode(),
                           "resultados_pipeline.csv", mime="text/csv")
    with ce2:
        emb    = result['embedding']
        cols_  = [f'y_t-{i*result["tau"]}' for i in range(emb.shape[1])]
        df_emb = pd.DataFrame(emb, columns=cols_)
        df_emb['epsilon'] = result['epsilon_series'][:len(df_emb)]
        st.download_button("⬇ Embedding CSV",
                           df_emb.to_csv(index=False).encode(),
                           "embedding.csv", mime="text/csv")

    st.caption(
        "ε dinámico (H1) · τ semidinamico AMI (H2) · R³ descriptor (H3) · "
        "δ: Peng 1995 · Grassberger & Procaccia 1983 · "
        "Mantegna & Stanley 1999 · Schreiber 2000"
    )


if __name__ == '__main__':
    main()
