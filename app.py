import streamlit as st
import numpy as np
import pandas as pd
from pipeline import AttractorPipeline

st.set_page_config(page_title='MODE Pipeline v2', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')

BG = '#0b0f14'
SURFACE = '#111821'
CARD = '#141d29'
BORDER = '#233041'
TEXT = '#d7e0ea'
MUTED = '#7f8b98'
ACCENT = '#4aa9ff'
GREEN = '#2dbe6c'
RED = '#e5534b'
ORANGE = '#d4a017'
PURPLE = '#a371f7'
TEAL = '#2fb392'

st.markdown(f"""
<style>
body, [class*='css'] {{ background: {BG}; color: {TEXT}; }}
[data-testid='stAppViewContainer'] {{ background: {BG}; }}
[data-testid='stSidebar'] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
h1, h2, h3 {{ color: {ACCENT}; }}
.stButton > button {{ background: linear-gradient(135deg, #1a4a7a, #0d2d52); color: {ACCENT}; border: 1px solid #1e3a5a; border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)

def fmt(v, d=6):
    try:
        return f'{float(v):.{d}f}'
    except Exception:
        return str(v)

def sine_low(n=1000, f=0.05, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2*np.pi*f*t) + noise * rng.normal(size=n)

def sine_high(n=1000, f=0.05, noise=0.15, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2*np.pi*f*t) + noise * rng.normal(size=n)

def logistic_map(n=1000, r=3.9, x0=0.1):
    x = x0
    out = [x]
    for _ in range(n):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out[1:])

def white_noise(n=1000, seed=2):
    return np.random.default_rng(seed).normal(size=n)

def pink_noise(n=1000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return np.cumsum(x)

def lorenz_ts(n=1000, sigma=10, rho=28, beta=8/3, dt=0.01):
    x = y = z = 1.0
    xs = []
    for _ in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        xs.append(x)
    return np.array(xs)

DEMOS = {
    'Senoidal bajo': sine_low,
    'Senoidal alto': sine_high,
    'Mapa logístico caótico': logistic_map,
    'Ruido blanco': white_noise,
    'Ruido rosa': pink_noise,
    'Lorenz': lorenz_ts,
}

st.title('MODE Pipeline v2')
st.caption('Calibración fina sobre señales sintéticas, con backend corregido.')

with st.sidebar:
    st.header('Configuración')
    mode = st.radio('Modo', ['Señal sintética', 'Cargar CSV', 'ECG MIT-BIH', 'Ventanas temporales'])
    st.divider()
    m = st.slider('m embedding', 2, 6, 3, 1)
    max_tau = st.slider('max tau AMI', 10, 80, 50, 5)
    st.divider()
    st.subheader('Calibración')
    show_advanced = st.checkbox('Mostrar detalles avanzados', value=True)
    st.divider()
    run = st.button('Ejecutar pipeline', type='primary', use_container_width=True)

x = None
label = 'sin datos'
extra = {}

if mode == 'Señal sintética':
    demo = st.selectbox('Señal demo', list(DEMOS.keys()))
    n = st.slider('N muestras', 300, 5000, 1000, 100)
    x = DEMOS[demo](n)
    label = demo
elif mode == 'Cargar CSV':
    up = st.file_uploader('CSV de una columna', type=['csv', 'txt'])
    if up is not None:
        x = pd.read_csv(up, header=None).iloc[:, 0].dropna().values.astype(float)
        label = getattr(up, 'name', 'csv')
elif mode == 'ECG MIT-BIH':
    dat = st.file_uploader('Archivo .dat', type=['dat'])
    hea = st.file_uploader('Archivo .hea', type=['hea'])
    st.info('Sección lista para integrar lectura MIT-BIH en la siguiente iteración.')
elif mode == 'Ventanas temporales':
    wmode = st.radio('Fuente', ['Señal sintética', 'Cargar CSV'])
    if wmode == 'Señal sintética':
        demo = st.selectbox('Señal demo', list(DEMOS.keys()), key='w_demo')
        n = st.slider('N total', 1000, 5000, 2000, 100)
        x = DEMOS[demo](n)
        label = demo
    else:
        up = st.file_uploader('CSV', type=['csv', 'txt'], key='w_csv')
        if up is not None:
            x = pd.read_csv(up, header=None).iloc[:, 0].dropna().values.astype(float)
            label = getattr(up, 'name', 'csv')
    winsize = st.slider('Tamano ventana', 200, 2000, 1000, 100)
    winstep = st.slider('Paso ventana', 50, 1000, 500, 50)
    winmax = st.slider('Max ventanas', 5, 50, 20, 5)
    extra = {'winsize': winsize, 'winstep': winstep, 'winmax': winmax}

if run and x is not None:
    pipe = AttractorPipeline(m=m, max_tau=max_tau, verbose=False)
    if mode != 'Ventanas temporales':
        res = pipe.run(x, label=label)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('tau', res['tau'])
        c2.metric('tau inicial', res['tau_initial'])
        c3.metric('epsilon mediana', fmt(res['epsilon'], 8))
        c4.metric('R3 Score', fmt(res['R3']['R3_score'], 8))
        c5.metric('Regimen', res['regime_desc'])
        st.divider()
        a, b = st.columns([2, 1])
        with a:
            st.subheader('Resultados')
            df = pd.DataFrame([
                ['tau semidinamico', res['tau']],
                ['tau inicial', res['tau_initial']],
                ['epsilon mediana', fmt(res['epsilon'], 8)],
                ['Regimen', res['regime_desc']],
                ['R3 Score', fmt(res['R3']['R3_score'], 8)],
                ['Coherente', 'SI' if res['R3']['coherent'] else 'NO'],
                ['lambda (Lyapunov)', fmt(res['metrics']['lambda'], 8)],
                ['D2 (dim. corr.)', fmt(res['metrics']['D2'], 8)],
                ['C_LZ (compl.)', fmt(res['metrics']['LZ'], 8)],
                ['TE (trans. ent.)', fmt(res['metrics']['TE'], 8)],
                ['SampEn (muestra)', fmt(res['metrics']['SampEn'], 8)],
            ], columns=['Variable', 'Valor'])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button('Descargar resultados CSV', df.to_csv(index=False).encode(), 'resultados.csv', 'text/csv')
        with b:
            st.subheader('R3 interno')
            st.json({'regimen': res['regime'], 'delta': res['R3']['delta'], 'n_valid': res['R3']['n_valid']})
    else:
        winsize = extra['winsize']; winstep = extra['winstep']; winmax = extra['winmax']
        rows = []
        for i, start in enumerate(range(0, len(x) - winsize + 1, winstep)):
            if i >= winmax:
                break
            seg = x[start:start + winsize]
            if np.std(seg) < 1e-8:
                continue
            r = pipe.run(seg, label=f'{label}_w{i}')
            rows.append({
                'ventana': i,
                'inicio': start,
                'tau': r['tau'],
                'epsilon': r['epsilon'],
                'R3': r['R3']['R3_score'],
                'coherente': r['R3']['coherent'],
                'regimen': r['regime_desc'],
                'lambda': r['metrics']['lambda'],
            })
        dfw = pd.DataFrame(rows)
        st.dataframe(dfw, use_container_width=True, hide_index=True)
        st.download_button('Descargar ventanas CSV', dfw.to_csv(index=False).encode(), 'ventanas.csv', 'text/csv')
else:
    st.info('Elegí una señal y presioná Ejecutar pipeline.')
