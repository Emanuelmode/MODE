import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from autosave_bundle.run_export import RunExporter, RunLabel
from pipeline import AttractorPipeline

st.set_page_config(page_title='Takens Adaptativo', layout='wide')

AUTHOR = 'Emanuel Duarte'
VERSION = '2026-05'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if 'signal_data' not in st.session_state:
    st.session_state.signal_data = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_saved' not in st.session_state:
    st.session_state.last_saved = None
if 'demo_signal_name' not in st.session_state:
    st.session_state.demo_signal_name = 'senoidal'


def _load_demo_signals(n=1000):
    t = np.linspace(0, 100, n)
    rng = np.random.default_rng(42)

    def _logistic_map(N=1000, r=3.9):
        x = 0.1
        out = [x]
        for _ in range(N - 1):
            x = r * x * (1 - x)
            out.append(x)
        return np.array(out)

    def _lorenz_ts(N=1000, sigma=10, rho=28, beta=8/3, dt=0.01):
        x, y, z = 1.0, 1.0, 1.0
        xs = []
        for _ in range(N):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            xs.append(x)
        return np.array(xs)

    return {
        'senoidal': np.sin(2 * np.pi * 0.1 * t) + 0.05 * rng.normal(size=n),
        'lorenz': _lorenz_ts(n),
        'logistica': _logistic_map(n, r=3.9),
        'ruido_blanco': rng.normal(size=n),
        'cuasiperiodica': np.sin(2 * np.pi * 0.07 * t) + 0.35 * np.sin(2 * np.pi * 0.17 * t),
    }


def _to_array(signal_obj):
    if signal_obj is None:
        return None
    if isinstance(signal_obj, pd.DataFrame):
        return signal_obj.iloc[:, 0].to_numpy()
    if isinstance(signal_obj, pd.Series):
        return signal_obj.to_numpy()
    return np.asarray(signal_obj)


def _e
