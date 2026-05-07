import json
from pathlib import Path

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


def _to_array(signal_obj):
    if signal_obj is None:
        return None
    if isinstance(signal_obj, pd.DataFrame):
        return signal_obj.iloc[:, 0].to_numpy()
    if isinstance(signal_obj, pd.Series):
        return signal_obj.to_numpy()
    return signal_obj


def _extract_result_dict(result):
    if not isinstance(result, dict):
        return {'resultado': str(result)}
    if 'resultados' in result and isinstance(result['resultados'], dict):
        return result['resultados']
    if 'metrics' in result and isinstance(result['metrics'], dict):
        metrics = result['metrics']
        out = {
            'tau semidinamico': result.get('tau', None),
            'tau inicial': result.get('tau_initial', None),
            'epsilon mediana': result.get('epsilon', None),
            'Regimen': result.get('regime_desc', result.get('regime', None)),
            'R3 Score': result.get('R3', {}).get('R3_score', None) if isinstance(result.get('R3'), dict) else None,
            'Coherente': result.get('R3', {}).get('coherent', None) if isinstance(result.get('R3'), dict) else None,
        }
        out.update(metrics)
        return out
    return result


def _extract_baselines(result):
    if not isinstance(result, dict):
        return {}
    baselines = result.get('baselines', None)
    if isinstance(baselines, dict):
        return baselines
    metrics = result.get('metrics', {}) if isinstance(result.get('metrics', {}), dict) else {}
    r3 = result.get('R3', {}) if isinstance(result.get('R3', {}), dict) else {}
    return {
        'lambda': metrics.get('lambda'),
        'D2': metrics.get('D2'),
        'LZ': metrics.get('LZ'),
        'TE': metrics.get('TE'),
        'SampEn': metrics.get('SampEn'),
        'R3_score': r3.get('R3_score'),
        'coherent': r3.get('coherent'),
        'regime': r3.get('regime'),
        'delta': r3.get('delta'),
    }


def _extract_embedding(result):
    if not isinstance(result, dict):
        return None
    emb = result.get('embedding', None)
    if emb is None:
        emb = result.get('embedding_df', None)
    if emb is None:
        return None
    if isinstance(emb, pd.DataFrame):
        return emb
    try:
        return pd.DataFrame(emb)
    except Exception:
        return None


def _result_label(result):
    if not isinstance(result, dict):
        return 'sin_resultado'
    return result.get('label', 'sin_label')


def _show_sidebar_status():
    with st.sidebar:
        st.markdown('### Estado')
        if st.session_state.signal_data is None:
            st.warning('No hay señal cargada.')
        else:
            st.success('Señal cargada.')
        if st.session_state.last_saved:
            st.info(f"Último guardado: {st.session_state.last_saved}")


def main():
    st.title('Takens Adaptativo')
    st.caption('Framework de legibilidad observacional para sistemas dinámicos no lineales.')

    _show_sidebar_status()

    with st.sidebar:
        st.header('Configuración')
        signal_type = st.selectbox('Tipo de señal', ['senoidal', 'lorenz', 'rossler', 'sintetica', 'custom'])
        user_label = st.text_input('Etiqueta', value='run_01')
        noise_level = st.selectbox('Ruido', ['sin_ruido', 'bajo', 'medio', 'alto'])
        sample_count = st.number_input('Cantidad de muestras', min_value=10, value=1000, step=1)
        tag = st.text_input('Tag', value='')
        max_tau = st.number_input('Tau maximo AMI', min_value=2, value=50, step=1)
        m = st.number_input('Dimension embedding m', min_value=2, value=3, step=1)
        run_btn = st.button('Ejecutar pipeline', type='primary')

    st.write('Seleccionar un modo en el sidebar y presionar Ejecutar pipeline.')

    if run_btn:
        try:
            signal = _to_array(st.session_state.signal_data)
            if signal is None:
                st.error('No hay señal cargada en session_state.signal_data.')
                st.stop()

            pipe = AttractorPipeline(m=int(m), max_tau=int(max_tau), verbose=False)
            result = pipe.run(signal, label=f'{signal_type}_{user_label}')

            result_dict = _extract_result_dict(result)
            baselines = _extract_baselines(result)
            embedding_df = _extract_embedding(result)
            label_value = _result_label(result)

            exporter = RunExporter(base_dir=str(OUTPUT_DIR))
            run_label = RunLabel(
                signal_type=signal_type,
                user_label=user_label,
                noise_level=noise_level,
                sample_count=int(sample_count),
                source='sintetica',
                tag=tag,
            )

            saved = exporter.save_bundle(
                label=run_label,
                resultado=result_dict,
                baselines=baselines,
                embedding_df=embedding_df,
                extra_meta={
                    'label': label_value,
                    'signal_type': signal_type,
                    'noise_level': noise_level,
                    'sample_count': int(sample_count),
                    'max_tau': int(max_tau),
                    'm': int(m),
                    'app_author': AUTHOR,
                    'app_version': VERSION,
                },
            )

            st.session_state.last_result = result
            st.session_state.last_saved = saved.get('meta_json', '')

            st.success('Pipeline ejecutado y guardado correctamente.')
            st.subheader('Archivos guardados')
            st.json(saved)
            st.subheader('Resultado')
            st.json(result_dict)

        except Exception as e:
            st.error(f'Error ejecutando pipeline: {e}')
            st.exception(e)

    if st.session_state.last_result is not None:
        st.subheader('Último resultado en sesión')
        try:
            st.json(st.session_state.last_result)
        except Exception:
            st.write(st.session_state.last_result)


if __name__ == '__main__':
    main()
