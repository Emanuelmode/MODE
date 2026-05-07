import streamlit as st
import pandas as pd
from autosave_bundle.run_export import RunExporter, RunLabel
from pipeline import AttractorPipeline

st.set_page_config(page_title='Takens Adaptativo', layout='wide')
st.title('Takens Adaptativo')
st.caption('Framework de legibilidad observacional para sistemas dinamicos no lineales')

if 'pipeline_ready' not in st.session_state:
    st.session_state.pipeline_ready = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header('Configuracion')
    signal_type = st.selectbox('Tipo de señal', ['senoidal', 'lorenz', 'rossler', 'sintetica', 'custom'])
    user_label = st.text_input('Etiqueta', value='run_01')
    noise_level = st.selectbox('Ruido', ['bajo', 'medio', 'alto', 'sin_ruido'])
    sample_count = st.number_input('N', min_value=10, value=1000, step=1)
    tag = st.text_input('Tag', value='')
    run_btn = st.button('Ejecutar pipeline', type='primary')

st.write('Seleccionar un modo en el sidebar y presionar Ejecutar pipeline.')

exporter = RunExporter(base_dir='outputs')

if run_btn:
    try:
        st.session_state.pipeline_ready = False
        signal = st.session_state.get('signal_data', None)
        if signal is None:
            st.error('No hay señal cargada.')
            st.stop()

        if isinstance(signal, pd.DataFrame):
            signal_values = signal.iloc[:, 0].to_numpy()
        else:
            signal_values = signal

        pipe = AttractorPipeline()
        resultado = pipe.run(signal_values)

        baselines = resultado.get('baselines', {}) if isinstance(resultado, dict) else {}
        embedding_df = None
        if isinstance(resultado, dict) and 'embedding_df' in resultado:
            embedding_df = resultado['embedding_df']

        label = RunLabel(
            signal_type=signal_type,
            user_label=user_label,
            noise_level=noise_level,
            sample_count=int(sample_count),
            source='sintetica',
            tag=tag,
        )

        saved = exporter.save_bundle(
            label=label,
            resultado=resultado.get('resultados', resultado) if isinstance(resultado, dict) else {'resultado': str(resultado)},
            baselines=baselines,
            embedding_df=embedding_df,
            extra_meta={'signal_type': signal_type, 'noise_level': noise_level, 'sample_count': int(sample_count)}
        )

        st.success('Pipeline ejecutado y guardado correctamente.')
        st.json(saved)
        st.session_state.last_result = resultado
        st.session_state.pipeline_ready = True

    except Exception as e:
        st.error(f'Error ejecutando pipeline: {e}')
        st.exception(e)

if st.session_state.last_result is not None:
    st.subheader('Ultimo resultado')
    st.write(st.session_state.last_result)
