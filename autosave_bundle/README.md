# autosave_bundle

Módulo para guardar automáticamente resultados etiquetados del pipeline.

## Archivos
- `run_export.py`: exportador de corridas.
- `README.md`: esta guía mínima.

## Uso
```python
from autosave_bundle.run_export import RunExporter, RunLabel
```

Crear un `RunExporter`, armar un `RunLabel` y llamar a `save_bundle(...)` al final de cada corrida.

## Estructura sugerida
```text
main/
  app.py
  pipeline.py
  r3_delta_library.py
  autosave_bundle/
    run_export.py
    README.md
```
