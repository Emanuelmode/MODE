"""
result_saver.py
═══════════════════════════════════════════════════════════════
Módulo de persistencia de resultados para MODE Attractor Pipeline.
Guarda automáticamente cada análisis etiquetado según el archivo
de origen, con marca de agua de authorship.

Uso:
    from result_saver import save_result, get_history, load_results_csv

    # Después de pipe.run()
    result = pipe.run(signal, label=filename)
    save_result(result, label=filename)

Autor: Investigador/dueño intelectual Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ── Configuración global ─────────────────────────────────────────
AUTHOR = "Investigador/dueño intelectual Emanuel Duarte"
RESULTS_DIR = "resultados_autosave"
HISTORY_JSONL = "history.jsonl"
INDEX_CSV = "index.csv"


# ── Utilidades internas ──────────────────────────────────────────

def _ensure_dir(path: str = RESULTS_DIR) -> Path:
    """Crea directorio de resultados si no existe."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_filename(name: str) -> str:
    """Limpia nombre para uso como nombre de archivo."""
    # Elimina caracteresproblemáticos para filesystems
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
        name = name.replace(c, '_')
    # Elimina extensión si existe (se la volveremos a agregar)
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    return name[:180]  # Limita largo para seguridad


def _timestamp() -> str:
    """Timestamp ISO para metadata."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_filename() -> str:
    """Timestamp para nombre de archivo único."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_native(obj):
    """Convierte objetos numpy/Python a tipos JSON serializables."""
    if hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (float, int)):
        # Maneja NaN e inf
        import math
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj
    elif isinstance(obj, bool):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return str(obj)


# ── Función principal ────────────────────────────────────────────

def save_result(
    result: dict,
    label: str = "unknown",
    results_dir: str = RESULTS_DIR,
    include_watermark: bool = True,
) -> dict:
    """
    Guarda resultado de análisis con etiquetado automático.

    Genera:
        {label}.csv        — métricas principales en formato plano
        {label}_detalle.json — resultado completo (sin wavefors)
        history.jsonl       — índice append con todos los análisis
        index.csv          — tabla resumen de todos los análisis

    Args:
        result: Dict retornado por AttractorPipeline.run()
        label: Nombre/etiqueta del archivo analizado (ej: "chb01_01.edf")
        results_dir: Directorio base para resultados
        include_watermark: Si True, incluye AUTHOR en metadata

    Returns:
        Dict con paths de archivos guardados y metadata del análisis
    """
    _ensure_dir(results_dir)
    ts = _timestamp()
    ts_file = _timestamp_filename()
    safe_label = _sanitize_filename(label)

    saved = {}

    # ── Extraer datos del result ────────────────────────────────
    r3 = result.get("R3", {})
    metrics = result.get("metrics", {})

    # ── 1. CSV principal con métricas planas ──────────────────
    csv_path = Path(results_dir) / f"{safe_label}.csv"

    # Si ya existe, append; si no, crear con header
    file_exists = csv_path.exists()

    # Convertir a tipos nativos para el CSV
    def fmt_val(v):
        if v is None:
            return ""
        try:
            if hasattr(v, 'item'):
                v = v.item()
            if isinstance(v, float):
                import math
                if math.isnan(v) or math.isinf(v):
                    return ""
            return v
        except:
            return str(v)

    row = {
        "timestamp": ts,
        "archivo_origen": label,
        "n_puntos": fmt_val(len(result.get("x_normalized", []))),
        "tau": fmt_val(result.get("tau")),
        "tau_inicial": fmt_val(result.get("tau_initial")),
        "epsilon": fmt_val(result.get("epsilon")),
        "regimen": fmt_val(r3.get("regime")),
        "regimen_desc": fmt_val(r3.get("regime_desc")),
        "R3_score": fmt_val(r3.get("R3_score")),
        "coherente": "TRUE" if r3.get("coherent") else "FALSE",
        "delta": fmt_val(r3.get("delta")),
        "n_validas": fmt_val(r3.get("n_valid")),
        "lambda_lyapunov": fmt_val(metrics.get("lambda")),
        "D2_corr_dim": fmt_val(metrics.get("D2")),
        "LZ_complejidad": fmt_val(metrics.get("LZ")),
        "TE_transferencia": fmt_val(metrics.get("TE")),
        "SampEn": fmt_val(metrics.get("SampEn")),
        "autor": AUTHOR if include_watermark else "",
    }

    with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    saved["csv_principal"] = str(csv_path)

    # ── 2. JSON detalle (sin datos pesados como epsilon_series) ─
    detail = {
        "timestamp": ts,
        "archivo_origen": label,
        "autor": AUTHOR if include_watermark else "",
        "pipeline": {
            "tau": _to_native(result.get("tau")),
            "tau_initial": _to_native(result.get("tau_initial")),
            "epsilon": _to_native(result.get("epsilon")),
        },
        "regimen": {
            "tipo": _to_native(r3.get("regime")),
            "descripcion": _to_native(r3.get("regime_desc")),
            "delta": _to_native(r3.get("delta")),
        },
        "r3": {
            "score": _to_native(r3.get("R3_score")),
            "coherente": bool(r3.get("coherent")) if r3.get("coherent") is not None else None,
            "n_validas": _to_native(r3.get("n_valid")),
        },
        "metrics": _to_native(metrics),
        "gradients": _to_native(r3.get("gradients", {})),
        "stability_map": _to_native(r3.get("stability_map", {})),
    }

    detail_path = Path(results_dir) / f"{safe_label}_{ts_file}_detalle.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    saved["json_detalle"] = str(detail_path)

    # ── 3. History JSONL (append global) ──────────────────────
    history_path = Path(results_dir) / HISTORY_JSONL
    history_entry = {
        "timestamp": ts,
        "archivo_origen": label,
        "regimen": _to_native(r3.get("regime")),
        "R3_score": _to_native(r3.get("R3_score")),
        "coherente": bool(r3.get("coherent")) if r3.get("coherent") is not None else None,
        "autor": AUTHOR if include_watermark else "",
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
    saved["history_jsonl"] = str(history_path)

    # ── 4. Index CSV actualizado ───────────────────────────────
    _update_index(results_dir)

    return {
        "timestamp": ts,
        "archivo_origen": label,
        "autor": AUTHOR if include_watermark else "",
        "saved_files": saved,
        "regimen": _to_native(r3.get("regime")),
        "R3_score": _to_native(r3.get("R3_score")),
        "coherente": bool(r3.get("coherent")) if r3.get("coherent") is not None else None,
    }


def _update_index(results_dir: str = RESULTS_DIR) -> None:
    """Actualiza el index.csv con todos los análisis."""
    index_path = Path(results_dir) / INDEX_CSV

    # Leer history.jsonl
    history_path = Path(results_dir) / HISTORY_JSONL
    if not history_path.exists():
        return

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Escribir index
    fieldnames = ["timestamp", "archivo_origen", "regimen", "R3_score", "coherente", "autor"]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {k: entry.get(k, "") for k in fieldnames}
            writer.writerow(row)


# ── Funciones de consulta ───────────────────────────────────────

def get_history(
    results_dir: str = RESULTS_DIR,
    limit: Optional[int] = None,
) -> List[dict]:
    """
    Carga historial de análisis desde history.jsonl.

    Args:
        results_dir: Directorio de resultados
        limit: Si se especifica, retorna solo los últimos N registros

    Returns:
        Lista de dicts con entries del historial
    """
    history_path = Path(results_dir) / HISTORY_JSONL
    if not history_path.exists():
        return []

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if limit:
        entries = entries[-limit:]

    return entries


def load_results_csv(
    label: str,
    results_dir: str = RESULTS_DIR,
) -> Optional[Path]:
    """
    Retorna el path al CSV de un análisis específico.

    Args:
        label: Nombre del archivo original (sin extensión)
        results_dir: Directorio de resultados

    Returns:
        Path al CSV o None si no existe
    """
    safe_label = _sanitize_filename(label)
    csv_path = Path(results_dir) / f"{safe_label}.csv"
    return csv_path if csv_path.exists() else None


def get_summary_stats(results_dir: str = RESULTS_DIR) -> dict:
    """
    Genera estadísticas resumidas del historial.

    Returns:
        Dict con:
            - total_analisis: número total de análisis
            - regimens: conteo por tipo de régimen
            - coherente_count/pct: análisis coherentes
            - R3_mean/min/max: estadísticas del score
            - archivos: lista de últimos 20 archivos
    """
    history = get_history(results_dir)
    if not history:
        return {
            "total_analisis": 0,
            "regimens": {},
            "coherente_count": 0,
            "coherente_pct": 0.0,
            "R3_mean": None,
            "R3_min": None,
            "R3_max": None,
            "archivos": [],
        }

    regimens = {}
    r3_scores = []
    coherent_count = 0

    for entry in history:
        r = entry.get("regimen", "unknown")
        regimens[r] = regimens.get(r, 0) + 1
        if entry.get("R3_score") is not None:
            r3_scores.append(entry["R3_score"])
        if entry.get("coherente"):
            coherent_count += 1

    return {
        "total_analisis": len(history),
        "regimens": regimens,
        "coherente_count": coherent_count,
        "coherente_pct": coherent_count / len(history) * 100 if history else 0.0,
        "R3_mean": sum(r3_scores) / len(r3_scores) if r3_scores else None,
        "R3_min": min(r3_scores) if r3_scores else None,
        "R3_max": max(r3_scores) if r3_scores else None,
        "archivos": [e.get("archivo_origen", "unknown") for e in history[-20:]],
    }


# ── Integración Streamlit ───────────────────────────────────────

def integration_snippet():
    """
    Retorna el snippet de código para integrar en app.py.

    Usage:
        En app.py, después de pipe.run(), agregar:

        from result_saver import save_result
        save_result(result, label=label)
    """
    return '''
# ── Agregar después de pipe.run() ───────────────────────────────
from result_saver import save_result

# En la sección de ejecución, después de:
#     r = pipe.run(xdata, label=label)
# Agregar:
save_result(r, label=label, results_dir="resultados_autosave")
    '''


# ── Demo/test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"result_saver.py — {AUTHOR}")
    print(f"Directorio de resultados: {RESULTS_DIR}")
    print()
    print("Funciones disponibles:")
    print("  save_result(result, label, results_dir)  — Guarda análisis")
    print("  get_history(limit)                       — Lee historial")
    print("  load_results_csv(label)                  — Busca CSV por nombre")
    print("  get_summary_stats()                      — Estadísticas resumidas")
    print()
    print("Para integrar en app.py:")
    print(integration_snippet())
