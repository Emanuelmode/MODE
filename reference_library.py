"""
reference.py
═══════════════════════════════════════════════════════════════
Biblioteca centralizada de umbrales δ_{k,r} por régimen dinámico.
Calibración final basada en gradientes RMS observados en señales
fisiológicas y sintéticas validadas.
Propósito: fuente única de verdad para R3Descriptor y calibración.
Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""
import warnings
from typing import Dict

REF_VERSION = "2026.05d"
REF_SOURCE  = "Calibración empírica final (Duarte 2026)"

DELTA_LIBRARY: Dict[str, Dict[str, float]] = {
    # ── Estable / Periódico ─────────────────────────────────
    'stable': {
        'lambda': 0.08,
        'D2':     0.12,
        'LZ':     0.20,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────
    'weakly_chaotic': {
        'lambda': 0.08,
        'D2':     0.12,
        'LZ':     0.20,
        'TE':     0.80,  # TE más variable en transiciones
        'SampEn': 2.50,
    },

    # ── Caótico ─────────────────────────────────────────────
    'chaotic': {
        'lambda': 0.10,
        'D2':     0.12,
        'LZ':     0.22,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Hipercaótico / Estructurado ──────────────────────────
    'hyperchaotic': {
        'lambda': 0.15,
        'D2':     0.12,
        'LZ':     0.25,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────
    'noisy': {
        'lambda': 0.20,
        'D2':     0.15,
        'LZ':     0.30,
        'TE':     0.20,
        'SampEn': 2.50,
    }
}

def _validate():
    """Valida estructura y rangos al importar."""
    REQUIRED = {'lambda', 'D2', 'LZ', 'TE', 'SampEn'}
    for regime, deltas in DELTA_LIBRARY.items():
        missing = REQUIRED - set(deltas.keys())
        if missing:
            raise ValueError(f"DELTA_LIBRARY['{regime}'] falta claves: {missing}")
        for k, v in deltas.items():
            if v <= 0:
                warnings.warn(f"δ['{regime}']['{k}']={v} ≤ 0. Usar valores >0 para evitar división por cero.")
_validate()

# ── UTILIDADES DE ACCESO ─────────────────────────────────────
def get_delta(regime: str) -> dict:
    """Retorna dict completo de δ por métrica para el régimen."""
    return DELTA_LIBRARY.get(regime, DELTA_LIBRARY['weakly_chaotic'])

def get_scalar(regime: str) -> float:
    """Retorna media de δ (escalar) para compatibilidad con UI (ax.axvline)."""
    vals = list(get_delta(regime).values())
    return float(sum(vals) / len(vals))
