"""
reference_library.py
═══════════════════════════════════════════════════════════════
Marcos referenciales por régimen dinámico.
Contiene: umbrales δ, rangos esperados por métrica, parámetros SampEn,
fuentes de calibración y validación automática al importar.
Propósito: eliminar números mágicos, garantizar reproducibilidad Q1/Q2,
           habilitar sweeps Monte Carlo sin tocar el core matemático.
═══════════════════════════════════════════════════════════════
"""
import warnings
from typing import Dict, Any, Tuple

REF_VERSION = "2026.05c"
REF_SOURCE  = "Literatura clásica + calibración empírica Duarte 2026 (Lorenz, Rössler, Logístico r=3.5/3.7/3.9)"

REGIME_REFERENCES: Dict[str, Dict[str, Any]] = {
    'stable': {
        'description': 'Órbitas periódicas o cuasi-degeneradas',
        'delta': {'lambda': 0.02, 'D2': 0.10, 'LZ': 0.02, 'TE': 0.03, 'SampEn': 0.65},
        'expected_ranges': {'lambda': (None, 0.00), 'D2': (0.5, 1.5), 'LZ': (0.00, 0.30), 'TE': (0.00, 0.10), 'SampEn': (0.00, 0.30)},
        'literature': 'Grassberger & Procaccia (1983); Richman & Moorman (2000)'
    },
    'weakly_chaotic': {
        'description': 'Transición orden-caos, estructura espiral débil',
        'delta': {'lambda': 0.08, 'D2': 0.05, 'LZ': 0.07, 'TE': 0.12, 'SampEn': 0.35},
        'expected_ranges': {'lambda': (0.00, 0.20), 'D2': (1.2, 2.0), 'LZ': (0.30, 0.70), 'TE': (0.05, 0.20), 'SampEn': (0.30, 0.80)},
        'literature': 'Rössler (1976); Duarte (2026) calibración empírica'
    },
    'chaotic': {
        'description': 'Caos determinista estable, divergencia controlada',
        'delta': {'lambda': 0.19, 'D2': 0.02, 'LZ': 0.18, 'TE': 0.07, 'SampEn': 0.02},
        'expected_ranges': {'lambda': (0.20, 0.80), 'D2': (1.8, 2.5), 'LZ': (0.60, 0.95), 'TE': (0.03, 0.15), 'SampEn': (0.40, 1.00)},
        'literature': 'Lorenz (1963); Logístico r=3.7; Rosenstein et al. (1993)'
    },
    'hyperchaotic': {
        'description': 'Múltiples exponentes positivos, complejidad multi-escala',
        'delta': {'lambda': 0.13, 'D2': 0.02, 'LZ': 0.02, 'TE': 0.06, 'SampEn': 0.17},
        'expected_ranges': {'lambda': (0.50, 1.50), 'D2': (2.5, 4.0), 'LZ': (0.85, 1.20), 'TE': (0.02, 0.10), 'SampEn': (1.00, 2.00)},
        'literature': 'Rössler hyperchaotic; Logístico r=3.9; Mantegna & Stanley (1999)'
    },
    'noisy': {
        'description': 'Incoherencia estructural, sin atractor detectable',
        'delta': {'lambda': 0.20, 'D2': 0.05, 'LZ': 0.07, 'TE': 0.12, 'SampEn': 0.35},
        'expected_ranges': {'lambda': (None, None), 'D2': (3.0, None), 'LZ': (0.90, 1.50), 'TE': (0.00, 0.05), 'SampEn': (1.50, None)},
        'literature': 'Ruido blanco/rosa; Peng et al. (1995) HRV patológico'
    }
}

def _validate():
    METRICS = {'lambda', 'D2', 'LZ', 'TE', 'SampEn'}
    for regime, ref in REGIME_REFERENCES.items():
        if set(ref['delta'].keys()) != METRICS:
            raise ValueError(f"delta en '{regime}' no coincide con métricas base")
        if set(ref['expected_ranges'].keys()) != METRICS:
            raise ValueError(f"expected_ranges en '{regime}' incompleto")
_validate()

# ── UTILIDADES ─────────────────────────────────────────────────────
def get_thresholds(regime: str) -> dict:
    return REGIME_REFERENCES.get(regime, REGIME_REFERENCES['weakly_chaotic'])['delta']

def check_in_range(metric: str, value: float, regime: str) -> Tuple[bool, str]:
    ref = REGIME_REFERENCES.get(regime, REGIME_REFERENCES['weakly_chaotic'])
    rng = ref['expected_ranges'].get(metric, (None, None))
    low, high = rng
    if low is not None and value < low: return False, f"< {low}"
    if high is not None and value > high: return False, f"> {high}"
    return True, "OK"
