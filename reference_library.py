"""
reference_library.py
═══════════════════════════════════════════════════════════════
Marcos referenciales por régimen dinámico.

REVISIÓN 2026-05f: δ_SampEn=2.50 para todos los regímenes.
Rangos actualizados según análisis de 9 señales de referencia.

Emanuel Duarte — 2026
═══════════════════════════════════════════════════════════════
"""
import warnings
from typing import Dict, Any, Tuple

REF_VERSION = "2026.05f"
REF_SOURCE = "Literatura clásica + calibración empírica Duarte 2026"
COHERENCE_THRESHOLD = 0.60

REGIME_REFERENCES: Dict[str, Dict[str, Any]] = {
    'stable': {
        'description': 'Órbitas periódicas o cuasi-degeneradas',
        'delta': {
            'lambda': 0.05, 'D2': 0.03, 'LZ': 0.14, 'TE': 0.08, 'SampEn': 2.50,
        },
        'expected_ranges': {
            'lambda': (None, 0.01),
            'D2': (0.5, 1.5),
            'LZ': (0.00, 0.30),
            'TE': (0.00, 0.10),
            'SampEn': (0.00, 0.30),
        },
        'expected_r3_range': (0.60, 0.95),
        'coherent': True,
        'literature': 'Grassberger & Procaccia (1983); Richman & Moorman (2000)',
    },

    'weakly_chaotic': {
        'description': 'Transición orden-caos, Rössler',
        'delta': {
            'lambda': 0.05, 'D2': 0.03, 'LZ': 0.14, 'TE': 0.08, 'SampEn': 2.50,
        },
        'expected_ranges': {
            'lambda': (0.01, 0.20),
            'D2': (1.2, 2.0),
            'LZ': (0.30, 0.70),
            'TE': (0.05, 0.20),
            'SampEn': (0.20, 0.80),
        },
        'expected_r3_range': (0.60, 0.95),
        'coherent': True,
        'literature': 'Rössler (1976); Duarte (2026)',
    },

    'chaotic': {
        'description': 'Caos determinista estable',
        'delta': {
            'lambda': 0.05, 'D2': 0.03, 'LZ': 0.14, 'TE': 0.08, 'SampEn': 2.50,
        },
        'expected_ranges': {
            'lambda': (0.15, 0.50),
            'D2': (1.8, 2.5),
            'LZ': (0.50, 0.90),
            'TE': (0.03, 0.15),
            'SampEn': (0.30, 1.00),
        },
        'expected_r3_range': (0.60, 0.95),
        'coherent': True,
        'literature': 'Lorenz (1963); Logístico r=3.7',
    },

    'hyperchaotic': {
        'description': 'Múltiples exponentes positivos',
        'delta': {
            'lambda': 0.05, 'D2': 0.03, 'LZ': 0.14, 'TE': 0.08, 'SampEn': 2.50,
        },
        'expected_ranges': {
            'lambda': (0.40, 1.20),
            'D2': (2.5, 4.0),
            'LZ': (0.85, 1.20),
            'TE': (0.02, 0.10),
            'SampEn': (0.80, 1.50),
        },
        'expected_r3_range': (0.60, 0.95),
        'coherent': True,
        'literature': 'Logístico r=3.9; Mantegna & Stanley (1999)',
    },

    'noisy': {
        'description': 'Incoherencia estructural, ruido',
        'delta': {
            'lambda': 0.05, 'D2': 0.03, 'LZ': 0.14, 'TE': 0.08, 'SampEn': 2.50,
        },
        'expected_ranges': {
            'lambda': (None, None),
            'D2': (3.0, None),
            'LZ': (0.90, 1.50),
            'TE': (0.00, 0.05),
            'SampEn': (1.50, None),
        },
        'expected_r3_range': (0.20, 0.60),  # NO debe ser coherente
        'coherent': False,
        'literature': 'Peng et al. (1995)',
    }
}


def _validate():
    METRICS = {'lambda', 'D2', 'LZ', 'TE', 'SampEn'}
    for regime, ref in REGIME_REFERENCES.items():
        if set(ref['delta'].keys()) != METRICS:
            raise ValueError(f"delta en '{regime}' incompleto")
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


def is_coherent_regime(regime: str) -> bool:
    return REGIME_REFERENCES.get(regime, {}).get('coherent', False)


def get_expected_r3(regime: str) -> Tuple[float, float]:
    return REGIME_REFERENCES.get(regime, REGIME_REFERENCES['weakly_chaotic']).get('expected_r3_range', (0.60, 0.95))
