"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

REVISIÓN 2026-05f: 
- δ_SampEn = 2.50 para todos (corrige peso=0 en señales de orden)
- Basado en análisis exhaustivo de 9 señales de referencia

Emanuel Duarte — 2026
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {

    # ── Estable / Periódico ──────────────────────────────────────────
    # Logístico r=3.5, senoidales
    'stable': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 2.50,  # ⬆ Elevado para evitar peso=0
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Rössler
    'weakly_chaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 2.50,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # Lorenz, Logístico r=3.7
    'chaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 2.50,
    },

    # ── Hipercaótico / Estructurado ──────────────────────────────────
    # Logístico r=3.9
    'hyperchaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 2.50,
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────────────
    'noisy': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 2.50,
    },
}

# ── CONSTANTES ─────────────────────────────────────────────────
COHERENCE_THRESHOLD = 0.60
DELTA_VERSION = "2026.05f"
