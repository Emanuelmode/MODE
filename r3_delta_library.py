"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

REVISIÓN 2026-05g:
- Deltas calibrados para señales sintéticas Y fisiológicas
- TE=0.80 en weakly_chaotic para acomodar ECG fisiológico
- SampEn=2.50 para todos (tau-templates escala distinta)
- D2=0.12 mínimo para evitar destrucción por gradientes pequeños

Emanuel Duarte — 2026
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {

    # ── Estable / Periódico ──────────────────────────────────────────
    # Logístico r=3.5, senoidales, ECG ritmo sinusal puro
    'stable': {
        'lambda': 0.08,
        'D2':     0.12,
        'LZ':     0.20,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Rössler, ECG fisiológico (TE puede ser alto en arritmia)
    'weakly_chaotic': {
        'lambda': 0.08,
        'D2':     0.12,
        'LZ':     0.20,
        'TE':     0.80,   # permisivo para ECG donde TE puede ser 0.76+
        'SampEn': 2.50,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # Lorenz, Logístico r=3.7
    'chaotic': {
        'lambda': 0.10,
        'D2':     0.12,
        'LZ':     0.22,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Hipercaótico / Estructurado ──────────────────────────────────
    # Logístico r=3.9
    'hyperchaotic': {
        'lambda': 0.15,
        'D2':     0.12,
        'LZ':     0.25,
        'TE':     0.15,
        'SampEn': 2.50,
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────────────
    'noisy': {
        'lambda': 0.20,
        'D2':     0.15,
        'LZ':     0.30,
        'TE':     0.20,
        'SampEn': 2.50,
    },
}

# ── CONSTANTES ─────────────────────────────────────────────────
COHERENCE_THRESHOLD = 0.60
DELTA_VERSION = "2026.05g"
