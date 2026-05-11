"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

REVISIÓN 2026-05e: Calibración baseda en análisis exhaustivo
de gradientes para señales de referencia.
- LZ: delta=0.14 (antes 0.02) para acomodar Lorenz (grad=0.11)
- TE: delta=0.08 (antes 0.03-0.12) como compromiso
- SampEn: delta=0.25 genérico, 2.50 para r=3.5

Señales de referencia: Lorenz, Rössler, Logístico r=3.5/3.7/3.9, Ruido

Emanuel Duarte — 2026
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {

    # ── Estable / Periódico ──────────────────────────────────────────
    # Logístico r=3.5: grad_LZ=0.003, grad_TE=0.02, grad_SampEn=0.85
    'stable': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,      # Para señales periódicas
        'TE': 0.08,
        'SampEn': 2.50,  # Alto para acomodar r=3.5 (caso extremo)
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Rössler: grad_LZ=?, grad_TE=?, grad_SampEn=?
    'weakly_chaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 0.25,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # Lorenz: grad_LZ=0.11, grad_TE=0.02, grad_SampEn=0.03
    'chaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,      # ⬆ Subido de 0.07 a 0.14
        'TE': 0.08,
        'SampEn': 0.25,
    },

    # ── Hipercaótico / Estructurado ──────────────────────────────────
    # Logístico r=3.9: grad_LZ=0.05, grad_TE=0.05, grad_SampEn=0.02
    'hyperchaotic': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 0.25,
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────────────
    # Ruido blanco: grad_LZ=0.10, grad_TE=0.28, grad_SampEn=0.01
    'noisy': {
        'lambda': 0.05,
        'D2': 0.03,
        'LZ': 0.14,
        'TE': 0.08,
        'SampEn': 0.25,  # Podría ser menor, pero TE ya filtra ruido
    },
}

# ── CONSTANTES ─────────────────────────────────────────────────
COHERENCE_THRESHOLD = 0.60
DELTA_VERSION = "2026.05e"
