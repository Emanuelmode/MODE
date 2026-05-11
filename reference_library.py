"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

REVISIÓN 2026-05d: Deltas calibrados para R3 ∈ [0.7, 0.95]
en señales coherentes. Los valores anteriores eran 10-50x
menores que los gradientes observados, forzando R3 ≈ 0.2-0.5.

Calibración: percentil 85 de gradientes / (1 - 0.80)
donde 0.80 es el peso objetivo.

Señales de referencia:
  stable:        Lorenz (dt=0.01) + Logístico r=3.5
  chaotic:       Logístico r=3.7
  hyperchaotic:  Logístico r=3.9
  weakly_chaotic: Rössler
  noisy:         Ruido blanco

Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {

    # ── Estable / Periódico ──────────────────────────────────────────
    # Gradientes observados: λ≈0.05-0.09, D2≈0.08, LZ≈0.05, TE≈0.03, SampEn≈0.65
    # target_weight=0.80 → delta = grad_p85 / 0.20
    'stable': {
        'lambda': 1.50,
        'D2':     0.80,
        'LZ':     1.20,
        'TE':     0.80,
        'SampEn': 1.50,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    'weakly_chaotic': {
        'lambda': 1.20,
        'D2':     0.60,
        'LZ':     1.00,
        'TE':     0.90,
        'SampEn': 1.00,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    'chaotic': {
        'lambda': 2.50,
        'D2':     1.00,
        'LZ':     1.50,
        'TE':     1.00,
        'SampEn': 1.20,
    },

    # ── Hipercaótico / Estructurado ──────────────────────────────────
    'hyperchaotic': {
        'lambda': 2.00,
        'D2':     1.00,
        'LZ':     1.50,
        'TE':     1.00,
        'SampEn': 1.20,
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────────────
    'noisy': {
        'lambda': 1.80,
        'D2':     0.80,
        'LZ':     1.20,
        'TE':     1.00,
        'SampEn': 1.20,
    },
}

# ── CONSTANTES GLOBALES ──────────────────────────────────────
COHERENCE_THRESHOLD = 0.60
TARGET_WEIGHT = 0.80

DELTA_VERSION = "2026.05d"
