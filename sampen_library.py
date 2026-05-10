"""
sampen_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de calibración y cálculo de Sample Entropy (SampEn)
por régimen dinámico.
Fuentes primarias:
  Richman & Moorman (2000)    — definición original de SampEn
  Lake et al. (2002)          — sensibilidad a r en señales fisiológicas
  Costa et al. (2005)         — robustez numérica y multiescala
  Duarte (2026)               — calibración empírica R3 + gradientes RMS
Propósito:
  • Eliminar umbrales binarios → pesos gaussianos suaves
  • Garantizar estabilidad numérica (evitar log(0), std≈0, colapso de r)
  • Documentar calibración para reproducibilidad Q1/Q2
  • Exponer interfaz limpia para pipeline.py
═══════════════════════════════════════════════════════════════
"""
import warnings          # ← IMPORTANTE: requerido por _validate()
import numpy as np
from typing import Dict

# ── METADATA & VERSIONADO ─────────────────────────────────────
SAMPEN_VERSION = "2026.05a"
SAMPEN_SOURCE  = "p80_gradient_RMS + compatibilidad_R3_empirica"

# ── CONFIGURACIÓN POR RÉGIMEN ─────────────────────────────────
SAMPEN_CONFIG: Dict[str, Dict[str, float]] = {
    # Estable / Periódico
    'stable': {
        'm': 2, 'r_ratio': 0.12,
        'mu': 0.05, 'sigma': 0.03,
        'rationale': 'Órbitas periódicas/casi-degeneradas. SampEn ≈ 0. r bajo evita sobre-suavizado.'
    },

    # Caos débil / Cuasiperiódico
    'weakly_chaotic': {
        'm': 2, 'r_ratio': 0.20,
        'mu': 0.57, 'sigma': 0.12,
        'rationale': 'Transición orden-caos. Alta sensibilidad a τ. Mu calibrado contra Rössler/Logístico r=3.6.'
    },

    # Caótico determinista
    'chaotic': {
        'm': 2, 'r_ratio': 0.25,
        'mu': 0.53, 'sigma': 0.10,
        'rationale': 'Caos determinista estable (Lorenz, Logístico r=3.7). Divergencia controlada, gradientes suaves.'
    },

    # Hipercaótico / Estructurado
    'hyperchaotic': {
        'm': 2, 'r_ratio': 0.30,
        'mu': 1.51, 'sigma': 0.30,
        'rationale': 'Múltiples exponentes positivos. Complejidad multi-escala. r alto acomoda estructura fractal densa.'
    },

    # Ruido / Sin estructura
    'noisy': {
        'm': 2, 'r_ratio': 0.35,
        'mu': 1.65, 'sigma': 0.35,
        'rationale': 'Incoherencia estructural. SampEn se satura. Peso de compatibilidad decae rápido para forzar R3<umbral.'
    }
}

# ── VALIDACIÓN DEFENSIVA ──────────────────────────────────────
def _validate():
    """Valida claves, rangos y coherencia matemática al importar."""
    REQUIRED = {'m', 'r_ratio', 'mu', 'sigma'}
    RANGES   = {'m': (1, 4), 'r_ratio': (0.05, 0.50), 'mu': (0, 3.0), 'sigma': (0.001, 1.0)}
    
    for regime, cfg in SAMPEN_CONFIG.items():
        # 1. Claves obligatorias
        missing = REQUIRED - set(cfg.keys())
        if missing:
            raise ValueError(f"SAMPEN_CONFIG['{regime}'] falta claves: {missing}")
            
        # 2. Rangos numéricos
        for k, v in cfg.items():
            if k in RANGES and not (RANGES[k][0] <= v <= RANGES[k][1]):
                raise ValueError(f"SAMPEN_CONFIG['{regime}']['{k}']={v} fuera de rango [{RANGES[k][0]}, {RANGES[k][1]}]")
            
        # 3. Coherencia cruzada: mu vs r_ratio (SOLO AVISO, NO FRENÁ EJECUCIÓN)
        if cfg['mu'] > cfg['r_ratio'] * 5.0:
            warnings.warn(
                f"SAMPEN_CONFIG['{regime}']['mu']={cfg['mu']} es alto para r_ratio={cfg['r_ratio']} "
                f"(ajuste intencional para hipercaos/ruido)"
            )

_validate()  # ← Se ejecuta automáticamente al importar el módulo

# ── CÁLCULO CORE (NUMÉRICAMENTE HONESTO) ──────────────────────
def compute(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, tau: int = 1) -> float:
    """
    Sample Entropy con lag τ y tolerancia adaptativa.
    Elimina colapso a 0.2000, protege contra log(0) y std≈0.
    Complejidad: O(N²) en memoria → seguro para N ≤ 5000.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        std_x = np.std(x, ddof=1)
        
        if std_x < 1e-12:
            return np.nan  # Señal plana o degenerada
            
        r = r_ratio * std_x
        if N < (m + 1) * tau:
            return np.nan  # Serie insuficiente para embedding

        def _phi(m_val: int) -> float:
            n_t = N - (m_val - 1) * tau
            if n_t < 2: 
                return np.nan
                
            # Templates alineados con Takens (paso τ explícito)
            templates = np.array([x[j : j + m_val*tau : tau] for j in range(n_t)])
            
            # Distancia de Chebyshev (máxima por dimensión)
            dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
            
            # Excluir auto-comparación
            counts = np.sum(dists <= r, axis=1) - 1
            mean_c = np.mean(counts)
            
            return np.log(mean_c) if mean_c > 0 else -np.inf

        phi_m  = _phi(m)
        phi_m1 = _phi(m + 1)

        if np.isinf(phi_m) or np.isinf(phi_m1):
            return np.nan  # Sin pares compatibles → SampEn indefinido
            
        # SampEn = -ln(A/B) = ln(B) - ln(A)
        return float(phi_m - phi_m1)
    except Exception:
        return np.nan

# ── PESO DE COMPATIBILIDAD (GAUSSIANO SUAVE) ──────────────────
def compatibility_weight(sampen_val: float, regime: str) -> float:
    """Retorna [0,1] según cuán coherente es SampEn con el régimen detectado."""
    cfg = SAMPEN_CONFIG.get(regime, SAMPEN_CONFIG['weakly_chaotic'])
    sigma = max(cfg['sigma'], 1e-12)
    return float(np.exp(-0.5 * ((sampen_val - cfg['mu']) / sigma)**2))

# ── UTILIDADES DE CALIBRACIÓN (PARA MONTE CARLO / PAPERS) ─────
def sweep_sensitivity(factor: float = 1.0) -> Dict[str, Dict[str, float]]:
    """Genera variante calibrada para análisis de sensibilidad."""
    return {
        r: {k: (v * factor if k in ('r_ratio', 'sigma') else v) for k, v in cfg.items()}
        for r, cfg in SAMPEN_CONFIG.items()
    }
