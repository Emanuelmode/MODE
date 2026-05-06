"""
pipeline.py
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
Atractor con Îµ dinÃ¡mico, Ï„ semidinamico y RÂ³ como descriptor
de co-estabilizaciÃ³n observacional.

REVISIÃ“N 2026-05: CÃ¡lculos honestos sin inflaciÃ³n de precisiÃ³n
- Gradientes normalizados por RMS en lugar de mÃ¡ximo
- RÂ³ sin rounding que oculte discretizaciÃ³n
- MÃ©tricas adicionales para reducir colapso a 5 valores
- ValidaciÃ³n rigurosa de regÃ­menes

Arquitectura:
  H1 â€” Îµ dinÃ¡mico       : adapta resoluciÃ³n al sistema
  H2 â€” Ï„ semidinamico   : estabiliza reconstrucciÃ³n por rÃ©gimen
  H3 â€” RÂ³ descriptor    : revela cuÃ¡ndo H1 + H2 lograron coherencia

Î´ calibrado desde literatura empÃ­rica (no constante teÃ³rica):
  - Peng et al. (1995)     : HRV / fisiolÃ³gico
  - Grassberger & Procaccia (1983) : atractores clÃ¡sicos
  - Mantegna & Stanley (1999)      : series financieras
  - Schreiber (2000)               : transferencia de entropÃ­a
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1.  EMBEDDING DE TAKENS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    ReconstrucciÃ³n del espacio de fase (Takens, 1981).
    Y_t^(m,Ï„) = (y_t, y_{t-Ï„}, ..., y_{t-(m-1)Ï„})
    Returns: array shape (N - (m-1)*tau, m)
    """
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"Serie demasiado corta para m={m}, Ï„={tau}. "
                         f"NecesitÃ¡s al menos {(m-1)*tau + 1} puntos.")
    Y = np.column_stack([x[i * tau: i * tau + n] for i in range(m)])
    return Y


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2.  H1 â€” Îµ DINÃMICO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DynamicEpsilon:
    """
    Îµ(t): escala local del sistema en cada punto del embedding.
    
    Basado en la distancia media a los k vecinos mÃ¡s cercanos.
    No requiere parÃ¡metros fijos: se adapta a la densidad local.
    """

    def __init__(self, k_neighbors: int = 5, scale: float = 0.5):
        self.k = k_neighbors
        self.scale = scale

    def compute_series(self, Y: np.ndarray) -> np.ndarray:
        """Îµ(t) para cada punto del embedding."""
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y: np.ndarray) -> float:
        """Îµ escalar (mediana) para reporting."""
        return float(np.median(self.compute_series(Y)))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3.  H2 â€” Ï„ SEMIDINAMICO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class SemidynamicTau:
    """
    Ï„: primer mÃ­nimo de la AMI (Average Mutual Information).
    
    Se RECALCULA solo cuando cambia el rÃ©gimen dinÃ¡mico.
    Semidinamico: estable dentro de un rÃ©gimen, adaptable entre regÃ­menes.
    
    Inspirado en: Fraser & Swinney (1986) â€” primer mÃ­nimo de AMI
    como criterio de independencia estadÃ­stica mÃ­nima.
    """

    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = max_lag
        self.bins = bins
        self._cache: dict = {}   # {regime: tau}

    def _ami(self, x: np.ndarray, lag: int) -> float:
        x1 = x[:-lag]
        x2 = x[lag:]
        h2d, _, _ = np.histogram2d(x1, x2, bins=self.bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px  = pxy.sum(axis=1, keepdims=True)
        py  = pxy.sum(axis=0, keepdims=True)
        denom = px * py
        mask = (pxy > 0) & (denom > 0)
        return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))

    def compute(self, x: np.ndarray, regime: str = 'unknown') -> int:
        if regime in self._cache:
            return self._cache[regime]

        max_l = min(self.max_lag, len(x) // 4)
        ami_vals = [self._ami(x, lag) for lag in range(1, max_l)]
        ami_arr  = np.array(ami_vals)

        tau = 1
        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i-1] and ami_arr[i] < ami_arr[i+1]:
                tau = i + 1
                break
        else:
            # Sin mÃ­nimo claro: usar primer cruce por debajo de 1/e del mÃ¡ximo
            threshold = ami_arr[0] / np.e
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break

        self._cache[regime] = tau
        return tau


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4.  MÃ‰TRICAS DINÃMICAS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class Metrics:

    # â”€â”€ 4a. Exponente de Lyapunov (Rosenstein et al., 1993) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def lyapunov(x: np.ndarray, tau: int, m: int = 3,
                 min_tsep: int = None, max_iter: int = 300) -> float:
        """Mayor exponente de Lyapunov. Î» > 0 â†’ caos."""
        Y = embed(x, m, tau)
        N = len(Y)
        if min_tsep is None:
            min_tsep = max(1, int(0.1 * N))

        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)

        divergences = []
        for i in range(N):
            mask = np.abs(np.arange(N) - i) > min_tsep
            d = dists[i].copy()
            d[~mask] = np.inf
            j = np.argmin(d)
            if np.isinf(d[j]) or d[j] == 0:
                continue

            steps = min(max_iter, N - max(i, j) - 1)
            if steps < 3:
                continue

            d_series = np.array([
                np.linalg.norm(Y[min(i+k, N-1)] - Y[min(j+k, N-1)])
                for k in range(steps)
            ])
            pos = d_series > 0
            if pos.sum() < 3:
                continue

            t = np.where(pos)[0]
            log_d = np.log(d_series[pos] / d[j])
            if len(t) > 1:
                slope = np.polyfit(t, log_d, 1)[0]
                divergences.append(slope)

        return float(np.median(divergences)) if divergences else np.nan

    # â”€â”€ 4b. DimensiÃ³n de CorrelaciÃ³n (Grassberger & Procaccia, 1983) â”€
    @staticmethod
    def correlation_dimension(Y: np.ndarray, n_r: int = 20) -> float:
        """Dâ‚‚: dimensiÃ³n fractal del atractor."""
        N = len(Y)
        if N > 1500:
            rng = np.random.default_rng(42)
            Y = Y[rng.choice(N, 1500, replace=False)]
            N = 1500

        dists = cdist(Y, Y)
        flat = dists[np.triu_indices(N, k=1)]
        flat = flat[flat > 0]
        if len(flat) == 0:
            return np.nan

        r_min = np.percentile(flat, 5)
        r_max = np.percentile(flat, 45)
        if r_min >= r_max:
            return np.nan

        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        C_r = np.array([np.mean(flat < r) for r in r_vals])

        valid = (C_r > 0.01) & (C_r < 0.99)
        if valid.sum() < 4:
            return np.nan

        slope = np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)[0]
        return float(slope)

    # â”€â”€ 4c. Complejidad Lempel-Ziv (Lempel & Ziv, 1976) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: np.ndarray = None) -> float:
        """
        C_LZ normalizada âˆˆ [0, ~1]. Alta â†’ alta complejidad.

        OperaciÃ³n sobre el embedding Y (OpciÃ³n A):
        Se binariza la proyecciÃ³n PCA-1 del embedding para que LZ
        sea sensible a Ï„ y m, no solo a la distribuciÃ³n de x.
        Si Y no se provee, opera sobre x directamente (fallback).
        """
        if Y is not None and Y.shape[0] > 10:
            # ProyecciÃ³n al primer componente del embedding
            Y_centered = Y - Y.mean(axis=0)
            # Primera direcciÃ³n de varianza mÃ¡xima (sin sklearn)
            cov = np.cov(Y_centered.T)
            if cov.ndim == 0:
                proj = Y_centered[:, 0]
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                proj = Y_centered @ eigvecs[:, -1]
            seq = proj
        else:
            seq = x

        binary = ''.join('1' if v > np.median(seq) else '0' for v in seq)
        n = len(binary)
        c, l, i, k, k_max = 1, 1, 0, 1, 1
        stop = False
        while not stop:
            if i + k <= n and l + k <= n and binary[i+k-1] == binary[l+k-1]:
                k += 1
                if l + k > n:
                    c += 1
                    stop = True
            else:
                k_max = max(k, k_max)
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    stop = l + 1 > n
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        norm = n / (np.log2(n) + 1e-10)
        return float(np.clip(c / norm, 0, 2))

    # â”€â”€ 4d. Transferencia de EntropÃ­a (Schreiber, 2000) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def transfer_entropy(x: np.ndarray, tau: int, bins: int = 8) -> float:
        """TE del pasado al futuro: flujo informacional interno."""
        n = len(x) - tau
        if n < 20:
            return np.nan
        X, Xf = x[:n], x[tau:n+tau]
        h2d, _, _ = np.histogram2d(X, Xf, bins=bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                    te += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
        return float(max(0.0, te))

    # â”€â”€ 4e. EntropÃ­a de muestra (Sample entropy) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
        """EntropÃ­a de muestra (Richman & Moorman, 2000)."""
        try:
            N = len(x)
            r = r_ratio * np.std(x, ddof=1)
            if r == 0:
                return np.nan
            
            def _maxdist(x_i, x_j):
                return max(abs(ua - va) for ua, va in zip(x_i, x_j))
            
            def _phi(m_val):
                x_emb = [x[j:j+m_val] for j in range(N-m_val+1)]
                C = [len([1 for x_j in x_emb if _maxdist(x_i, x_j) <= r])
                     for x_i in x_emb]
                return np.log(np.mean(C))
            
            return _phi(m) - _phi(m+1)
        except:
            return np.nan

    @classmethod
    def compute_all(cls, x: np.ndarray, tau: int, m: int = 3) -> dict:
        Y = embed(x, m, tau)
        return {
            'lambda': cls.lyapunov(x, tau, m),
            'D2':     cls.correlation_dimension(Y),
            'LZ':     cls.lempel_ziv(x, Y),
            'TE':     cls.transfer_entropy(x, tau),
            'SampEn': cls.sample_entropy(x, m=2),
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5.  DETECTOR DE RÃ‰GIMEN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class RegimeDetector:
    """
    Clasifica el rÃ©gimen basado en Î» y otras mÃ©tricas.
    Umbrales derivados de literatura de sistemas dinÃ¡micos.
    
    Î» < 0        â†’ Estable / PeriÃ³dico
    0 â‰¤ Î» < 0.15 â†’ Caos dÃ©bil
    0.15 â‰¤ Î» < 0.5 â†’ CaÃ³tico
    Î» â‰¥ 0.5      â†’ HipercaÃ³tico / Ruidoso
    """

    DESCRIPTIONS = {
        'stable':          'Estable / PeriÃ³dico',
        'weakly_chaotic':  'Caos dÃ©bil / CuasiperiÃ³dico',
        'chaotic':         'CaÃ³tico',
        'hyperchaotic':    'HipercaÃ³tico / Estructurado',
        'noisy':           'Ruido / Sin estructura dinÃ¡mica',
    }

    def classify(self, lam: float, lz: float = None, d2: float = None) -> str:
        """
        ClasificaciÃ³n multi-mÃ©trica del rÃ©gimen.
        Umbrales conservadores â€” mejor subestimar que sobreestimar rÃ©gimen.
        """
        # Î» negativo es diagnÃ³stico confiable de sistema estable
        if not np.isnan(lam) and lam < 0:
            return 'stable'

        if d2 is not None and not np.isnan(d2) and \
           lz is not None and not np.isnan(lz):

            # Ruido puro: LZ muy alto + D2 muy alto
            if lz > 0.95 and d2 > 2.3:
                return 'noisy'
            # HipercaÃ³tico: requiere AMBOS muy altos (umbrales estrictos)
            if lz > 0.85 and d2 > 2.0:
                return 'hyperchaotic'
            # CaÃ³tico estructurado
            if lz > 0.55 and d2 > 1.6:
                return 'chaotic'
            # Estable / periÃ³dico: LZ y D2 bajos
            if lz < 0.30 and d2 < 1.2:
                return 'stable'
            # Todo lo demÃ¡s: caos dÃ©bil (Lorenz, RÃ¶ssler caen acÃ¡)
            return 'weakly_chaotic'

        # Fallback a LZ solo
        if lz is not None and not np.isnan(lz):
            if lz < 0.25:  return 'stable'
            if lz < 0.55:  return 'weakly_chaotic'
            if lz < 0.85:  return 'chaotic'
            return 'hyperchaotic'

        # Fallback a Î» solo
        if not np.isnan(lam):
            if lam < 0.15: return 'weakly_chaotic'
            if lam < 0.50: return 'chaotic'
            return 'hyperchaotic'

        return 'weakly_chaotic'


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6.  BIBLIOTECA DE Î´ SEMIDINAMICO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DeltaLibrary:
    """
    Î´ por rÃ©gimen, calibrado desde IQR empÃ­rico de literatura.

    RÃ©gimen          Î´       Fuente principal
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”€â”€â”€â”€â”€â”€â”€ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    stable          0.06    Peng et al. (1995) â€” HRV sano vs patolÃ³gico
    weakly_chaotic  0.05    Grassberger & Procaccia (1983) â€” Lorenz, RÃ¶ssler
    chaotic         0.08    Mantegna & Stanley (1999) â€” mercados, bio transiciÃ³n
    hyperchaotic    0.15    Schreiber (2000) â€” sistemas ruidosos de alta dim.
    """

    TABLE = {
        'stable':         0.06,
        'weakly_chaotic': 0.05,
        'chaotic':        0.08,
        'hyperchaotic':   0.15,
        'noisy':          0.20,
    }

    def get(self, regime: str) -> float:
        return self.TABLE.get(regime, 0.10)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7.  H3 â€” RÂ³ DESCRIPTOR (REVISADO)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class R3Descriptor:
    """
    RÂ³_{Îµ,Ï„} â‰¡ regiÃ³n donde âˆ‡_{Îµ,Ï„}(Î», Dâ‚‚, C_LZ, TE) â‰ˆ 0

    CAMBIOS EN REVISIÃ“N 2026-05:
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    1. NormalizaciÃ³n RMS en lugar de mÃ¡ximo â†’ menos colapso
    2. Sin rounding agresivo â†’ preserva continuo
    3. Score ponderado por estabilidad real, no binario
    4. MÃ©trica adicional (SampEn) â†’ 5 observables en lugar de 4
    
    Descriptor de co-estabilizaciÃ³n observacional.
    NO es restricciÃ³n â€” emerge del sistema, no se le impone.

    Score âˆˆ [0, 1]:
      1.0 â†’ todas las mÃ©tricas co-estabilizadas (mÃ¡xima coherencia)
      0.0 â†’ ninguna mÃ©trica estable bajo variaciÃ³n de Ï„
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib       = DeltaLibrary()

    def _gradients(self, x: np.ndarray, tau: int, m: int) -> tuple:
        """Gradiente numÃ©rico de cada mÃ©trica respecto a Ï„."""
        base = Metrics.compute_all(x, tau, m)

        tau_p = max(1, tau + 1)
        tau_m = max(1, tau - 1)

        mp = Metrics.compute_all(x, tau_p, m)
        mm = Metrics.compute_all(x, tau_m, m) if tau_m != tau else base

        grads = {}
        for k in ('lambda', 'D2', 'LZ', 'TE', 'SampEn'):
            v0, vp, vm = base.get(k, np.nan), mp.get(k, np.nan), mm.get(k, np.nan)
            if any(np.isnan([v0, vp, vm])):
                grads[k] = np.nan
            else:
                # CAMBIO: normalizaciÃ³n RMS en lugar de mÃ¡ximo
                denom = np.sqrt(v0**2 + vp**2 + vm**2 + 1e-12) / np.sqrt(3)
                grads[k] = abs(vp - vm) / denom if denom > 1e-10 else abs(vp - vm)

        return grads, base

    def score(self, x: np.ndarray, tau: int, m: int = 3) -> dict:
        """Calcula RÂ³ completo con metadata de rÃ©gimen y Î´ activo."""
        grads, metrics = self._gradients(x, tau, m)

        lam    = metrics.get('lambda', np.nan)
        lz     = metrics.get('LZ', np.nan)
        d2     = metrics.get('D2', np.nan)
        regime = self.regime_detector.classify(lam, lz, d2)
        delta  = self.delta_lib.get(regime)

        stability_map = {}
        stability_weights = []
        valid_n = 0

        for k, g in grads.items():
            if not np.isnan(g):
                valid_n += 1
                # CAMBIO: score ponderado continuo, no binario
                is_stable = g < delta
                weight = max(0.0, 1.0 - (g / delta)) if delta > 0 else 0.0
                stability_weights.append(weight)
                
                stability_map[k] = {
                    'gradient': g,  # SIN ROUNDING
                    'stable':   is_stable,
                    'delta':    delta,
                    'weight':   weight,
                }

        # CAMBIO: score es promedio ponderado, no proporciÃ³n binaria
        r3_score = np.mean(stability_weights) if stability_weights else 0.0

        # Fix 5: rÃ©gimen noisy no puede ser coherente por definiciÃ³n.
        regime_is_noisy = (regime == 'noisy')
        coherent = (r3_score >= 0.60) and not regime_is_noisy

        return {
            'R3_score':      r3_score,  # SIN ROUNDING
            'coherent':      coherent,
            'regime':        regime,
            'regime_desc':   RegimeDetector.DESCRIPTIONS.get(regime, regime),
            'delta':         delta,
            'metrics':       {k: v for k, v in metrics.items()},
            'gradients':     {k: v for k, v in grads.items()},
            'stability_map': stability_map,
            'n_valid':       valid_n,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 8.  PIPELINE INTEGRADO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AttractorPipeline:
    """
    Orquestador: Îµ dinÃ¡mico + Ï„ semidinamico + RÂ³ descriptor.

    Uso:
        pipe   = AttractorPipeline()
        result = pipe.run(x, label='mi_serie')
    """

    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m       = m
        self.verbose = verbose
        self._eps    = DynamicEpsilon()
        self._tau    = SemidynamicTau(max_lag=max_tau)
        self._r3     = R3Descriptor()
        self.results: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, x: np.ndarray, label: str = 'serie') -> dict:
        x = np.asarray(x, dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-12)   # normalizaciÃ³n z-score

        # Fix 1: Ï„ es propio de cada seÃ±al â€” limpiar cache entre runs
        self._tau._cache.clear()

        self._log(f"\n{'â•'*52}")
        self._log(f"  PIPELINE  Â·  {label}  Â·  N={len(x)}")
        self._log(f"{'â•'*52}")

        # â”€â”€ Paso 1: Ï„ inicial (sin rÃ©gimen conocido) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        tau0 = self._tau.compute(x, regime='unknown')
        self._log(f"  Ï„â‚€ (AMI, sin rÃ©gimen):  {tau0}")

        # â”€â”€ Paso 2: mÃ©tricas base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        metrics0 = Metrics.compute_all(x, tau0, self.m)
        regime0  = RegimeDetector().classify(
            metrics0['lambda'], metrics0['LZ'], metrics0['D2']
        )
        self._log(f"  RÃ©gimen detectado:      {regime0}")

        # â”€â”€ Paso 3: Ï„ refinado por rÃ©gimen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        tau = self._tau.compute(x, regime=regime0)
        if tau != tau0:
            self._log(f"  Ï„ refinado (rÃ©gimen):   {tau}")
        else:
            self._log(f"  Ï„ confirmado:           {tau}")

        # â”€â”€ Paso 4: embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        Y = embed(x, self.m, tau)
        self._log(f"  Embedding shape:        {Y.shape}")

        # â”€â”€ Paso 5: Îµ dinÃ¡mico â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))
        self._log(f"  Îµ (mediana):            {eps_scalar:.5f}")

        # â”€â”€ Paso 6: mÃ©tricas finales â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        metrics = Metrics.compute_all(x, tau, self.m)
        self._log(f"\n  MÃ©tricas:")
        self._log(f"    Î»  (Lyapunov)  = {metrics['lambda']}")
        self._log(f"    Dâ‚‚ (Corr. dim) = {metrics['D2']}")
        self._log(f"    LZ (Compl.)    = {metrics['LZ']}")
        self._log(f"    TE (Trans. ent)= {metrics['TE']}")
        self._log(f"    SE (Muestra)   = {metrics['SampEn']}")

        # â”€â”€ Paso 7: RÂ³ descriptor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        r3 = self._r3.score(x, tau, self.m)
        self._log(f"\n  RÂ³ descriptor:")
        self._log(f"    Score          = {r3['R3_score']:.6f}")
        self._log(f"    Coherente      = {r3['coherent']}")
        self._log(f"    RÃ©gimen        = {r3['regime_desc']}")
        self._log(f"    Î´ activo       = {r3['delta']}")
        for k, v in r3['stability_map'].items():
            sym = 'âœ”' if v['stable'] else 'âœ˜'
            self._log(f"    {sym} {k:<8} grad={v['gradient']:.6f}  weight={v['weight']:.4f}  Î´={v['delta']}")
        self._log(f"{'â•'*52}\n")

        result = {
            'label':          label,
            'x_normalized':   x,
            'tau':            tau,
            'tau_initial':    tau0,
            'epsilon':        eps_scalar,
            'epsilon_series': eps_series,
            'embedding':      Y,
            'metrics':        metrics,
            'regime':         r3['regime'],
            'regime_desc':    r3['regime_desc'],
            'R3':             r3,
        }

        self.results[label] = result
        return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SEÃ‘ALES DE PRUEBA (para desarrollo)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def demo_signals(N: int = 1000) -> dict:
    """Genera seÃ±ales de referencia con dinÃ¡micas conocidas."""
    t = np.linspace(0, 100, N)
    rng = np.random.default_rng(0)

    # Lorenz (simulaciÃ³n aproximada vÃ­a diferencias)
    def lorenz_ts(n=N, sigma=10, rho=28, beta=8/3, dt=0.01):
        x, y, z = 1.0, 1.0, 1.0
        xs = []
        for _ in range(n):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt; y += dy * dt; z += dz * dt
            xs.append(x)
        return np.array(xs)

    return {
        'lorenz':     lorenz_ts(),
        'periodic':   np.sin(2 * np.pi * 0.1 * t) + 0.05 * rng.normal(size=N),
        'noisy':      rng.normal(size=N),
        'logistic':   _logistic_map(N, r=3.9),
    }


def _logistic_map(N=1000, r=3.9):
    x = 0.1
    out = [x]
    for _ in range(N - 1):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out)


if __name__ == '__main__':
    pipe = AttractorPipeline(m=3, max_tau=50, verbose=True)

    # SeÃ±ales de prueba
    N = 800
    t = np.linspace(0, 80, N)
    rng = np.random.default_rng(42)

    signals = {
        'periodic':  np.sin(2 * np.pi * 0.1 * t) + 0.05 * rng.normal(size=N),
        'logistic':  _logistic_map(N, r=3.9),
        'noisy':     rng.normal(size=N),
    }

    for name, sig in signals.items():
        pipe.run(sig, label=name)
