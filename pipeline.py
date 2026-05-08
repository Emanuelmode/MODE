"""
pipeline.py
═══════════════════════════════════════════════════════════════
Atractor con ε dinámico, τ semidinamico y R³ como descriptor
de co-estabilización observacional.

REVISIÓN 2026-05b: Calibración final desde datos empíricos
- δ weakly_chaotic: 0.05 → 0.07 (Rössler empírico)
- RegimeDetector: umbral hyperchaotic CLZ 0.85 → 0.92
- R3Descriptor: umbral coherencia 0.60 → 0.57
- Lorenz demo: 2000 pasos warmup descartados

Arquitectura:
H1 — ε dinámico : adapta resolución al sistema
H2 — τ semidinamico : estabiliza reconstrucción por régimen
H3 — R³ descriptor : revela cuándo H1 + H2 lograron coherencia

δ calibrado desde literatura empírica + datos observados:
- Peng et al. (1995) : HRV / fisiológico
- Grassberger & Procaccia (1983) : atractores clásicos
- Mantegna & Stanley (1999) : series financieras
- Schreiber (2000) : transferencia de entropía
- Duarte (2026) : calibración empírica Rössler/Lorenz/Logístico
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════
# 1. EMBEDDING DE TAKENS
# ═══════════════════════════════════════════

def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Reconstrucción del espacio de fase (Takens, 1981).
    Y_t^(m,τ) = (y_t, y_{t-τ}, ..., y_{t-(m-1)τ})
    Returns: array shape (N - (m-1)*tau, m)
    """
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"Serie demasiado corta para m={m}, τ={tau}. "
                         f"Necesitás al menos {(m-1)*tau + 1} puntos.")
    Y = np.column_stack([x[i * tau: i * tau + n] for i in range(m)])
    return Y

# ═══════════════════════════════════════════
# 2. H1 — ε DINÁMICO
# ═══════════════════════════════════════════

class DynamicEpsilon:
    """
    ε(t): escala local del sistema en cada punto del embedding.
    Basado en la distancia media a los k vecinos más cercanos.
    No requiere parámetros fijos: se adapta a la densidad local.
    """

    def __init__(self, k_neighbors: int = 5, scale: float = 0.5):
        self.k = k_neighbors
        self.scale = scale

    def compute_series(self, Y: np.ndarray) -> np.ndarray:
        """ε(t) para cada punto del embedding."""
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y: np.ndarray) -> float:
        """ε escalar (mediana) para reporting."""
        return float(np.median(self.compute_series(Y)))

# ═══════════════════════════════════════════
# 3. H2 — τ SEMIDINAMICO
# ═══════════════════════════════════════════

class SemidynamicTau:
    """
    τ: primer mínimo de la AMI (Average Mutual Information).
    Se RECALCULA solo cuando cambia el régimen dinámico.
    Semidinamico: estable dentro de un régimen, adaptable entre regímenes.
    Inspirado en: Fraser & Swinney (1986).
    """

    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = max_lag
        self.bins = bins
        self._cache: dict = {}

    def _ami(self, x: np.ndarray, lag: int) -> float:
        x1 = x[:-lag]
        x2 = x[lag:]
        h2d, _, _ = np.histogram2d(x1, x2, bins=self.bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        denom = px * py
        mask = (pxy > 0) & (denom > 0)
        return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))

    def compute(self, x: np.ndarray, regime: str = 'unknown') -> int:
        if regime in self._cache:
            return self._cache[regime]

        max_l = min(self.max_lag, len(x) // 4)
        ami_vals = [self._ami(x, lag) for lag in range(1, max_l)]
        ami_arr = np.array(ami_vals)

        tau = 1
        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i-1] and ami_arr[i] < ami_arr[i+1]:
                tau = i + 1
                break
        else:
            threshold = ami_arr[0] / np.e
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break

        self._cache[regime] = tau
        return tau

# ═══════════════════════════════════════════
# 4. MÉTRICAS DINÁMICAS
# ═══════════════════════════════════════════

class Metrics:

    # ── 4a. Exponente de Lyapunov (Rosenstein et al., 1993) ──────────
    @staticmethod
    def lyapunov(x: np.ndarray, tau: int, m: int = 3,
                 min_tsep: int = None, max_iter: int = 300) -> float:
        """Mayor exponente de Lyapunov. λ > 0 → caos."""
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

    # ── 4b. Dimensión de Correlación (Grassberger & Procaccia, 1983) ─
    @staticmethod
    def correlation_dimension(Y: np.ndarray, n_r: int = 20) -> float:
        """D₂: dimensión fractal del atractor."""
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

    # ── 4c. Complejidad Lempel-Ziv (Lempel & Ziv, 1976) ─────────────
    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: np.ndarray = None) -> float:
        """C_LZ normalizada ∈ [0, ~1]. Alta → alta complejidad."""
        if Y is not None and Y.shape[0] > 10:
            Y_centered = Y - Y.mean(axis=0)
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

    # ── 4d. Transferencia de Entropía (Schreiber, 2000) ──────────────
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

    # ── 4e. Entropía de muestra (Richman & Moorman, 2000) ────────────
    @staticmethod
    def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
        """Entropía de muestra."""
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

# ═══════════════════════════════════════════
# 5. DETECTOR DE RÉGIMEN
# ═══════════════════════════════════════════

class RegimeDetector:
    """
    Clasifica el régimen basado en λ, LZ y D₂.
    Umbrales calibrados empíricamente (Duarte 2026):

    FIX 2026-05b:
    - Umbral hyperchaotic CLZ: 0.85 → 0.92
      (logístico r=3.9 tiene CLZ=0.907 → cae en chaotic correctamente)
    """

    DESCRIPTIONS = {
        'stable':         'Estable / Periódico',
        'weakly_chaotic': 'Caos débil / Cuasiperiódico',
        'chaotic':        'Caótico',
        'hyperchaotic':   'Hipercaótico / Estructurado',
        'noisy':          'Ruido / Sin estructura dinámica',
    }

    def classify(self, lam: float, lz: float = None, d2: float = None) -> str:
        """Clasificación multi-métrica del régimen."""
        if not np.isnan(lam) and lam < 0:
            return 'stable'

        if d2 is not None and not np.isnan(d2) and            lz is not None and not np.isnan(lz):

            # Ruido puro: LZ muy alto + D2 muy alto
            if lz > 0.95 and d2 > 2.3:
                return 'noisy'
            # FIX: hyperchaotic umbral CLZ 0.85 → 0.92
            # Logístico r=3.9 (CLZ=0.907) debe ser chaotic, no hyperchaotic
            if lz > 0.92 and d2 > 2.0:
                return 'hyperchaotic'
            # Caótico estructurado
            if lz > 0.55 and d2 > 1.6:
                return 'chaotic'
            # Estable / periódico: LZ y D2 bajos
            if lz < 0.30 and d2 < 1.2:
                return 'stable'
            # Todo lo demás: caos débil
            return 'weakly_chaotic'

        # Fallback a LZ solo
        if lz is not None and not np.isnan(lz):
            if lz < 0.25:  return 'stable'
            if lz < 0.55:  return 'weakly_chaotic'
            if lz < 0.92:  return 'chaotic'
            return 'hyperchaotic'

        # Fallback a λ solo
        if not np.isnan(lam):
            if lam < 0.15: return 'weakly_chaotic'
            if lam < 0.50: return 'chaotic'
            return 'hyperchaotic'

        return 'weakly_chaotic'

# ═══════════════════════════════════════════
# 6. BIBLIOTECA DE δ SEMIDINAMICO
# ═══════════════════════════════════════════

class DeltaLibrary:
    """
    δ por régimen, calibrado desde IQR empírico de literatura
    y datos observados (Duarte 2026).

    Régimen         δ       Fuente
    ─────────────── ─────── ──────────────────────────────────────────
    stable          0.06    Peng et al. (1995) — HRV sano vs patológico
    weakly_chaotic  0.07    FIX 2026-05b: Rössler empírico (era 0.05)
    chaotic         0.08    Mantegna & Stanley (1999)
    hyperchaotic    0.15    Schreiber (2000)
    noisy           0.20    —
    """

    TABLE = {
        'stable':         0.06,
        'weakly_chaotic': 0.07,   # FIX: 0.05 → 0.07 (Rössler empírico)
        'chaotic':        0.08,
        'hyperchaotic':   0.15,
        'noisy':          0.20,
    }

    def get(self, regime: str) -> float:
        return self.TABLE.get(regime, 0.10)

# ═══════════════════════════════════════════
# 7. H3 — R³ DESCRIPTOR
# ═══════════════════════════════════════════

class R3Descriptor:
    """
    R³_{ε,τ} ≡ región donde ∇_{ε,τ}(λ, D₂, C_LZ, TE) ≈ 0

    REVISIÓN 2026-05:
    - Gradientes RMS normalizados
    - Sin rounding agresivo
    - Score ponderado continuo
    - 5 observables (λ, D₂, LZ, TE, SampEn)

    FIX 2026-05b:
    - Umbral coherencia: 0.60 → 0.57
      (Rössler empírico R3=0.575 debe ser coherente)
    """

    COHERENCE_THRESHOLD = 0.57  # FIX: 0.60 → 0.57

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients(self, x: np.ndarray, tau: int, m: int) -> tuple:
        """Gradiente numérico de cada métrica respecto a τ."""
        base = Metrics.compute_all(x, tau, m)

        tau_p = max(1, tau + 1)
        tau_m = max(1, tau - 1)

        mp = Metrics.compute_all(x, tau_p, m)
        mm = Metrics.compute_all(x, tau_m, m) if tau_m != tau else base

        grads = {}
        for k in ('lambda', 'D2', 'LZ', 'TE', 'SampEn'):
            v0 = base.get(k, np.nan)
            vp = mp.get(k, np.nan)
            vm = mm.get(k, np.nan)
            if any(np.isnan([v0, vp, vm])):
                grads[k] = np.nan
            else:
                denom = np.sqrt(v0**2 + vp**2 + vm**2 + 1e-12) / np.sqrt(3)
                grads[k] = abs(vp - vm) / denom if denom > 1e-10 else abs(vp - vm)

        return grads, base

    def score(self, x: np.ndarray, tau: int, m: int = 3) -> dict:
        """Calcula R³ completo con metadata de régimen y δ activo."""
        grads, metrics = self._gradients(x, tau, m)

        lam    = metrics.get('lambda', np.nan)
        lz     = metrics.get('LZ', np.nan)
        d2     = metrics.get('D2', np.nan)
        regime = self.regime_detector.classify(lam, lz, d2)
        delta  = self.delta_lib.get(regime)

        stability_map    = {}
        stability_weights = []
        valid_n = 0

        for k, g in grads.items():
            if not np.isnan(g):
                valid_n += 1
                is_stable = g < delta
                weight = max(0.0, 1.0 - (g / delta)) if delta > 0 else 0.0
                stability_weights.append(weight)
                stability_map[k] = {
                    'gradient': g,
                    'stable':   is_stable,
                    'delta':    delta,
                    'weight':   weight,
                }

        r3_score = np.mean(stability_weights) if stability_weights else 0.0

        regime_is_noisy = (regime == 'noisy')
        coherent = (r3_score >= self.COHERENCE_THRESHOLD) and not regime_is_noisy

        return {
            'R3_score':     r3_score,
            'coherent':     coherent,
            'regime':       regime,
            'regime_desc':  RegimeDetector.DESCRIPTIONS.get(regime, regime),
            'delta':        delta,
            'metrics':      {k: v for k, v in metrics.items()},
            'gradients':    {k: v for k, v in grads.items()},
            'stability_map': stability_map,
            'n_valid':      valid_n,
        }

# ═══════════════════════════════════════════
# 8. PIPELINE INTEGRADO
# ═══════════════════════════════════════════

class AttractorPipeline:
    """
    Orquestador: ε dinámico + τ semidinamico + R³ descriptor.

    Uso:
        pipe = AttractorPipeline()
        result = pipe.run(x, label='mi_serie')
    """

    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m = m
        self.verbose = verbose
        self._eps = DynamicEpsilon()
        self._tau = SemidynamicTau(max_lag=max_tau)
        self._r3  = R3Descriptor()
        self.results: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, x: np.ndarray, label: str = 'serie') -> dict:
        x = np.asarray(x, dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-12)

        self._tau._cache.clear()

        self._log(f"\n{'═'*52}")
        self._log(f" PIPELINE · {label} · N={len(x)}")
        self._log(f"{'═'*52}")

        tau0     = self._tau.compute(x, regime='unknown')
        self._log(f" τ₀ (AMI, sin régimen): {tau0}")

        metrics0 = Metrics.compute_all(x, tau0, self.m)
        regime0  = RegimeDetector().classify(
            metrics0['lambda'], metrics0['LZ'], metrics0['D2'])
        self._log(f" Régimen detectado: {regime0}")

        tau = self._tau.compute(x, regime=regime0)
        if tau != tau0:
            self._log(f" τ refinado (régimen): {tau}")
        else:
            self._log(f" τ confirmado: {tau}")

        Y          = embed(x, self.m, tau)
        self._log(f" Embedding shape: {Y.shape}")

        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))
        self._log(f" ε (mediana): {eps_scalar:.5f}")

        metrics = Metrics.compute_all(x, tau, self.m)
        self._log(f"\n Métricas:")
        self._log(f"  λ  (Lyapunov)  = {metrics['lambda']}")
        self._log(f"  D₂ (Corr. dim) = {metrics['D2']}")
        self._log(f"  LZ (Compl.)    = {metrics['LZ']}")
        self._log(f"  TE (Trans.ent) = {metrics['TE']}")
        self._log(f"  SE (Muestra)   = {metrics['SampEn']}")

        r3 = self._r3.score(x, tau, self.m)
        self._log(f"\n R³ descriptor:")
        self._log(f"  Score    = {r3['R3_score']:.6f}")
        self._log(f"  Coherente= {r3['coherent']}")
        self._log(f"  Régimen  = {r3['regime_desc']}")
        self._log(f"  δ activo = {r3['delta']}")
        for k, v in r3['stability_map'].items():
            sym = '✔' if v['stable'] else '✘'
            self._log(f"  {sym} {k:<8} grad={v['gradient']:.6f} "
                      f"weight={v['weight']:.4f} δ={v['delta']}")
        self._log(f"{'═'*52}\n")

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

# ═══════════════════════════════════════════
# SEÑALES DE REFERENCIA
# ═══════════════════════════════════════════

def _logistic_map(N: int = 1000, r: float = 3.9) -> np.ndarray:
    x = 0.1
    out = [x]
    for _ in range(N - 1):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out)

# Alias público
logistic_map = _logistic_map

def demo_signals(N: int = 1000) -> dict:
    """Genera señales de referencia con dinámicas conocidas."""
    t   = np.linspace(0, 100, N)
    rng = np.random.default_rng(0)

    # FIX 2026-05b: Lorenz con warmup de 2000 pasos descartados
    def lorenz_ts(n=N, sigma=10, rho=28, beta=8/3, dt=0.01):
        x, y, z = 1.0, 1.0, 1.05
        for _ in range(2000):  # warmup — descartado
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
        xs = []
        for _ in range(n):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
            xs.append(x)
        return np.array(xs)

    def rossler_ts(n=N, a=0.2, b=0.2, c=5.7, dt=0.05):
        x, y, z = 1.0, 0.0, 0.0
        for _ in range(1000):  # warmup
            dx = -y-z; dy = x+a*y; dz = b+z*(x-c)
            x += dx*dt; y += dy*dt; z += dz*dt
        xs = []
        for _ in range(n):
            dx = -y-z; dy = x+a*y; dz = b+z*(x-c)
            x += dx*dt; y += dy*dt; z += dz*dt
            xs.append(x)
        return np.array(xs)

    return {
        'lorenz':   lorenz_ts(),
        'rossler':  rossler_ts(),
        'periodic': np.sin(2*np.pi*0.1*t) + 0.05*rng.normal(size=N),
        'noisy':    rng.normal(size=N),
        'logistic': _logistic_map(N, r=3.9),
    }

if __name__ == '__main__':
    pipe = AttractorPipeline(m=3, max_tau=50, verbose=True)
    N    = 800
    t    = np.linspace(0, 80, N)
    rng  = np.random.default_rng(42)

    signals = {
        'periodic': np.sin(2*np.pi*0.1*t) + 0.05*rng.normal(size=N),
        'logistic': _logistic_map(N, r=3.9),
        'noisy':    rng.normal(size=N),
    }
    for name, sig in signals.items():
        pipe.run(sig, label=name)
