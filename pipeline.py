"""
pipeline.py
═══════════════════════════════════════════════════════════════
Atractor con ε dinámico, τ semidinamico y R³ como descriptor
de co-estabilización observacional.
REVISIÓN 2026-05e: Fix NameError SampEnAdaptor + integración limpia
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ── IMPORTS EXTERNOS DE CALIBRACIÓN (BLINDADOS) ──────────────
try:
    from sampen_library import SAMPEN_CONFIG, compute as compute_sampen, compatibility_weight as sampen_weight
    from r3_delta_library import DELTA_LIBRARY
except ImportError:
    # Fallback total si las librerías no existen en el entorno
    print("⚠ Usando calibración fallback (sampen_library/r3_delta_library no encontrados)")
    SAMPEN_CONFIG = {
        'stable': {'m': 2, 'r_ratio': 0.15, 'mu': 0.02, 'sigma': 0.015},
        'weakly_chaotic': {'m': 2, 'r_ratio': 0.20, 'mu': 0.57, 'sigma': 0.12},
        'chaotic': {'m': 2, 'r_ratio': 0.25, 'mu': 0.53, 'sigma': 0.10},
        'hyperchaotic': {'m': 2, 'r_ratio': 0.30, 'mu': 1.51, 'sigma': 0.30},
        'noisy': {'m': 2, 'r_ratio': 0.35, 'mu': 1.65, 'sigma': 0.35}
    }
    DELTA_LIBRARY = {
        'stable': {'lambda': 0.02, 'D2': 0.10, 'LZ': 0.02, 'TE': 0.03, 'SampEn': 0.65},
        'weakly_chaotic': {'lambda': 0.08, 'D2': 0.05, 'LZ': 0.07, 'TE': 0.12, 'SampEn': 0.35},
        'chaotic': {'lambda': 0.19, 'D2': 0.02, 'LZ': 0.18, 'TE': 0.07, 'SampEn': 0.02},
        'hyperchaotic': {'lambda': 0.13, 'D2': 0.02, 'LZ': 0.02, 'TE': 0.06, 'SampEn': 0.17},
        'noisy': {'lambda': 0.20, 'D2': 0.05, 'LZ': 0.07, 'TE': 0.12, 'SampEn': 0.35}
    }
    compute_sampen = lambda x, m=2, r_ratio=0.2, tau=1: np.nan
    sampen_weight  = lambda v, r: 1.0


# ── 1. EMBEDDING DE TAKENS ────────────────────────────────────
def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"Serie demasiado corta para m={m}, τ={tau}.")
    return np.column_stack([x[i * tau : i * tau + n] for i in range(m)])


# ── 2. H1 — ε DINÁMICO ────────────────────────────────────────
class DynamicEpsilon:
    def __init__(self, k_neighbors: int = 5, scale: float = 0.5):
        self.k = k_neighbors
        self.scale = scale

    def compute_series(self, Y: np.ndarray) -> np.ndarray:
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y: np.ndarray) -> float:
        return float(np.median(self.compute_series(Y)))


# ── 3. H2 — τ SEMIDINAMICO ───────────────────────────────────
class SemidynamicTau:
    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = max_lag
        self.bins = bins
        self._cache: dict = {}

    def _ami(self, x: np.ndarray, lag: int) -> float:
        x1, x2 = x[:-lag], x[lag:]
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


# ── 4. MÉTRICAS DINÁMICAS ────────────────────────────────────
class Metrics:
    @staticmethod
    def lyapunov(x: np.ndarray, tau: int, m: int = 3, min_tsep: int = None, max_iter: int = 300) -> float:
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
            d_series = np.array([np.linalg.norm(Y[min(i+k, N-1)] - Y[min(j+k, N-1)]) for k in range(steps)])
            pos = d_series > 0
            if pos.sum() < 3:
                continue
            t = np.where(pos)[0]
            log_d = np.log(d_series[pos] / d[j])
            if len(t) > 1:
                divergences.append(np.polyfit(t, log_d, 1)[0])
        return float(np.median(divergences)) if divergences else np.nan

    @staticmethod
    def correlation_dimension(Y: np.ndarray, n_r: int = 20) -> float:
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
        r_min, r_max = np.percentile(flat, 5), np.percentile(flat, 45)
        if r_min >= r_max:
            return np.nan
        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        C_r = np.array([np.mean(flat < r) for r in r_vals])
        valid = (C_r > 0.01) & (C_r < 0.99)
        if valid.sum() < 4:
            return np.nan
        return float(np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)[0])

    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: np.ndarray = None) -> float:
        if Y is not None and Y.shape[0] > 10:
            Y_centered = Y - Y.mean(axis=0)
            cov = np.cov(Y_centered.T)
            if cov.ndim == 0:
                proj = Y_centered[:, 0]
            else:
                _, eigvecs = np.linalg.eigh(cov)
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

    @staticmethod
    def transfer_entropy(x: np.ndarray, tau: int, bins: int = 8) -> float:
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

    @classmethod
    def compute_all(cls, x: np.ndarray, tau: int, m: int = 3, regime: str = None) -> dict:
        Y = embed(x, m, tau)
        cfg = SAMPEN_CONFIG.get(regime, SAMPEN_CONFIG['weakly_chaotic']) if regime else {'m': 2, 'r_ratio': 0.2}
        return {
            'lambda': cls.lyapunov(x, tau, m),
            'D2':      cls.correlation_dimension(Y),
            'LZ':     cls.lempel_ziv(x, Y),
            'TE':     cls.transfer_entropy(x, tau),
            'SampEn': compute_sampen(x, m=cfg['m'], r_ratio=cfg['r_ratio'], tau=tau)
        }


# ── 5. DETECTOR DE RÉGIMEN ───────────────────────────────────
class RegimeDetector:
    DESCRIPTIONS = {
        'stable': 'Estable / Periódico',
        'weakly_chaotic': 'Caos débil / Cuasiperiódico',
        'chaotic': 'Caótico',
        'hyperchaotic': 'Hipercáótico / Estructurado',
        'noisy': 'Ruido / Sin estructura dinámica'
    }
    def classify(self, lam: float, lz: float = None, d2: float = None) -> str:
        if not np.isnan(lam) and lam < 0:
            return 'stable'
        if d2 is not None and not np.isnan(d2) and lz is not None and not np.isnan(lz):
            if lz > 0.95 and d2 > 2.3:
                return 'noisy'
            if lz > 0.92 and d2 > 2.0:
                return 'hyperchaotic'
            if lz > 0.55 and d2 > 1.6:
                return 'chaotic'
            if lz < 0.30 and d2 < 1.2:
                return 'stable'
            return 'weakly_chaotic'
        if lz is not None and not np.isnan(lz):
            if lz < 0.25: return 'stable'
            if lz < 0.55: return 'weakly_chaotic'
            if lz < 0.92: return 'chaotic'
            return 'hyperchaotic'
        if not np.isnan(lam):
            if lam < 0.15: return 'weakly_chaotic'
            if lam < 0.50: return 'chaotic'
            return 'hyperchaotic'
        return 'weakly_chaotic'


# ── 6. BIBLIOTECA DE δ (UI COMPATIBLE) ───────────────────────
class DeltaLibrary:
    def get(self, regime: str) -> dict:
        return DELTA_LIBRARY.get(regime, {
            'lambda': 0.08, 'D2': 0.05, 'LZ': 0.07, 'TE': 0.12, 'SampEn': 0.35
        })


# ── 7. H3 — R³ DESCRIPTOR ────────────────────────────────────
class R3Descriptor:
    COHERENCE_THRESHOLD = 0.57
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients(self, x: np.ndarray, tau: int, m: int) -> tuple:
        base = Metrics.compute_all(x, tau, m)
        regime = self.regime_detector.classify(
            base.get('lambda', np.nan), 
            base.get('LZ', np.nan), 
            base.get('D2', np.nan)
        )
        
        cfg = SAMPEN_CONFIG.get(regime, SAMPEN_CONFIG['weakly_chaotic'])
        base['SampEn'] = compute_sampen(x, m=cfg['m'], r_ratio=cfg['r_ratio'], tau=tau)
        base['_regime'] = regime

        tau_p, tau_m = max(1, tau + 1), max(1, tau - 1)
        mp = Metrics.compute_all(x, tau_p, m, regime=regime)
        mm = Metrics.compute_all(x, tau_m, m, regime=regime) if tau_m != tau else base

        grads = {}
        for k in ('lambda', 'D2', 'LZ', 'TE', 'SampEn'):
            v0, vp, vm = base.get(k, np.nan), mp.get(k, np.nan), mm.get(k, np.nan)
            if any(np.isnan([v0, vp, vm])):
                grads[k] = np.nan
            else:
                denom = np.sqrt(v0**2 + vp**2 + vm**2 + 1e-12) / np.sqrt(3)
                grads[k] = abs(vp - vm) / denom if denom > 1e-10 else abs(vp - vm)
        return grads, base

    def score(self, x: np.ndarray, tau: int, m: int = 3) -> dict:
        grads, metrics = self._gradients(x, tau, m)
        regime = metrics.pop('_regime', 'weakly_chaotic')
        delta_dict = self.delta_lib.get(regime)

        stability_map, stability_weights = {}, []
        for k, g in grads.items():
            if not np.isnan(g):
                delta_k = delta_dict.get(k, 0.1)
                w_stab = max(0.0, 1.0 - (g / delta_k)) if delta_k > 1e-12 else 0.0
                w_compat = sampen_weight(metrics['SampEn'], regime) if k == 'SampEn' else 1.0
                weight = w_stab * w_compat
                stability_weights.append(weight)
                stability_map[k] = {
                    'gradient': g, 'stable': g < delta_k, 'delta': delta_k,
                    'weight': weight, 'w_compat': w_compat
                }

        valid_n = len(stability_weights)
        if valid_n < 2:
            r3_score, r3_std, r3_min, r3_dominant = np.nan, np.nan, np.nan, 'insuficiente'
        else:
            r3_score = float(np.mean(stability_weights))
            r3_std = float(np.std(stability_weights))
            r3_min = float(np.min(stability_weights))
            r3_dominant = min(stability_map, key=lambda k: stability_map[k]['weight'])

        coherent = (
            not np.isnan(r3_score) and
            r3_score >= self.COHERENCE_THRESHOLD and
            r3_min >= self.COHERENCE_THRESHOLD / 2 and
            regime != 'noisy'
        )

        # Escalar para UI (ax.axvline)
        delta_scalar = float(np.mean(list(delta_dict.values())))

        return {
            'R3_score': r3_score, 'R3_std': r3_std, 'R3_min': r3_min, 'R3_dominant': r3_dominant,
            'R3_vector': {k: round(v['weight'], 8) for k, v in stability_map.items()},
            'coherent': coherent, 'regime': regime, 'regime_desc': RegimeDetector.DESCRIPTIONS.get(regime, regime),
            'delta': delta_scalar,
            'metrics': metrics, 'gradients': grads, 'stability_map': stability_map, 'n_valid': valid_n
        }


# ── 8. PIPELINE INTEGRADO ────────────────────────────────────
class AttractorPipeline:
    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m, self.verbose = m, verbose
        self._eps, self._tau, self._r3 = DynamicEpsilon(), SemidynamicTau(max_lag=max_tau), R3Descriptor()
        self.results: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, x: np.ndarray, label: str = 'serie') -> dict:
        x = np.asarray(x, dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-12)
        self._tau._cache.clear()

        tau0 = self._tau.compute(x, regime='unknown')
        metrics0 = Metrics.compute_all(x, tau0, self.m)
        regime0 = RegimeDetector().classify(metrics0['lambda'], metrics0['LZ'], metrics0['D2'])
        tau = self._tau.compute(x, regime=regime0)

        Y = embed(x, self.m, tau)
        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))

        metrics = Metrics.compute_all(x, tau, self.m)
        r3 = self._r3.score(x, tau, self.m)

        self._log(f"τ={tau} | ε={eps_scalar:.5f} | Régimen={r3['regime_desc']} | R3={r3['R3_score']:.6f}")

        return {
            'label': label, 'x_normalized': x, 'tau': tau, 'tau_initial': tau0,
            'epsilon': eps_scalar, 'epsilon_series': eps_series, 'embedding': Y,
            'metrics': metrics, 'regime': r3['regime'], 'regime_desc': r3['regime_desc'], 'R3': r3
        }


# ── SEÑALES DE REFERENCIA ────────────────────────────────────
def _logistic_map(N: int = 1000, r: float = 3.9) -> np.ndarray:
    x = 0.1; out = [x]
    for _ in range(N - 1): x = r * x * (1 - x); out.append(x)
    return np.array(out)

def demo_signals(N: int = 1000) -> dict:
    t = np.linspace(0, 100, N); rng = np.random.default_rng(0)
    def lorenz_ts(n=N, sigma=10, rho=28, beta=8/3, dt=0.01):
        x, y, z = 1.0, 1.0, 1.05
        for _ in range(2000):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
        xs = []
        for _ in range(n):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt; xs.append(x)
        return np.array(xs)
    return {
        'lorenz': lorenz_ts(), 'periodic': np.sin(2*np.pi*0.1*t) + 0.05*rng.normal(size=N),
        'noisy': rng.normal(size=N), 'logistic': _logistic_map(N, r=3.9),
    }

if __name__ == '__main__':
    pipe = AttractorPipeline(m=3, max_tau=50, verbose=True)
    for name, sig in demo_signals(800).items(): pipe.run(sig, label=name)
