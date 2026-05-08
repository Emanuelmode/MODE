"""
pipeline.py
MODE Pipeline · Emanuel Duarte · 2026
Atractor con epsilon dinamico, tau semidinamico y R3
como descriptor de co-estabilizacion observacional.
Revision 2026-05: Calculos HONESTOS, sin inflacion de precision.
"""

import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# SEÑAL DE PRUEBA (logisticmap exportado — requerido por app.py)
# ─────────────────────────────────────────────────────────────────

def logisticmap(N=1000, r=3.9):
    x = 0.1
    out = [x]
    for _ in range(N - 1):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out)


# ─────────────────────────────────────────────────────────────────
# 1. EMBEDDING DE TAKENS
# ─────────────────────────────────────────────────────────────────

def embed(x, m, tau):
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(
            f"Serie demasiado corta para m={m}, tau={tau}. "
            f"Necesitas al menos {(m-1)*tau + 1} puntos."
        )
    Y = np.column_stack([x[i * tau: i * tau + n] for i in range(m)])
    return Y


# ─────────────────────────────────────────────────────────────────
# 2. EPSILON DINAMICO
# ─────────────────────────────────────────────────────────────────

class DynamicEpsilon:
    def __init__(self, k_neighbors=5, scale=0.5):
        self.k = k_neighbors
        self.scale = scale

    def compute_series(self, Y):
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y):
        return float(np.median(self.compute_series(Y)))


# ─────────────────────────────────────────────────────────────────
# 3. TAU SEMIDINAMICO (AMI)
# ─────────────────────────────────────────────────────────────────

class SemidynamicTau:
    def __init__(self, max_lag=50, bins=16):
        self.max_lag = max_lag
        self.bins = bins
        self.cache = {}

    def _ami(self, x, lag):
        x1 = x[:-lag]
        x2 = x[lag:]
        h2d, _, _ = np.histogram2d(x1, x2, bins=self.bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        denom = px * py
        mask = (pxy > 0) & (denom > 0)
        return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))

    def compute(self, x, regime='unknown'):
        if regime in self.cache:
            return self.cache[regime]
        max_l = min(self.max_lag, len(x) // 4)
        if max_l < 2:
            return 1
        ami_vals = [self._ami(x, lag) for lag in range(1, max_l)]
        ami_arr = np.array(ami_vals)
        tau = 1
        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i - 1] and ami_arr[i] < ami_arr[i + 1]:
                tau = i + 1
                break
        else:
            threshold = ami_arr[0] / np.e if len(ami_arr) > 0 else 1
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break
        tau = max(1, tau)
        self.cache[regime] = tau
        return tau


# ─────────────────────────────────────────────────────────────────
# 4. METRICAS DINAMICAS
# ─────────────────────────────────────────────────────────────────

class Metrics:

    @staticmethod
    def lyapunov(x, tau, m=3, min_tsep=None, max_iter=300):
        try:
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
                    np.linalg.norm(Y[min(i + k, N - 1)] - Y[min(j + k, N - 1)])
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
        except Exception:
            return np.nan

    @staticmethod
    def correlation_dimension(Y, n_r=20):
        try:
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
        except Exception:
            return np.nan

    @staticmethod
    def lempel_ziv(x, Y=None):
        try:
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
            if n < 4:
                return np.nan
            c, l, i, k, k_max = 1, 1, 0, 1, 1
            stop = False
            while not stop:
                if i + k <= n and l + k <= n and binary[i + k - 1] == binary[l + k - 1]:
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
        except Exception:
            return np.nan

    @staticmethod
    def transfer_entropy(x, tau, bins=8):
        try:
            n = len(x) - tau
            if n < 20:
                return np.nan
            X, Xf = x[:n], x[tau:n + tau]
            h2d, _, _ = np.histogram2d(X, Xf, bins=bins)
            pxy = h2d / (h2d.sum() + 1e-12)
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            te = 0.0
            for i in range(bins):
                for j in range(bins):
                    if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        te += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
            return float(max(0.0, te))
        except Exception:
            return np.nan

    @staticmethod
    def sample_entropy(x, m=2, r_ratio=0.2):
        try:
            N = len(x)
            if N < 10:
                return np.nan
            r = r_ratio * np.std(x, ddof=1)
            if r == 0:
                return np.nan

            def _maxdist(xi, xj):
                return max(abs(a - b) for a, b in zip(xi, xj))

            def _phi(m_val):
                x_emb = [x[j:j + m_val] for j in range(N - m_val + 1)]
                C = [len([1 for xj in x_emb if _maxdist(xi, xj) <= r]) for xi in x_emb]
                return np.log(np.mean(C) + 1e-12)

            return float(_phi(m) - _phi(m + 1))
        except Exception:
            return np.nan

    @classmethod
    def compute_all(cls, x, tau, m=3):
        try:
            Y = embed(x, m, tau)
        except Exception:
            Y = None
        return {
            'lambda': cls.lyapunov(x, tau, m),
            'D2': cls.correlation_dimension(Y) if Y is not None else np.nan,
            'LZ': cls.lempel_ziv(x, Y),
            'TE': cls.transfer_entropy(x, tau),
            'SampEn': cls.sample_entropy(x, m=2),
        }


# ─────────────────────────────────────────────────────────────────
# 5. DETECTOR DE REGIMEN
# ─────────────────────────────────────────────────────────────────

class RegimeDetector:
    DESCRIPTIONS = {
        'stable': 'Estable / Periodico',
        'weakly_chaotic': 'Caos debil / Cuasiperiodico',
        'chaotic': 'Caotico',
        'hyperchaotic': 'Hipercaotico / Estructurado',
        'noisy': 'Ruido / Sin estructura dinamica',
    }

    def classify(self, lam, lz=None, d2=None):
        if lam is not None and not np.isnan(lam) and lam < 0:
            return 'stable'
        if d2 is not None and not np.isnan(d2) and lz is not None and not np.isnan(lz):
            if lz > 0.95 and d2 > 2.3:
                return 'noisy'
            if lz > 0.85 and d2 > 2.0:
                return 'hyperchaotic'
            if lz > 0.55 and d2 > 1.6:
                return 'chaotic'
            if lz < 0.30 and d2 < 1.2:
                return 'stable'
            return 'weakly_chaotic'
        if lz is not None and not np.isnan(lz):
            if lz < 0.25: return 'stable'
            if lz < 0.55: return 'weakly_chaotic'
            if lz < 0.85: return 'chaotic'
            return 'hyperchaotic'
        if lam is not None and not np.isnan(lam):
            if lam < 0.15: return 'weakly_chaotic'
            if lam < 0.50: return 'chaotic'
            return 'hyperchaotic'
        return 'weakly_chaotic'


# ─────────────────────────────────────────────────────────────────
# 6. DELTA POR REGIMEN
# ─────────────────────────────────────────────────────────────────

class DeltaLibrary:
    TABLE = {
        'stable': 0.06,
        'weakly_chaotic': 0.05,
        'chaotic': 0.08,
        'hyperchaotic': 0.15,
        'noisy': 0.20,
    }

    def get(self, regime):
        return self.TABLE.get(regime, 0.10)


# ─────────────────────────────────────────────────────────────────
# 7. R3 DESCRIPTOR
# ─────────────────────────────────────────────────────────────────

class R3Descriptor:
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients(self, x, tau, m):
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
            if any(np.isnan(v) for v in [v0, vp, vm]):
                grads[k] = np.nan
            else:
                denom = np.sqrt(v0**2 + vp**2 + vm**2 + 1e-12) / np.sqrt(3)
                grads[k] = abs(vp - vm) / denom if denom > 1e-10 else abs(vp - vm)
        return grads, base

    def score(self, x, tau, m=3):
        grads, metrics = self._gradients(x, tau, m)
        lam = metrics.get('lambda', np.nan)
        lz = metrics.get('LZ', np.nan)
        d2 = metrics.get('D2', np.nan)
        regime = self.regime_detector.classify(lam, lz, d2)
        delta = self.delta_lib.get(regime)
        stability_map = {}
        stability_weights = []
        valid_n = 0
        for k, g in grads.items():
            if not np.isnan(g):
                valid_n += 1
                is_stable = g < delta
                weight = max(0.0, 1.0 - (g / delta)) if delta > 0 else 0.0
                stability_weights.append(weight)
                stability_map[k] = {
                    'gradient': float(g),
                    'stable': bool(is_stable),
                    'delta': float(delta),
                    'weight': float(weight),
                }
            else:
                stability_map[k] = {
                    'gradient': np.nan,
                    'stable': False,
                    'delta': float(delta),
                    'weight': 0.0,
                }
        r3_score = float(np.mean(stability_weights)) if stability_weights else 0.0
        coherent = (r3_score >= 0.60) and (regime != 'noisy')
        return {
            'R3_score': r3_score,
            'coherent': bool(coherent),
            'regime': regime,
            'regime_desc': RegimeDetector.DESCRIPTIONS.get(regime, regime),
            'delta': float(delta),
            'metrics': {k: (float(v) if not np.isnan(v) else None) for k, v in metrics.items()},
            'gradients': {k: (float(v) if not np.isnan(v) else None) for k, v in grads.items()},
            'stability_map': stability_map,
            'n_valid': int(valid_n),
        }


# ─────────────────────────────────────────────────────────────────
# 8. PIPELINE INTEGRADO
# ─────────────────────────────────────────────────────────────────

class AttractorPipeline:
    def __init__(self, m=3, max_tau=50, verbose=False):
        self.m = m
        self.verbose = verbose
        self._eps = DynamicEpsilon()
        self._tau = SemidynamicTau(max_lag=max_tau)
        self._r3 = R3Descriptor()
        self.results = {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self, x, label='serie'):
        x = np.asarray(x, dtype=float)
        if len(x) < 10:
            raise ValueError("La serie es demasiado corta (minimo 10 puntos).")
        x = (x - x.mean()) / (x.std() + 1e-12)

        self._tau.cache.clear()

        self._log(f"PIPELINE · {label} · N={len(x)}")

        tau0 = self._tau.compute(x, regime='unknown')
        metrics0 = Metrics.compute_all(x, tau0, self.m)
        regime0 = RegimeDetector().classify(
            metrics0.get('lambda'), metrics0.get('LZ'), metrics0.get('D2')
        )
        tau = self._tau.compute(x, regime=regime0)
        tau = max(1, tau)

        try:
            Y = embed(x, self.m, tau)
        except ValueError:
            tau = 1
            Y = embed(x, self.m, tau)

        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))

        metrics = Metrics.compute_all(x, tau, self.m)
        r3 = self._r3.score(x, tau, self.m)

        result = {
            'label': str(label),
            'x_normalized': x,
            'tau': int(tau),
            'tau_initial': int(tau0),
            'epsilon': eps_scalar,
            'epsilon_series': eps_series,
            'embedding': Y,
            'metrics': metrics,
            'regime': str(r3['regime']),
            'regime_desc': str(r3['regime_desc']),
            'R3': r3,
        }
        self.results[label] = result
        return result
