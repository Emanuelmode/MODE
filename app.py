import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

TAU_WINDOW = [-2, -1, 0, 1, 2]


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(f'Serie demasiado corta para m={m}, tau={tau}.')
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


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


class SemidynamicTau:
    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = max_lag
        self.bins = bins
        self._cache = {}

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
        ami_arr = np.array([self._ami(x, lag) for lag in range(1, max_l)])
        tau = 1
        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i - 1] and ami_arr[i] < ami_arr[i + 1]:
                tau = i + 1
                break
        else:
            threshold = ami_arr[0] / np.e if len(ami_arr) else 0
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break
        self._cache[regime] = tau
        return tau


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
            d_series = np.array([np.linalg.norm(Y[min(i + k, N - 1)] - Y[min(j + k, N - 1)]) for k in range(steps)])
            pos = d_series > 0
            if pos.sum() < 3:
                continue
            t = np.where(pos)[0]
            log_d = np.log(d_series[pos] / d[j])
            if len(t) > 1:
                slope = np.polyfit(t, log_d, 1)[0]
                divergences.append(slope)
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

    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: np.ndarray = None) -> float:
        seq = x
        if Y is not None and Y.shape[0] > 10:
            Yc = Y - Y.mean(axis=0)
            cov = np.cov(Yc.T)
            if np.ndim(cov) == 0:
                proj = Yc[:, 0]
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                proj = Yc @ eigvecs[:, -1]
            seq = proj
        binary = ''.join('1' if v > np.median(seq) else '0' for v in seq)
        n = len(binary)
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

    @staticmethod
    def transfer_entropy(x: np.ndarray, tau: int, bins: int = 8) -> float:
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

    @staticmethod
    def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
        try:
            N = len(x)
            r = r_ratio * np.std(x, ddof=1)
            if r == 0:
                return np.nan

            def _maxdist(x_i, x_j):
                return max(abs(ua - va) for ua, va in zip(x_i, x_j))

            def _phi(m_val):
                x_emb = [x[j:j + m_val] for j in range(N - m_val + 1)]
                C = [len([1 for x_j in x_emb if _maxdist(x_i, x_j) <= r]) for x_i in x_emb]
                return np.log(np.mean(C))

            return _phi(m) - _phi(m + 1)
        except Exception:
            return np.nan

    @classmethod
    def compute_all(cls, x: np.ndarray, tau: int, m: int = 3) -> dict:
        Y = embed(x, m, tau)
        return {
            'lambda': cls.lyapunov(x, tau, m),
            'D2': cls.correlation_dimension(Y),
            'LZ': cls.lempel_ziv(x, Y),
            'TE': cls.transfer_entropy(x, tau),
            'SampEn': cls.sample_entropy(x, m=2),
        }


class RegimeDetector:
    DESCRIPTIONS = {
        'stable': 'Estable / Periódico',
        'weakly_chaotic': 'Caos débil / Cuasiperiódico',
        'chaotic': 'Caótico',
        'hyperchaotic': 'Hipercaótico / Estructurado',
        'noisy': 'Ruido / Sin estructura dinámica',
    }

    def classify(self, lam: float, lz: float = None, d2: float = None) -> str:
        if not np.isnan(lam) and lam < 0:
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
            if lz < 0.25:
                return 'stable'
            if lz < 0.55:
                return 'weakly_chaotic'
            if lz < 0.85:
                return 'chaotic'
            return 'hyperchaotic'
        if not np.isnan(lam):
            if lam < 0.15:
                return 'weakly_chaotic'
            if lam < 0.50:
                return 'chaotic'
            return 'hyperchaotic'
        return 'weakly_chaotic'


class DeltaLibrary:
    TABLE = {
        'stable': 0.06,
        'weakly_chaotic': 0.05,
        'chaotic': 0.08,
        'hyperchaotic': 0.15,
        'noisy': 0.20,
    }

    def get(self, regime: str) -> float:
        return self.TABLE.get(regime, 0.10)


def tau_window_values(tau):
    return [max(1, tau + d) for d in TAU_WINDOW]


def delta_smooth(lam, d2, lz):
    lam_n = np.clip(abs(lam) / 0.01, 0, 1)
    d2_n = np.clip(d2 / 3.0, 0, 1) if np.isfinite(d2) else 0.5
    lz_n = np.clip(lz / 1.5, 0, 1) if np.isfinite(lz) else 0.5
    return float(0.04 + 0.12 * (0.45 * lam_n + 0.30 * d2_n + 0.25 * lz_n))


def soft_weight(g, delta):
    return float(np.exp(-g / max(delta, 1e-12)))


def r3_balanced(weights, valid_n, total_metrics=5):
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return 0.0
    return float(np.mean(weights) * (valid_n / total_metrics))


class R3Descriptor:
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients_window(self, x: np.ndarray, tau: int, m: int) -> tuple:
        taus = tau_window_values(tau)
        base = Metrics.compute_all(x, tau, m)
        metric_vals = {k: [] for k in base}
        for t in taus:
            vals = Metrics.compute_all(x, t, m)
            for k in base:
                metric_vals[k].append(vals.get(k, np.nan))
        grads = {}
        for k, vals in metric_vals.items():
            arr = np.asarray(vals, dtype=float)
            if np.all(np.isnan(arr)):
                grads[k] = np.nan
                continue
            v0 = base.get(k, np.nan)
            if np.isnan(v0):
                grads[k] = np.nan
                continue
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2:
                grads[k] = np.nan
                continue
            denom = np.sqrt(np.mean(arr ** 2) + v0 ** 2 + 1e-12)
            grads[k] = float(np.std(arr) / denom)
        return grads, base

    def score(self, x: np.ndarray, tau: int, m: int = 3) -> dict:
        grads, metrics = self._gradients_window(x, tau, m)
        lam = metrics.get('lambda', np.nan)
        lz = metrics.get('LZ', np.nan)
        d2 = metrics.get('D2', np.nan)
        regime = self.regime_detector.classify(lam, lz, d2)
        delta = delta_smooth(lam, d2, lz)
        stability_map = {}
        stability_weights = []
        valid_n = 0
        for k, g in grads.items():
            if not np.isnan(g):
                valid_n += 1
                weight = soft_weight(g, delta)
                stability_weights.append(weight)
                stability_map[k] = {'gradient': g, 'stable': g < delta, 'delta': delta, 'weight': weight}
        r3_score = r3_balanced(stability_weights, valid_n, total_metrics=5)
        coherent = (r3_score >= 0.60) and (regime != 'noisy')
        return {
            'R3_score': r3_score,
            'coherent': coherent,
            'regime': regime,
            'regime_desc': self.regime_detector.DESCRIPTIONS.get(regime, regime),
            'delta': delta,
            'metrics': {k: v for k, v in metrics.items()},
            'gradients': {k: v for k, v in grads.items()},
            'stability_map': stability_map,
            'n_valid': valid_n,
        }


class AttractorPipeline:
    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m = m
        self.verbose = verbose
        self._eps = DynamicEpsilon()
        self._tau = SemidynamicTau(max_lag=max_tau)
        self._r3 = R3Descriptor()
        self.results = {}

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
        result = {
            'label': label,
            'x_normalized': x,
            'tau': tau,
            'tau_initial': tau0,
            'epsilon': eps_scalar,
            'epsilon_series': eps_series,
            'embedding': Y,
            'metrics': metrics,
            'regime': r3['regime'],
            'regime_desc': r3['regime_desc'],
            'R3': r3,
        }
        self.results[label] = result
        return result
