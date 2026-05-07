from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _slug(text: str) -> str:
    text = (text or '').strip().lower()
    text = re.sub(r'[^a-z0-9áéíóúñü\-\s_]', '', text)
    text = text.replace('ñ', 'n')
    text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ü', 'u')
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or 'senal'


def _safe_float(x: Any) -> Any:
    try:
        return float(x)
    except Exception:
        return x


@dataclass
class RunLabel:
    signal_type: str
    user_label: str = ''
    noise_level: str = ''
    sample_count: Optional[int] = None
    source: str = 'sintetica'
    tag: str = ''

    def build_prefix(self) -> str:
        parts = [
            _slug(self.source),
            _slug(self.signal_type),
            _slug(self.user_label),
            _slug(self.noise_level),
            f'n{self.sample_count}' if self.sample_count else '',
            _slug(self.tag),
        ]
        parts = [p for p in parts if p]
        return '_'.join(parts)


class RunExporter:
    def __init__(self, base_dir: str = 'outputs'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def create_run_dir(self, label: RunLabel) -> Path:
        run_name = f"{label.build_prefix()}_{self._timestamp()}"
        run_dir = self.base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_bundle(
        self,
        label: RunLabel,
        resultado: Dict[str, Any],
        baselines: Optional[Dict[str, Any]] = None,
        embedding_df: Optional[pd.DataFrame] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        figures: Optional[Iterable[tuple[str, Any]]] = None,
    ) -> Dict[str, str]:
        run_dir = self.create_run_dir(label)
        prefix = label.build_prefix()

        saved = {}

        if resultado:
            result_rows = [
                {'Variable': k, 'Valor': _safe_float(v), 'Referencia': '---'}
                for k, v in resultado.items()
            ]
            result_df = pd.DataFrame(result_rows)
            path = run_dir / f'{prefix}_resultados.csv'
            result_df.to_csv(path, index=False)
            saved['resultados_csv'] = str(path)

        if baselines:
            base_rows = [
                {'metrica': k, 'valor': _safe_float(v)}
                for k, v in baselines.items()
            ]
            base_df = pd.DataFrame(base_rows)
            path = run_dir / f'{prefix}_baselines.csv'
            base_df.to_csv(path, index=False)
            saved['baselines_csv'] = str(path)

        if embedding_df is not None and not embedding_df.empty:
            path = run_dir / f'{prefix}_embedding.csv'
            embedding_df.to_csv(path, index=False)
            saved['embedding_csv'] = str(path)

        meta = {
            'timestamp': datetime.now().isoformat(),
            'label': asdict(label),
            'extra_meta': extra_meta or {},
        }
        meta_path = run_dir / f'{prefix}_meta.json'
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        saved['meta_json'] = str(meta_path)

        if figures:
            for fig_name, fig_obj in figures:
                fig_slug = _slug(fig_name)
                fig_path = run_dir / f'{prefix}_{fig_slug}.png'
                try:
                    fig_obj.savefig(fig_path, dpi=180, bbox_inches='tight')
                    saved[f'fig_{fig_slug}'] = str(fig_path)
                except Exception:
                    pass

        return saved
