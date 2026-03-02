"""Shared config loader."""
import json
from pathlib import Path
from typing import Optional


def load_config(config_path: Optional[Path] = None) -> dict:
    """config 로드 (json 우선, yaml fallback)."""
    if config_path is None:
        base = Path(__file__).resolve().parent.parent.parent / "config"
        for name in ("feature_selection_config.json", "feature_selection_config.yaml", "feature_selection_config.yml"):
            p = base / name
            if p.exists():
                config_path = p
                break
        else:
            return {}
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix == ".json":
                return json.load(f)
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                return {}
    return {}
