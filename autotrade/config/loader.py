from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, config_path: str = "configs/universe.yaml"):
        # Resolve absolute path relative to project root if possible, or use as is
        from autotrade.common.paths import PROJECT_ROOT
        self.config_path = (PROJECT_ROOT / config_path).resolve()
        self._config = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    @property
    def symbols(self) -> list[str]:
        # Always reload to pick up changes
        self._reload_if_changed()
        return self._config.get("symbols", [])

    def _reload_if_changed(self):
        try:
            current_mtime = self.config_path.stat().st_mtime
            if not hasattr(self, "_last_mtime") or current_mtime > self._last_mtime:
                self._config = self._load()
                self._last_mtime = current_mtime
        except Exception:
            pass  # Keep old config if reload fails

    @property
    def timeframe(self) -> str:
        # A 股固定使用日线数据
        return "1d"

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


# Singleton instance for easy access
# Users can also instantiate their own if they need a different path
# but defaults are good for standard structure
try:
    default_config = ConfigLoader()
except Exception:
    # Allow import even if config is missing (e.g. during CI/tests setup)
    default_config = None
