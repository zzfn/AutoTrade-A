
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

print(f"Project Root: {project_root}")

from autotrade.common.paths import PROJECT_ROOT
print(f"Detected PROJECT_ROOT: {PROJECT_ROOT}")

config_path = PROJECT_ROOT / "configs/universe.yaml"
print(f"Config Path: {config_path}")
if config_path.exists():
    with open(config_path) as f:
        print(f"Config Content:\n{f.read()}")
else:
    print("Config file does not exist!")

from autotrade.config.loader import ConfigLoader
try:
    loader = ConfigLoader()
    print(f"ConfigLoader symbols: {loader.symbols}")
except Exception as e:
    print(f"ConfigLoader failed: {e}")

from autotrade.trade_manager import TradeManager
tm = TradeManager()
print(f"TradeManager universe symbols: {tm._get_universe_symbols()}")

resolved = tm._resolve_symbols(tm._get_universe_symbols())
print(f"Resolved symbols count: {len(resolved)}")
print(f"Resolved symbols sample: {resolved[:5]}")
