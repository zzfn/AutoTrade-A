from pathlib import Path

# Project Root
# This file is in autotrade/common/paths.py
# So PROJECT_ROOT is ../../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Config Paths
CONFIG_DIR = PROJECT_ROOT / "autotrade" / "config"

# UI Paths
UI_DIR = PROJECT_ROOT / "autotrade" / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
