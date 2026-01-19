# Design: AutoTrade-A Architecture

## Context

The system is an A-share stock selection and prediction platform using LightGBM for signals and VectorBT for backtesting. The goal is to provide daily buy/sell signals based on historical data.

## Goals

- **Separation of Concerns**: Clear distinction between data fetching, feature engineering, model training, prediction, and backtesting.
- **Reproducibility**: Experiments (notebooks) should rely on the same `autotrade` library code as production.
- **Scalability**: Easy to add new strategies or models.
- **Clarity**: Standardized directory structure.

## Directory Structure

```
AutoTrade-A/
├── artifacts/              # Generated files (ignored by git except placeholders)
│   ├── models/             # Trained model binaries
│   └── logs/               # Execution logs
├── autotrade/              # Source Package
│   ├── common/             # Utils, constants, logging config
│   ├── data/               # AKShare wrappers, data loaders
│   ├── features/           # Feature engineering logic
│   ├── models/             # Model wrappers (LightGBM, etc.)
│   ├── strategies/         # Strategy logic (Signal generation rules)
│   ├── backtest/           # VectorBT backtest runners and analysis
│   └── workflow/           # Workflows (training, daily prediction scripts)
├── configs/                # Configuration files (YAML)
├── notebooks/              # Jupyter notebooks for research
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts (not core app logic)
├── openspec/               # Specifications
├── main.py                 # CLI Entry point
├── pyproject.toml          # Dependencies (uv)
└── README.md
```

## Decisions

- **Unified Package**: All core logic in `autotrade/`.
- **Artifacts Folder**: Move `models/` (root) to `artifacts/models/` to avoid clutter.
- **Config Driven**: `configs/universe.yaml` and others control behavior.
- **VectorBT Integration**: Use VectorBT for fast, vectorized backtesting of signal-based strategies.

## Migration Plan

1. Move existing `autotrade/research` code to appropriate `autotrade/{data,features,models}` modules.
2. Move root `models/` to `artifacts/models/`.
3. Update `main.py` to use new paths.
