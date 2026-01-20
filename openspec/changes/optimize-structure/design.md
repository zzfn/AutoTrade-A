# Design: New Project Structure

## Proposed Directory Layout

```text
AutoTrade-A/
├── autotrade/              # Core Python package
│   ├── config/             # Configuration (consolidated)
│   ├── data/               # Data adapters and providers
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and management logic
│   ├── strategies/         # Strategy and signal generation
│   ├── backtest/           # Backtesting engine
│   ├── ui/                 # Web interface (templates/static)
│   └── web_server.py       # API and Web Server
├── data/                   # Data storage (local files)
│   └── qlib_data/          # Qlib binary data
├── outputs/                # All generated artifacts
│   ├── models/             # Trained model instances (.pkl, .json)
│   ├── backtests/          # Backtest results (HTML, CSV)
│   ├── logs/               # Application logs
│   └── reports/            # Quantitative reports
├── scripts/                # Utility scripts (init, train, etc.)
├── notebooks/              # Jupyter research notebooks
├── openspec/               # Specification and change tracking
├── pyproject.toml / uv.lock # Dependencies
└── README.md               # Documentation
```

## Key Mapping Changes

| Current Path                | New Path             | Reason                                    |
| --------------------------- | -------------------- | ----------------------------------------- |
| `models/` (root)            | `outputs/models/`    | Separation of logic (code) and artifacts. |
| `qlib_data/` (root)         | `data/qlib_data/`    | Grouping data storage.                    |
| `*.html` (root)             | `outputs/backtests/` | Cleanup root directory.                   |
| `*.csv` (root)              | `outputs/backtests/` | Cleanup root directory.                   |
| `autotrade/common/config/`  | `autotrade/config/`  | Flatten structure for easier imports.     |
| `autotrade/workflow/`       | (Remove)             | Empty and unused.                         |
| `autotrade/features/alpha/` | (Remove)             | Empty and unused.                         |

## Path Management Strategy

A centralized `PathManager` or similar utility in `autotrade/common/paths.py` will be used to resolve absolute paths based on the project root. This ensures consistency across scripts, the web server, and the core package.
