# Change: Replace LumiBot with Qlib Backtest Engine

## Why

Currently, the system uses LumiBot for backtesting `QlibMLStrategy`. While LumiBot is a good general-purpose trading bot framework, using Qlib's native backtesting engine offers better integration with Qlib's data layout, model predictions, and performance metrics (IC, ICIR, Sharpe). It eliminates the overhead of converting Qlib predictions to LumiBot signals and allows for more standard Quantitative workflow.

## What Changes

- **Refactor `QlibMLStrategy`**: Change base class from `lumibot.strategies.strategy.Strategy` to `qlib.contrib.strategy.BaseStrategy`.
- **Update `TradeManager`**: implementation of `run_backtest` will now use `qlib.workflow.backtest` instead of `lumibot`'s backtest loop.
- **Data Flow**: Ensure backtest uses Qlib's native data handling (likely via `QlibDataAdapter` ensuring data availability, then Qlib accessing it).
- **Remove Dependency**: Uninstall `lumibot` from `pyproject.toml` and remove all LumiBot-related imports and logic from the codebase.

## Impact

- **Affected Specs**: `qlib-ml-strategy` (Implementation details regarding LumiBot inheritance), `project-infrastructure` (Tech stack update).
- **Affected Code**: `autotrade/execution/strategies/qlib_ml_strategy.py`, `autotrade/trade_manager.py`, `pyproject.toml`, `uv.lock`.
- **Breaking**: Yes, `QlibMLStrategy` will no longer work with LumiBot runner. All LumiBot functionality will be removed.
