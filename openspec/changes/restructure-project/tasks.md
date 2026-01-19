## 1. Structure & Cleanup

- [x] 1.1 Create `artifacts/models` and `artifacts/logs` directories.
- [x] 1.2 Move existing root `models/` content to `artifacts/models/`.
- [x] 1.3 Create new package directories: `autotrade/{common,data,features,models,strategies,backtest,workflow}`.

## 2. Code Refactoring

- [x] 2.1 Refactor Data: Move AKShare code to `autotrade/data`.
- [x] 2.2 Refactor Features: Move feature generation to `autotrade/features`.
- [x] 2.3 Refactor Models: Wrap LightGBM logic in `autotrade/models`.
- [x] 2.4 Refactor Strategies: Implement strategy logic in `autotrade/strategies` independent of execution engine.
- [x] 2.5 Refactor Backtest: Create VectorBT backtest runner in `autotrade/backtest`.
- [x] 2.6 Create Entry Points: Update `main.py` to support `train`, `predict`, `backtest` commands.

## 3. Verification

- [x] 3.1 Verify data fetching works.
- [x] 3.2 Verify feature generation works.
- [x] 3.3 Verify model training and saving works.
- [x] 3.4 Verify backtesting runs.
