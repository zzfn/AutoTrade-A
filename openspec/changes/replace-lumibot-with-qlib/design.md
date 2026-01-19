## Context

The project currently uses LumiBot for executing the ML strategy backtest. The user wants to switch to Qlib's built-in backtester (`PortAnalysisRecord`, `BacktestRecord`).

## Goals

- Run backtest using `qlib.workflow.backtest`.
- Maintain A-share specific logic (trading unit = 100, etc.) if possible, either via Qlib's `SimulatorExecutor` settings or custom Strategy logic.
- Reuse existing `QlibFeatureGenerator` and models.

## Decisions

- **Strategy Base Class**: Adopt `qlib.contrib.strategy.BaseStrategy`.
- **Execution Engine**: Use `qlib.backtest.executor.SimulatorExecutor`.
- **Configuration**:
  - Use `check_hold_limit=True` (if supported via config or custom validator) for limit up/down.
  - Use `deal_price='1d'` (or close price) as per current logic.
- **Data**: Continue using `QlibDataAdapter` to sync data to disk, then `qlib.init` to mount that data directory.

## Migration Plan

1.  Modify `QlibMLStrategy` to implement `generate_trade_decision(self, execute_result=None, **kwargs)`.
2.  In `TradeManager.run_backtest`:
    - Initialize Qlib with correct provider uri.
    - Generate predictions (if not already done) to get `pred_score` (pd.DataFrame).
    - Instantiate `QlibMLStrategy` (which will now wrap the predictions).
    - Run `backtest_loop` or `common_backtest`.
    - Parse results and generate report (Qlib generates `portfolio_analysis` etc., we might need to format it for the UI).

## Open Questions

- **Report Format**: LumiBot produces HTML tearsheets. Qlib produces Pandas DataFrames / Charts. We might need to generate a similar HTML report or just save the CSVs and update UI to read them. For MVP, we likely just save the CSVs and maybe generate a simple summary. _Update: User just asked to change backtester, UI adaptation might be outside immediate scope or minimal._ -> I will assume we should update `TradeManager` to capture the output similarly to how it did before (saving report path).
