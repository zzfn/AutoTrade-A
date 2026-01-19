## 1. Implementation

- [ ] 1.1 Refactor `autotrade/execution/strategies/qlib_ml_strategy.py` to inherit `qlib.contrib.strategy.BaseStrategy`.
- [ ] 1.2 Implement `generate_trade_decision` method in `QlibMLStrategy` to return `Order` objects.
- [ ] 1.3 Update `TradeManager.run_backtest` in `autotrade/trade_manager.py` to use `qlib.workflow.backtest`.
- [ ] 1.4 Ensure Qlib initialization (`qlib.init`) is handled correctly in the backtest process.
- [ ] 1.5 Verify that A-share specific implementation (lots, limits) is preserved or handled by Executor.
- [ ] 1.6 Remove `lumibot` from `pyproject.toml` and update lockfile.
- [ ] 1.7 Clean up `TradeManager` (remove `YahooDataBacktesting`, `start_strategy` thread wrapper, and other LumiBot artifacts).
- [ ] 1.8 Verify backtest functionality by running a small backtest.
- [ ] 1.9 Update `openspec/project.md` to reflect the change in backtesting engine and removal of LumiBot.
