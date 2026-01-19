# Tasks: Replace Lumibot with VectorBT

- [x] Add `vectorbt` to `pyproject.toml` and remove `lumibot`. <!-- id: 0 -->
- [x] Create `autotrade/core/signal_generator.py` to encapsulate prediction logic independent of backtester. <!-- id: 1 -->
- [x] Create `autotrade/backtesting/engine.py` using `vectorbt`. <!-- id: 2 -->
- [x] Refactor `QlibMLStrategy` to use `SignalGenerator` for daily predictions. <!-- id: 3 -->
- [x] Update `TradeManager.run_backtest` to use `autotrade/backtesting/engine.py`. <!-- id: 4 -->
- [x] Verify daily prediction functionality (signals). <!-- id: 5 -->
- [x] Verify backtesting functionality (stats generation). <!-- id: 6 -->
- [x] Clean up strict `lumibot` imports and inheritance. <!-- id: 7 -->
