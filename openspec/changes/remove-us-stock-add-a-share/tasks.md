# Tasks: Remove US Stock Support and Add A-Share Prediction

- [x] Remove Alpaca dependencies (`alpaca-py`) from `pyproject.toml` and run `uv lock`. <!-- id: 0 -->
- [x] Modify `autotrade/research/data/providers.py` to remove `AlpacaDataProvider` class. <!-- id: 1 -->
- [x] Modify `autotrade/research/data/providers.py` to update `DataProviderFactory` to remove US support and default to A-share (`AKShareDataProvider`). <!-- id: 2 -->
- [x] Modify `autotrade/trade_manager.py` to remove Alpaca initialization logic in `initialize_and_start`. <!-- id: 3 -->
- [x] Modify `autotrade/trade_manager.py` to implement `get_latest_predictions(symbols)` method using `QlibMLStrategy`. <!-- id: 4 -->
- [x] Modify `autotrade/web_server.py` to remove Alpaca monkey patch and add `/api/predict` endpoint. <!-- id: 5 -->
- [x] Modify `autotrade/ui/templates/index.html` to add a "Latest Prediction Signals" widget and fetch data from `/api/predict`. <!-- id: 6 -->
- [x] Modify `autotrade/ui/templates/backtest.html` to remove "Market" selection (default to A-share) and US-specific fields. <!-- id: 7 -->
- [x] Modify `autotrade/research/data/qlib_adapter.py` to default to `cn` market and handle A-share specifics if needed. <!-- id: 8 -->
- [x] Remove `US` references in `openspec/project.md`. <!-- id: 9 -->
