# Tasks: Structure Optimization

## Preparation

- [x] Create `autotrade/common/paths.py` to define directory constants. <!-- id: 0 -->
- [x] Ensure `outputs/models`, `outputs/backtests`, `outputs/logs`, `data/qlib_data` directories exist. <!-- id: 1 -->

## Moving Artifacts

- [x] Move all root `.html`, `.csv`, `.json` backtest results to `outputs/backtests/`. <!-- id: 2 -->
- [x] Move root `models/*` to `outputs/models/`. <!-- id: 3 -->
- [x] Move root `qlib_data/` to `data/qlib_data/` (ensure it exists first). <!-- id: 4 -->

## Consolidating Code

- [x] Move `autotrade/common/config/*` to `autotrade/config/` and update imports. <!-- id: 5 -->
- [x] Remove empty directory `autotrade/workflow/`. <!-- id: 6 -->
- [x] Remove empty directory `autotrade/features/alpha/`. <!-- id: 7 -->

## Updating Configuration and Paths

- [x] Update `autotrade/common/config/settings.py` (or the new `autotrade/config/settings.py`) to use the new paths. <!-- id: 8 -->
- [x] Update `scripts/*.py` to use new model/data paths. <!-- id: 9 -->
- [x] Update `autotrade/web_server.py` and `autotrade/backtest/engine.py` path references. <!-- id: 10 -->

## Validation

- [x] Run `uv run python scripts/train_model.py --help` (check if it fails due to paths). <!-- id: 11 -->
- [x] Start web server `uv run python autotrade/web_server.py` and check if UI loads correctly. <!-- id: 12 -->
