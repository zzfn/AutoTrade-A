# Tasks: Persistent Data Caching for AKShare

## Phase 1: Infrastructure

- [x] Install dependencies (`pyarrow`) if not present.
- [x] Create `autotrade/research/data/cache.py` (optional, or just put it in `providers.py`).
- [x] Define the cache directory path in a configurable way.

## Phase 2: Implementation in AKShareDataProvider

- [x] Implement `_load_cache(symbol)` and `_save_cache(symbol, df)`.
- [x] Refactor `fetch_data` to use cache:
  - [x] Logic to identify missing date ranges.
  - [x] Logic to fetch and merge data.
  - [x] Special handling for today's data (ensure it's not cached permanently until market close).
- [x] Refactor `is_suspended`, `is_limit_up`, `is_limit_down` to call `self.fetch_data`.
- [x] Add a way to clear/bypass cache (e.g. `force_update=True`).

## Phase 3: Integration and Testing

- [x] Verify `TradeManager.get_latest_predictions` works faster after first run.
- [x] Verify `QlibDataAdapter` still functions correctly.
- [x] Add unit tests for the caching logic.
