# Design: Persistent Data Caching for AKShare

## Architecture

The caching layer will be integrated into `AKShareDataProvider`. It will use a directory-based storage where each symbol has its own data file (Parquet format for efficiency).

### Cache Directory Structure

```
data/cache/akshare/
├── 000001.SZ.parquet
├── 600000.SH.parquet
└── ...
```

### Key Components

1. **Cache Manager internal to `AKShareDataProvider`**:
   - `_get_cached_data(symbol, start_date, end_date)`: Retrieves data from local storage.
   - `_save_to_cache(symbol, df)`: Merges new data with existing local data and saves it.
2. **Updated `fetch_data`**:
   - For each symbol:
     - Check what range is missing in the local cache compared to requested (start_date, end_date).
     - Fetch only missing pieces (or fetch all and merge if it's easier).
     - Return merged result.
3. **Optimized `is_suspended`, `is_limit_up`, etc.**:
   - These methods will now call `fetch_data` (which uses the cache) instead of `ak.stock_zh_a_hist` directly.

## Implementation Details

- **Storage Format**: `pandas.DataFrame.to_parquet` and `read_parquet` (requires `pyarrow` or `fastparquet`).
- **Concurrency**: Basic file locking or just relying on single-threaded updates (TradeManager usually runs synchronization in one thread).
- **Date Handling**: Current day data is tricky because it changes during the day. We might want to only cache "closed" days or have a TTL for today's data.
