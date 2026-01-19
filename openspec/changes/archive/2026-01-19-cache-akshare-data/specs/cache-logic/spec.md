# Spec: Persistent Data Caching

## ADDED Requirements

### Requirement: Persistent Data Caching

The system MUST implement a persistent caching layer for `AKShareDataProvider` to store historical OHLCV data.
The cache MUST be stored in a local directory (e.g. `data/cache/akshare`) in a structured format (e.g. Parquet or CSV).
The cache MUST merge new fetched data with existing historical data to maintain a continuous record.

#### Scenario: Caching Historical Data

- **Given** a request for data for symbol "000001.SZ" from 2023-01-01 to 2023-12-31
- **When** the data is fetched for the first time
- **Then** the system MUST store the result in a local file "data/cache/akshare/000001.SZ.parquet"
- **And** a subsequent request for the same range MUST load data from the file instead of the network.

### Requirement: Efficient Trading Rule Checks

Methods that check for stock status (e.g. `is_suspended`, `is_limit_up`, `is_limit_down`) MUST prioritize using data from the local cache.
These methods MUST NOT trigger a separate network request if the required data date is already present in the cache.

#### Scenario: Status Check from Cache

- **Given** data for "600519.SH" for "2024-01-15" is already in the cache
- **When** `is_suspended("600519.SH", "2024-01-15")` is called
- **Then** the system MUST use the cached record to determine status
- **And** no network request to AKShare MUST be made.
