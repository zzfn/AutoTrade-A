# Proposal: Persistent Data Caching for AKShare

## Problem

Currently, `AKShareDataProvider` fetches data from the network for every request, including individual checks for suspension and price limits. This leads to:

1. High latency during backtesting and signal generation.
2. Redundant network traffic.
3. Potential rate limiting from AKShare/EM data sources.

## Proposed Solution

Implement a persistent caching layer in `AKShareDataProvider` that stores fetched OHLCV data locally.

## Scope

- Modify `AKShareDataProvider` to use a file-based cache.
- Update data fetching methods to check cache first.
- Optimize trading rule checks to leverage cached data.

## Risks

- Cache staleness: Need a way to ensure the current day's data is updated after market close.
- Disk space: Many stocks over many years could grow, but OHLCV is relatively small.
