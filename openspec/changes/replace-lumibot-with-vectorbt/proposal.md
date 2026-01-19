# Replace Lumibot with VectorBT

## Summary

Replace the `lumibot` backtesting framework with `vectorbt` to improve backtesting performance and align better with the ML-based strategy architecture.

## Motivation

- **Performance**: `vectorbt` is significantly faster for backtesting large datasets compared to event-driven frameworks like `lumibot`.
- **Alignment**: Our ML models generate predictions in a vectorized manner (DataFrame inputs). `vectorbt` naturally works with this data structure, whereas `lumibot` requires iterating row-by-row.
- **Simplification**: Since we do not perform live trading (only signaling), the complex event loop of `lumibot` is unnecessary overhead.

## Scope

- Remove `lumibot` dependency.
- Add `vectorbt` dependency.
- Re-implement backtesting logic using `vectorbt`.
- Refactor `QlibMLStrategy` to support vectorized signal generation for both backtesting and daily prediction.
