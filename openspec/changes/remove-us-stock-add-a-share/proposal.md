# Proposal: Remove US Stock Support and Enforce A-Share Only

## Background

The current codebase is a copy of a US-stock trading system based on LumiBot and Alpaca. The user wants to pivot this specific project (`AutoTrade-A`) to be an A-share (Chinese stock) only platform. The US stock functionality is no longer needed and should be removed to simplify the architecture.

## Objective

1.  Remove all US stock related code and dependencies (Alpaca, etc.).
2.  Make A-share (AKShare) the sole and default data provider.
3.  Ensure the system operates on a Daily frequency only.
4.  Update the frontend to show prediction signals based on the latest data immediately upon entry.

## Scope

### In Scope

- **Data Provider**: Remove `AlpacaDataProvider`. Promote `AKShareDataProvider` to default.
- **Dependencies**:
  - **Remove**: `alpaca-py` (direct dependency).
  - **Keep**: `lumibot`. It is essential for the **Backtesting Engine** and existing **Strategy Structure**. Rewriting the backtester locally is high-effort and risky; retaining LumiBot allows reusing the `YahooDataBacktesting` (adapted for A-share) and `Strategy` life-cycle.
- **Backend**: Update `TradeManager` and `web_server.py` to handle A-share symbols (6 digits) and daily frequency.
- **Frontend**: Update `index.html` to display prediction signals. Update `backtest.html` to remove US options.

### Out of Scope

- Implementing real-time automated trading execution for A-shares (usually requires complex external interface like QMT/PTrade). The focus is on "Prediction Signals" and "Backtest".
