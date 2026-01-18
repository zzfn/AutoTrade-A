# Design: A-Share Only Architecture

## Context

The project is transitioning from a US-stock auto-trading bot (LumiBot + Alpaca) to an A-share prediction and backtesting system.

## Architectural Changes

### 1. Data Provider

- **Current**: `DataProviderFactory` supports `us` (Alpaca) and `cn` (AKShare).
- **New**: `DataProviderFactory` only returns `AKShareDataProvider`.
- **Data Frequency**: Enforced to Daily (`1d`).

### 2. Strategy & Backtesting

- Keep the `Strategy` class interface (LumiBot style) to reuse existing strategy logic for backtesting.
- Backtesting engine continues to run on historical data from AKShare.

### 3. Prediction Mode (New)

- **Goal**: "Enter frontend page -> Show predicted signals".
- **Mechanism**:
  1.  When the dashboard loads (or via explicit API call on load), the backend triggers a "Prediction Run".
  2.  The backend fetches the latest `N` days of data (enough for the model's lookback window) from AKShare for the configured pool of stocks.
  3.  The model/strategy `on_trading_iteration` (or a specific `predict` method) is invoked with this data.
  4.  The output (Signal: Buy/Sell/Hold, Confidence, etc.) is returned to the frontend.
  5.  The frontend displays these signals in a "Latest Signals" card.

### 4. Frontend

- **Dashboard**: Add "Latest Prediction Signals" widget. Remove connection status to Alpaca. Show "Market Data Date" (latest available data date).
- **Backtest**: Remove "Market" dropdown (default to A-share). Remove US-specific params (e.g. "After Hours").

### 5. Backend Services

- `TradeManager`:
  - Remove Alpaca connection logic (`initialize_and_start` needs change).
  - Add `get_latest_predictions()` method.
  - Ensure `run_backtest` forces A-share provider.
