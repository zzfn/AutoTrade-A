# Design: VectorBT Integration

## Architecture Changes

### 1. Strategy Decoupling

Currently, `QlibMLStrategy` mixes signal generation and execution logic within the `lumibot` event loop.
We will separate this into:

- **SignalGenerator**: Responsible for fetching data, generating features, and running the ML model to produce target positions/signals. This component will return a DataFrame of signals.
- **BacktestEngine**: Uses `vectorbt` to take the signals from `SignalGenerator` and simulate performance.
- **TradeManager Integration**: `TradeManager` will coordinate the two. For daily predictions, it calls `SignalGenerator`. For backtests, it calls `SignalGenerator` then `BacktestEngine`.

### 2. Backtest Implementation

- Use `vectorbt.Portfolio.from_signals` or `from_orders` to simulate trades.
- Inputs: Close prices (DataFrames), Signal/Entries/Exits (DataFrames).
- Outputs: Stats (Sharpe, Returns, Max Drawdown), Plots.

### 3. Data Flow

- **Data Source**: Qlib/AKShare (unchanged).
- **Preprocessing**: `QlibFeatureGenerator` (unchanged).
- **Modeling**: LightGBM (unchanged).
- **Execution**: `vectorbt` (New).

## Trade-offs

- **Pros**: Much faster backtests, cleaner code for ML strategies.
- **Cons**: `vectorbt` steep learning curve (but standard usage is simple); less flexibility for complex order types (e.g. trailing stops) compared to event-driven, but sufficient for our "signal" use case.

## Data Structures

- **Signals**: DataFrame with index=Date, columns=Symbols, values=Target Position (or Signal -1/0/1).
