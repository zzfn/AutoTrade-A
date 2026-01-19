# Backtesting Engine Specs

## MODIFIED Requirements

### Requirement: Backtesting Framework

The system MUST use `vectorbt` vectorized engine for all backtesting operations.

#### Scenario: Run Backtest

- Given a date range and a list of symbols.
- When `run_backtest` is called.
- Then it should fetch historical data.
- And generate ML predictions for all dates (vectorized).
- And calculate portfolio performance using `vectorbt`.
- And return performance metrics (Sharpe, Return, Drawdown).

### Requirement: Strategy Structure

The strategy logic SHALL be implemented as a standalone class producing DataFrame signals, decoupled from any execution engine.

#### Scenario: Generate Daily Signals

- Given today's market data.
- When asking for latest predictions.
- Then it uses the shared `SignalGenerator` logic.
- And returns the top-k stocks to buy/sell.
