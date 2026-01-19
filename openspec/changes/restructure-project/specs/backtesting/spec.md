## ADDED Requirements

### Requirement: Historical Backtesting

The system SHALL verify strategy performance using historical data via VectorBT.

#### Scenario: Run Backtest

- **WHEN** user initiates a backtest for "StrategyA" from "2023-01-01" to "2023-12-31"
- **THEN** the system simulates trading using historical prices
- **AND** produces a performance report (CAGR, Max Drawdown)

### Requirement: Strategy Definition

The system SHALL allow defining trading strategies that utilize prediction signals.

#### Scenario: Signal Based Strategy

- **WHEN** a strategy is defined to buy top 5 stocks by prediction score
- **THEN** it executes buy orders for those stocks during the backtest
