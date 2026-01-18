# Spec: A-Share Only Support

## REMOVED Requirements

### Requirement: US Stock Support

The system MUST NOT support US stock symbols or data providers (Alpaca).
The system MUST NOT require API keys for US brokers (Alpaca API Key/Secret).
The system MUST NOT support 'minute' or 'hour' data granularity (only Daily).

#### Scenario: No US Connection

- Given the system configuration
- When the system starts
- Then it does NOT attempt to connect to Alpaca
- And it does NOT log errors about missing Alpaca keys.

## ADDED Requirements

### Requirement: A-Share Data Provider

The system MUST use `AKShareDataProvider` as the primary and only data source.
The system MUST automatically handle A-share symbol formatting (e.g. adding `.SZ` or `.SH` suffix if missing, though 6 digits is standard user input).
The system MUST support retrieving Daily OHLCV data for A-shares.

#### Scenario: Fetch Data

- Given a list of A-share symbols (e.g. "000001", "600519") and a date range
- When data is requested
- Then the system returns Daily OHLCV DataFrame from AKShare
- And automatic suffix handling ensures valid requests.

### Requirement: Latest Prediction Display

The system MUST provide an API endpoint to retrieve the latest trading signals based on the most recent market data.
The Frontend Dashboard MUST display these signals immediately upon loading.

#### Scenario: Dashboard Load

- Given the backend is running and a strategy/model is configured
- When the user visits the root page (`/`)
- Then the dashboard fetches and displays "Latest Prediction Signals" for the monitored stock pool.
- And the signals are based on the latest available daily data from AKShare.

## MODIFIED Requirements

### Requirement: Backtesting Interface

The Backtest UI MUST NOT allow selection of "US" market.
The Backtest UI MUST default to "A-Share" market and "Daily" frequency.

#### Scenario: Backtest Configuration

- Given the user is on the Backtest page
- When the user looks at the configuration options
- Then the "Market" dropdown is either hidden (implying A-Share) or forced to "A-Share"
- And the "US" option is not available.
