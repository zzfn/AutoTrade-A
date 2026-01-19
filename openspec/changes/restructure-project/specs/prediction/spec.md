## ADDED Requirements

### Requirement: Daily Stock Prediction

The system SHALL generate daily buy/sell signals for a given universe of stocks using trained models.
**Inputs**: Stock list (universe), date.
**Outputs**: Prediction scores/signals.

#### Scenario: Generate Daily Signals

- **WHEN** user runs predictions for actual date "2024-01-01"
- **THEN** the system fetches valid data for the date
- **AND** loads the latest trained model
- **AND** outputs prediction scores for each stock in the universe

### Requirement: Model Management

The system SHALL support training and saving models for future predictions.

#### Scenario: Train Model

- **WHEN** user initiates model training
- **THEN** signals are generated from historical data
- **AND** model is trained and saved to `artifacts/models/`
