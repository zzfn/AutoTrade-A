## ADDED Requirements

### Requirement: Standardized Directory Structure

The project SHALL follow a clear separation between source code, data storage, and generated artifacts.

#### Scenario: Verify Code Location

- **WHEN** searching for core Python logic
- **THEN** all files MUST be located within the `autotrade/` directory

#### Scenario: Verify Artifact Location

- **WHEN** searching for trained models or backtest results
- **THEN** all files MUST be located within the `outputs/` directory

#### Scenario: Verify Data Location

- **WHEN** searching for raw or processed datasets (e.g. qlib_data)
- **THEN** all files MUST be located within the `data/` directory

### Requirement: Centralized Path Management

The system SHALL use a centralized configuration for resolving file and directory paths.

#### Scenario: Resolve Output Path

- **WHEN** the system needs to save a backtest result
- **THEN** it MUST use the path configured in the central path manager rather than a hardcoded relative path
