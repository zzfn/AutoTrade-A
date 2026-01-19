# Change: Restructure Project and Define Capabilities

## Why

The current project structure allows model artifacts at the root (`models/`) and lacks a clear separation between research (notebooks), core logic (signals), and the backtesting engine. The user requested a redesign of the directory and code structure to support an A-share signal prediction and stock selection system with backtesting capabilities (using VectorBT).

## What Changes

- **Directory Structure Reorganization**:
  - Move model artifacts to `artifacts/models/`.
  - Consolidate source code within `autotrade/`.
  - Establish clear modules for `data`, `features`, `models`, `strategies`, and `backtesting`.
  - Ensure `configs/` is properly used.
- **Formalize Capabilities**:
  - Define `prediction` capability (daily signal generation).
  - Define `backtesting` capability (strategy validation via VectorBT).

## Impact

- **Affected Specs**: `prediction`, `backtesting`.
- **Affected Code**:
  - `autotrade/` package structure.
  - `main.py` entry point.
  - Model loading/saving paths.
