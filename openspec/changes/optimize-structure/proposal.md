# Change: Optimize Project Architecture and Cleanup

## Why

The current project structure is cluttered and inconsistent:

- Root directory is filled with backtest result files (`.html`, `.csv`, `.json`).
- Model artifacts are stored in both root `models/` and `outputs/models/`.
- Several empty or near-empty directories exist (e.g., `autotrade/workflow/`, `autotrade/features/alpha/`).
- The directory structure doesn't clearly separate code from artifacts/data.

## What Changes

- **Consolidate Artifacts**: Move all generated content (models, backtest results, logs) into a unified `outputs/` directory.
- **Root Cleanup**: Move `qlib_data/` and `models/` into appropriate storage locations. Clean up the root directory of all non-configuration and non-source files.
- **Refine `autotrade/` package**:
  - Remove empty modules.
  - Flatten deeply nested configuration directories.
- **Standardize Paths**: Ensure all paths in the codebase use centralized configuration to avoid hardcoded relative paths that break when files are moved.

## Impact

- **File Locations**: Many files will be moved.
- **Code Changes**: Configuration loading logic needs to be updated to support the new paths.
- **Developer Experience**: A cleaner workspace and clearer separation of concerns.
