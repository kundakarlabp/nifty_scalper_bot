# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - YYYY-MM-DD

### Added
- Initial release after major refactoring.
- **Architecture**: Refactored monolithic `RealTimeTrader` into a modular, dependency-injected architecture (`Application`, `StrategyRunner`, `TradingSession`).
- **Configuration**: Replaced `.env` parsing with a validated Pydantic configuration system.
- **Strategy**: Corrected execution logic to use the intended score-based `generate_signal` method with both spot and option data.
- **Backtesting**: Added a new high-fidelity backtest engine (`tests/true_backtest_dynamic.py`) with 100% parity to live logic.
- **Risk Management**: Hardened all risk controls (Daily DD, 3-loss shutdown) and made them testable.
- **CI/CD**: Added GitHub Actions for `ruff`, `mypy`, and `pytest`.
- **Documentation**: Added `AUDIT.md`, `RUNBOOK.md`, and this `CHANGELOG.md`.

### Changed
- **Trading Logic**: The bot now uses a high-confidence, score-based strategy instead of the previous simple breakout logic.
- **Dependencies**: Consolidated web frameworks and added `pydantic` and `trading_calendars`.
- **Entry Point**: Simplified startup to a single `python3 -m src.main start` command.

### Fixed
- **Critical Flaw**: The bot no longer trades "blind" and uses the correct, intended strategy.
- **Market Hours**: The bot is now aware of market holidays and will not attempt to trade on closed days.
- Numerous bugs related to inconsistent configuration and untested logic.

### Removed
- `src/backtesting/backtest_engine.py`: Removed the old, flawed backtesting engine.
- `config.py`: Removed the empty root-level config file.
