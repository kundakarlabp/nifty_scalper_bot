# Deployment & Monitoring Guide

This document summarises how to deploy the Nifty Scalper bot, run the new backtesting engine, and operate the accompanying monitoring stack.

## Backtesting

1. Ensure dependencies are installed (`pip install -r requirements.txt`).
2. Provide OHLCV data as a `pandas.DataFrame` indexed by `DatetimeIndex`.
3. Implement a strategy exposing `name` and `generate_signals(data)` returning {-1, 0, 1}.
4. Run the engine:

```python
from src.nifty_scalper_bot.backtesting import BacktestEngine, BacktestConfig

engine = BacktestEngine(data, strategy, BacktestConfig())
result = engine.run()
print(result.performance)
```

The engine writes interactive reports into `backtest_reports/` by default.

## Container deployment

```bash
./scripts/deploy_backtest.sh
```

This builds the Docker image and launches the compose stack (bot, Prometheus, Grafana).

## Systemd service

Copy `deploy/systemd/nifty-scalper.service` to `/etc/systemd/system/` and enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nifty-scalper.service
```

## Monitoring stack

1. Initialise monitoring directories and data volumes:

   ```bash
   ./scripts/setup_monitoring.sh
   ```

2. Access Prometheus at <http://localhost:9090> and Grafana at <http://localhost:3000> (default credentials `admin:admin`).

Prometheus uses alert rules defined in `ops/monitoring/alert_rules.yml`. Grafana automatically provisions the "Nifty Scalper Overview" dashboard.

## Backups

Run scheduled backups via:

```bash
./scripts/backup_data.sh
```

Old backups (>14 days) are automatically rotated.
