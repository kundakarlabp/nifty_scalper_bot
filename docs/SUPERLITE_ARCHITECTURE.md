# Lightweight UI isolation

The admin controls and read-only review page run as separate processes from the trading engine. They share one bounded service unit and use the external instance environment without overwriting unspecified values.
