# WebSocket pong alert storm

## Symptom

Telegram repeatedly emitted `websocket_pong_timeout` warnings with changing `age=` values and then sent one-event aggregation summaries for the same persistent condition.

## Root cause

The log-to-alert bridge keyed warning records by logger, level, and function. It forwarded every `Condition met:` record to the Telegram queue, so dynamic numeric fields made a persistent watchdog condition generate a new queued event on every logging cycle.

## Correction

`AlertLogHandler` now extracts the semantic condition name from `Condition met: <event>` records, includes that event in the alert key, and applies a five-minute monotonic repeat window before queue insertion.

The first occurrence remains visible, distinct conditions from the same function remain independent, ordinary warnings are not suppressed, and application logging plus WebSocket reconnect behavior are unchanged.
