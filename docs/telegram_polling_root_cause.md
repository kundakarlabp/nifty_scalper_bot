# Telegram polling error root cause

The repeated alert text

```text
telegram.ext.Updater: default_error_callback - Exception happened while polling for updates.
```

is emitted by python-telegram-bot's Updater default polling error callback. It is not an order or strategy failure. The bot can still send outbound heartbeat messages while inbound command polling is failing or transiently retrying.

## Root cause

`TelegramBot.start()` registered an Application error handler, but `Application.add_error_handler()` does not handle transport errors raised by `Updater.get_updates()` during polling. Because `updater.start_polling()` was called without an explicit `error_callback`, PTB used its own `default_error_callback`, which logs a generic ERROR without the exception type. The alert aggregator then grouped that third-party generic ERROR into repeated aggregated alerts.

This hid the real distinction between:

- transient Telegram network/timeout errors, and
- true duplicate-poller conflicts where another process/VM/laptop is polling the same bot token.

## Fix

- Add a synchronous Updater polling error callback.
- Pass it to `updater.start_polling(error_callback=...)`.
- Classify duplicate polling conflicts as `TELEGRAM_POLLING_CONFLICT`.
- Classify timeouts/network errors as `telegram_polling_transient` at WARNING level.
- Keep generic Telegram polling failures visible as `TELEGRAM_UPDATER_ERROR`.
- Add a host-wide token lock to prevent same-host duplicate pollers, in addition to the previous process-local guard.
