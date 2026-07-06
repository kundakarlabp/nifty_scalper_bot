# Canonical time policy

This bot is operated for Indian markets. All bot-facing wall-clock timestamps should use `Asia/Kolkata` / `IST` / UTC+05:30.

Policy:

- Displayed timestamps, log timestamps, candle timestamps, tick timestamps, Telegram status, and persisted bot records should be IST.
- Naive timestamps from files or broker adapters should be interpreted as IST.
- Timezone-aware external timestamps should be converted to IST at ingestion.
- Duration clocks such as `time.monotonic()` remain monotonic clocks and are not wall-clock timestamps.
- External APIs may require their own protocol timezone, but bot-internal state should normalize back to IST immediately after ingestion.
