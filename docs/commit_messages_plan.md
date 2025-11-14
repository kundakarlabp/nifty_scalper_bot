1. fix(ws): production-grade Zerodha WS (state, queue flush, headers, IPv4, token sanitize)
2. build: pin kiteconnect and websocket-client versions
3. fix(telegram): make AIORateLimiter optional, no-crash startup
4. chore(entrypoint): unbuffered logging; main module wiring
5. feat(telegram): add /ws_status, /ws_reconnect, /tick, /subscribe, /unsubscribe
6. chore(ws): env token sanitization helper
7. chore(ws): header merge + Origin injection
8. chore(ws): IPv6 disabled for websocket-client
9. chore(test): add scripts/ws_smoke_test.py
10. chore(backtest): keep backtest entry intact
