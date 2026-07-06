# Boot diagnostics

The service keeps live execution closed outside the exchange session. Repeated unchanged startup diagnostics are rate-controlled so logs stay readable while state changes still pass through immediately.

Covered diagnostic families:

- `CONTRACT_SSOT_*`
- `LIVE_UNIVERSE_BOOTSTRAP_STATUS`
- `SELECTED_OPTION_SUBSCRIPTION_STATE`
- closed-market runner diagnostics

The session adapter keeps selected-option quote and execution-bar details quiet outside the open session. It does not enable live trading earlier because the market-state blocker remains present.
