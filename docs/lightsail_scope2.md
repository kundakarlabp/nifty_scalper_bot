# AWS Lightsail Operational Scope

The production deployment authority is the Ubuntu AWS Lightsail instance managed by `systemd`.

This layer owns host provisioning, the `niftybot` service, host-local environment variables, HTTPS termination, validated updates, health verification and rollback. It does not own strategy decisions, order state, bracket state or broker reconciliation; those remain with the canonical application modules documented in `ARCHITECTURE_TRADING_PATH.md`.

Railway-related files may remain for compatibility or historical reference, but they are not used to operate the production instance.
