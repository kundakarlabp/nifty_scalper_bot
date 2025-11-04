#!/usr/bin/env bash
set -euo pipefail

log() {
    printf '%s\n' "$1"
}

log '=== Archiving orphaned files ==='

mkdir -p archive/scripts archive/src

# Archive orphaned scripts
for file in \
    scripts/true_backtest_dynamic.py \
    scripts/zerodha_token_cli.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/scripts/
            log "Moved $file to archive/scripts/"
        fi
    done

# Archive duplicate backtesting modules (keep main backtest engine)
for file in \
    src/backtesting/cli.py \
    src/backtesting/sim_connector.py \
    src/backtesting/synth.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

# Archive duplicate broker modules (kiteconnect is used)
for file in \
    src/brokers/ \
    src/broker/instruments.py \
    src/broker/interface.py
    do
        if [ -e "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

# Archive old strategies modules
for file in \
    src/strategies/patches.py \
    src/strategies/v1.py \
    src/strategies/warmup.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

# Archive duplicate execution modules
for file in \
    src/execution/broker_executor.py \
    src/execution/broker_retry.py \
    src/execution/order_executor.py \
    src/execution/order_state.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

# Archive duplicate diagnostics
for file in \
    src/diagnostics/checks.py \
    src/diagnostics/file_check.py \
    src/diagnostics/healthkit.py \
    src/diagnostics/metrics.py \
    src/diagnostics/registry.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

# Archive duplicate utils
for file in \
    src/utils/broker_errors.py \
    src/utils/circuit_breaker.py \
    src/utils/env.py \
    src/utils/events.py \
    src/utils/image_ops.py \
    src/utils/logger_setup.py \
    src/utils/logging_tools.py
    do
        if [ -f "$file" ]; then
            mv "$file" archive/src/
            log "Moved $file to archive/src/"
        fi
    done

log '✅ Orphaned files archived'
