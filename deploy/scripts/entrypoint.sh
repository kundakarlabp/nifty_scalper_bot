#!/usr/bin/env bash

################################################################################
# Nifty Scalper Bot - Production Deployment Entrypoint Script
# Purpose: Initialize environment, log deployment metrics, and start application
# Author: kundakarlabhp
# Last Updated: 2026-02-04
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

################################################################################
# CONFIGURATION & CONSTANTS
################################################################################

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly APP_DIR="${APP_DIR:-.}"
readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"
readonly STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-60}"

# Color codes for log output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'  # No Color

# Application settings
readonly APP_MODULE="nifty_scalper_bot.deployment_main:app"
readonly APP_PORT="${PORT:-8080}"
readonly APP_HOST="0.0.0.0"
readonly WORKERS="${WORKERS:-1}"
readonly LOG_DIR="${LOG_DIR:-/tmp/nifty_scalper_logs}"

# PID tracking
APP_PID=""
CHILD_PIDS=()

################################################################################
# LOGGING FUNCTIONS
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    
    case "$level" in
        INFO)
            echo -e "${BLUE}[${timestamp}] [INFO]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] [SUCCESS]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[${timestamp}] [WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] [ERROR]${NC} $message" >&2
            ;;
        DEBUG)
            if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
                echo -e "${BLUE}[${timestamp}] [DEBUG]${NC} $message"
            fi
            ;;
        *)
            echo "[${timestamp}] $message"
            ;;
    esac
}

################################################################################
# SIGNAL HANDLERS & CLEANUP
################################################################################

cleanup() {
    local exit_code=$?
    
    log INFO "Received shutdown signal, initiating graceful shutdown..."
    
    if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
        log INFO "Sending SIGTERM to application process [PID: $APP_PID]"
        kill -TERM "$APP_PID" 2>/dev/null || true
        
        # Wait with timeout
        local timeout=$STARTUP_TIMEOUT
        while kill -0 "$APP_PID" 2>/dev/null && [[ $timeout -gt 0 ]]; do
            sleep 1
            ((timeout--))
        done
        
        # Force kill if still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            log WARN "Process did not exit gracefully, sending SIGKILL"
            kill -9 "$APP_PID" 2>/dev/null || true
        fi
    fi
    
    log INFO "Shutdown complete. Exit code: $exit_code"
    exit $exit_code
}

handle_error() {
    local line_num="$1"
    local error_code="$2"
    log ERROR "Error occurred in script at line $line_num with exit code $error_code"
    cleanup
}

# Register signal handlers
trap cleanup SIGTERM SIGINT
trap 'handle_error ${LINENO} $?' ERR

################################################################################
# ENVIRONMENT VALIDATION
################################################################################

validate_environment() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Validating deployment environment..."
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check required commands
    local required_commands=("python3" "bash" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            log ERROR "Required command not found: $cmd"
            return 1
        fi
    done
    log SUCCESS "All required commands available"
    
    # Validate Python version
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log INFO "Python version: $python_version"
    
    # Check if running in Railway environment
    if [[ -n "${RAILWAY_ENVIRONMENT_NAME:-}" ]]; then
        log SUCCESS "Running in Railway environment: $RAILWAY_ENVIRONMENT_NAME"
    else
        log WARN "Not running in Railway environment"
    fi
    
    return 0
}

################################################################################
# SYSTEM INFORMATION LOGGING
################################################################################

log_system_info() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "System Information"
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    log INFO "Hostname: $(hostname)"
    log INFO "Current user: $(whoami)"
    log INFO "Working directory: $(pwd)"
    log INFO "Script directory: $SCRIPT_DIR"
    log INFO "App directory: $APP_DIR"
    
    log INFO "────── Filesystem ──────"
    log INFO "Disk usage: $(df -h / | awk 'NR==2 {print $5 " used of " $2}')"
    log INFO "Available space: $(df -h / | awk 'NR==2 {print $4}')"
    
    log INFO "────── Memory ──────"
    local mem_info=$(free -h | awk 'NR==2 {print $3 " / " $2}')
    log INFO "Memory usage: $mem_info"
    
    log INFO "────── CPU ──────"
    log INFO "CPU count: $(nproc)"
    log INFO "Load average: $(uptime | awk -F'load average:' '{print $2}')"
    
    log INFO "────── Python Environment ──────"
    log INFO "Python executable: $(which python3)"
    log INFO "Python path: $PYTHONPATH"
    if [[ -f "$APP_DIR/requirements.txt" ]]; then
        log INFO "Installed packages: $(pip3 list --format=count 2>/dev/null || echo 'N/A')"
    fi
    
    log INFO "────── Network ──────"
    log INFO "Default gateway: $(ip route | grep default | awk '{print $3}' | head -1)"
    log INFO "Hostname resolution: $(nslookup localhost 2>/dev/null | grep -i address | head -1 || echo 'N/A')"
}

################################################################################
# DEPENDENCY CHECK
################################################################################

check_dependencies() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Checking Python dependencies..."
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ ! -f "$APP_DIR/requirements.txt" ]]; then
        log WARN "requirements.txt not found at $APP_DIR/requirements.txt"
        return 0
    fi
    
    # Check if pip packages are installed
    if ! python3 -c "import importlib; import sys; packages = [line.split('==')[0].split('[')[0] for line in open('requirements.txt').readlines() if line.strip() and not line.startswith('#')]; missing = [p for p in packages if not importlib.util.find_spec(p)]; sys.exit(len(missing))" 2>/dev/null; then
        log WARN "Some dependencies may be missing, but continuing..."
    else
        log SUCCESS "All required dependencies are installed"
    fi
}

################################################################################
# LOG DIRECTORY SETUP
################################################################################

setup_log_directory() {
    log INFO "Setting up log directory at: $LOG_DIR"
    
    mkdir -p "$LOG_DIR"
    
    if [[ ! -w "$LOG_DIR" ]]; then
        log ERROR "Cannot write to log directory: $LOG_DIR"
        return 1
    fi
    
    log SUCCESS "Log directory ready: $LOG_DIR"
}

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

preflight_checks() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Running pre-flight checks..."
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check port availability
    if netstat -tuln 2>/dev/null | grep -q ":$APP_PORT "; then
        log ERROR "Port $APP_PORT is already in use"
        return 1
    fi
    log SUCCESS "Port $APP_PORT is available"
    
    # Check if main app module exists
    if [[ ! -d "$(echo $APP_MODULE | sed 's/\.app$//')" ]]; then
        log WARN "App module directory might not exist: $(echo $APP_MODULE | sed 's/\.app$//')"
    fi
    
    # Check required environment variables
    local required_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log WARN "Optional environment variable not set: $var"
        fi
    done
    
    log SUCCESS "Pre-flight checks completed"
}

################################################################################
# APPLICATION STARTUP
################################################################################

start_application() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Starting Nifty Scalper Bot Application..."
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    log INFO "Application Configuration:"
    log INFO "  Module: $APP_MODULE"
    log INFO "  Host: $APP_HOST"
    log INFO "  Port: $APP_PORT"
    log INFO "  Workers: $WORKERS"
    log INFO "  Log Level: $LOG_LEVEL"
    log INFO ""
    
    log INFO "Starting application process..."
    
    # Start the application with proper output handling
    if python3 -m uvicorn \
        "$APP_MODULE" \
        --host "$APP_HOST" \
        --port "$APP_PORT" \
        --workers "$WORKERS" \
        --access-log \
        --log-level "${LOG_LEVEL,,}" \
        2>&1 &
    then
        APP_PID=$!
        log SUCCESS "Application started with PID: $APP_PID"
        CHILD_PIDS+=($APP_PID)
        
        sleep 2  # Give the app a moment to initialize
        
        # Verify process is still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            log SUCCESS "Application process verified as running"
            return 0
        else
            log ERROR "Application process exited immediately after startup"
            return 1
        fi
    else
        log ERROR "Failed to start application"
        return 1
    fi
}

################################################################################
# HEALTH CHECK
################################################################################

health_check() {
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log INFO "Performing health checks..."
    log INFO "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local max_attempts=10
    local attempt=1
    local wait_time=2
    
    while [[ $attempt -le $max_attempts ]]; do
        log INFO "Health check attempt $attempt/$max_attempts..."
        
        if curl -sf "http://localhost:$APP_PORT/readyz" >/dev/null 2>&1 || \
           curl -sf "http://localhost:$APP_PORT/" >/dev/null 2>&1; then
            log SUCCESS "Application is responding to HTTP requests"
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            log INFO "Waiting ${wait_time}s before next attempt..."
            sleep "$wait_time"
        fi
        
        ((attempt++))
    done
    
    log WARN "Health check failed after $max_attempts attempts, but continuing..."
    return 0
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    log INFO "╔════════════════════════════════════════════════════════════╗"
    log INFO "║     Nifty Scalper Bot - Railway Deployment Started        ║"
    log INFO "╚════════════════════════════════════════════════════════════╝"
    log INFO ""
    
    # Execute startup sequence
    validate_environment || exit 1
    log_system_info
    check_dependencies
    setup_log_directory || exit 1
    preflight_checks || exit 1
    start_application || exit 1
    health_check || true
    
    log INFO ""
    log SUCCESS "╔════════════════════════════════════════════════════════════╗"
    log SUCCESS "║          Nifty Scalper Bot is READY for operations        ║"
    log SUCCESS "╚════════════════════════════════════════════════════════════╝
