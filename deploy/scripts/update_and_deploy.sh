#!/usr/bin/env bash
# Automated deployment and update script for Nifty Scalper Bot on Lightsail
# This script updates Kite API credentials and ensures the bot is running

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Nifty Scalper Bot - Update & Deploy${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Define paths
BOT_DIR="$HOME/nifty_scalper_bot"
ENV_FILE="$BOT_DIR/.env"
BACKUP_DIR="$BOT_DIR/backups"
LOG_DIR="$BOT_DIR/logs"

# Create directories if they don't exist
mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Navigate to bot directory
log "Navigating to bot directory..."
cd "$BOT_DIR" || { error "Bot directory not found. Please clone the repository first."; exit 1; }

# Backup current .env file
log "Creating backup of current .env file..."
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$BACKUP_DIR/.env.backup.$(date +%Y%m%d_%H%M%S)"
    log "Backup created successfully"
else
    warning ".env file not found, will create new one"
fi

# Pull latest code from GitHub
log "Pulling latest code from GitHub..."
if git pull origin main; then
    log "Code updated successfully"
else
    warning "Git pull failed or already up-to-date"
fi

# Update Kite API credentials
log "Updating Kite API credentials..."

# Create or update .env file with new credentials
if [ -f "$ENV_FILE" ]; then
    # Update existing credentials
    sed -i 's/^KITE_ACCESS_TOKEN=.*/KITE_ACCESS_TOKEN=cezgmttnTDjs1ENmaBbIKE3MB9q2NECs/' "$ENV_FILE"
    sed -i 's/^KITE_API_KEY=.*/KITE_API_KEY=0gcfl5dfdr9rjfhy/' "$ENV_FILE"
    sed -i 's/^KITE_API_SECRET=.*/KITE_API_SECRET=26i06t29jnv7m6eofvpftjh55xw1cr1s/' "$ENV_FILE"
    log "Credentials updated in existing .env file"
else
    error ".env file not found after pulling code"
    exit 1
fi

# Verify credentials were updated
log "Verifying credential updates..."
if grep -q "KITE_ACCESS_TOKEN=cezgmttnTDjs1ENmaBbIKE3MB9q2NECs" "$ENV_FILE" && \
   grep -q "KITE_API_KEY=0gcfl5dfdr9rjfhy" "$ENV_FILE" && \
   grep -q "KITE_API_SECRET=26i06t29jnv7m6eofvpftjh55xw1cr1s" "$ENV_FILE"; then
    log "✓ All credentials verified successfully"
else
    error "Credential verification failed"
    exit 1
fi

# Install/update dependencies
log "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --quiet || warning "Some dependencies may have failed to install"
    log "Dependencies installed"
else
    warning "requirements.txt not found"
fi

# Check if systemd service exists
SERVICE_NAME="nifty-scalper-bot"
if systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
    log "Systemd service found, managing via systemctl..."
    
    # Reload systemd daemon
    sudo systemctl daemon-reload
    
    # Restart the service
    log "Restarting bot service..."
    if sudo systemctl restart "$SERVICE_NAME"; then
        log "✓ Service restarted successfully"
    else
        error "Failed to restart service"
        exit 1
    fi
    
    # Wait a moment for service to start
    sleep 3
    
    # Check service status
    log "Checking service status..."
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        log "✓ Bot is running via systemd"
        echo ""
        log "Service Status:"
        sudo systemctl status "$SERVICE_NAME" --no-pager -l
    else
        error "Bot service is not running"
        sudo systemctl status "$SERVICE_NAME" --no-pager -l
        exit 1
    fi
else
    log "No systemd service found, checking for running processes..."
    
    # Kill any existing bot processes
    if pgrep -f "nifty_scalper_bot" > /dev/null; then
        log "Stopping existing bot processes..."
        pkill -f "nifty_scalper_bot"
        sleep 2
    fi
    
    # Start bot manually
    log "Starting bot manually..."
    cd "$BOT_DIR"
    nohup python3 -m nifty_scalper_bot.app > "$LOG_DIR/bot.log" 2>&1 &
    BOT_PID=$!
    log "Bot started with PID: $BOT_PID"
    
    # Wait and verify
    sleep 3
    if ps -p $BOT_PID > /dev/null; then
        log "✓ Bot is running (PID: $BOT_PID)"
    else
        error "Bot failed to start"
        tail -n 50 "$LOG_DIR/bot.log"
        exit 1
    fi
fi

# Show recent logs
log ""
log "Recent bot logs:"
echo -e "${YELLOW}----------------------------------------${NC}"
if [ -f "$LOG_DIR/nifty_scalper_bot.log" ]; then
    tail -n 20 "$LOG_DIR/nifty_scalper_bot.log"
elif [ -f "$LOG_DIR/bot.log" ]; then
    tail -n 20 "$LOG_DIR/bot.log"
else
    # Check journalctl if using systemd
    if systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager
    else
        warning "No log files found"
    fi
fi
echo -e "${YELLOW}----------------------------------------${NC}"

echo ""
log "${GREEN}========================================${NC}"
log "${GREEN}Deployment Complete!${NC}"
log "${GREEN}========================================${NC}"
echo ""
log "Bot Status: ${GREEN}RUNNING${NC}"
log "Configuration: Updated with new Kite credentials"
log "To view logs: tail -f $LOG_DIR/nifty_scalper_bot.log"
log "To check status: sudo systemctl status $SERVICE_NAME"
echo ""
