#!/bin/bash

# Nifty Scalper Bot Runner Script
# This script provides easy commands to manage the trading bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_status "Copying .env.example to .env..."
        cp .env.example .env
        print_warning "Please edit .env file with your credentials before running the bot"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
}

# Run the bot locally
run_local() {
    check_env
    print_status "Starting Nifty Scalper Bot locally..."
    python nifty_scalper_bot.py
}

# Run with Docker
run_docker() {
    check_env
    print_status "Building and starting Docker containers..."
    docker-compose up --build
}

# Run Docker in background
run_docker_bg() {
    check_env
    print_status "Starting Docker containers in background..."
    docker-compose up -d --build
    print_success "Bot started in background"
    print_status "View logs with: ./run.sh logs"
}

# Stop Docker containers
stop_docker() {
    print_status "Stopping Docker containers..."
    docker-compose down
    print_success "Containers stopped"
}

# View Docker logs
view_logs() {
    print_status "Viewing bot logs..."
    docker-compose logs -f nifty-scalper-bot
}

# Clean up Docker
clean_docker() {
    print_status "Cleaning up Docker containers and images..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Docker cleanup completed"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python -m pytest tests/ -v
    print_success "Tests completed"
}

# Check bot health
health_check() {
    print_status "Checking bot health..."
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:10000/health || echo "000")
    
    if [ "$response" = "200" ]; then
        print_success "Bot is healthy and running"
    else
        print_error "Bot is not responding (HTTP $response)"
        exit 1
    fi
}

# Show bot status
show_status() {
    print_status "Fetching bot status..."
    curl -s http://localhost:10000/status | python -m json.tool
}

# Deploy to Render
deploy_render() {
    print_status "Preparing for Render deployment..."
    
    # Check if git repo is clean
    if [ -n "$(git status --porcelain)" ]; then
        print_warning "You have uncommitted changes. Please commit them first."
        git status --short
        exit 1
    fi
    
    print_status "Pushing to main branch..."
    git push origin main
    print_success "Code pushed to GitHub. Check Render dashboard for deployment status."
}

# Create backup
create_backup() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_dir="backups/backup_$timestamp"
    
    print_status "Creating backup..."
    mkdir -p "$backup_dir"
    
    # Copy important files (excluding .env for security)
    cp -r *.py requirements.txt Dockerfile docker-compose.yml README.md "$backup_dir/"
    
    # Copy logs if they exist
    if [ -d "logs" ]; then
        cp -r logs "$backup_dir/"
    fi
    
    print_success "Backup created at $backup_dir"
}

# Show help
show_help() {
    echo -e "${BLUE}Nifty Scalper Bot - Management Script${NC}"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install     - Install Python dependencies"
    echo "  local       - Run bot locally"
    echo "  docker      - Run with Docker (foreground)"
    echo "  start       - Run with Docker (background)"
    echo "  stop        - Stop Docker containers"
    echo "  logs        - View Docker logs"
    echo "  clean       - Clean up Docker containers and images"
    echo "  test        - Run tests"
    echo "  health      - Check bot health"
    echo "  status      - Show bot status"
    echo "  deploy      - Deploy to Render"
    echo "  backup      - Create backup"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh install    # Install dependencies"
    echo "  ./run.sh local      # Run locally"
    echo "  ./run.sh start      # Start with Docker"
    echo "  ./run.sh logs       # View logs"
    echo "  ./run.sh health     # Check if bot is running"
}

# Main script logic
case "${1:-help}" in
    install)
        install_deps
        ;;
    local)
        run_local
        ;;
    docker)
        run_docker
        ;;
    start)
        run_docker_bg
        ;;
    stop)
        stop_docker
        ;;
    logs)
        view_logs
        ;;
    clean)
        clean_docker
        ;;
    test)
        run_tests
        ;;
    health)
        health_check
        ;;
    status)
        show_status
        ;;
    deploy)
        deploy_render
        ;;
    backup)
        create_backup
        ;;
    help|*)
        show_help
        ;;
esac
