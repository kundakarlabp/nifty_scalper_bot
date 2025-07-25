#!/usr/bin/env python3
"""
Bot Monitoring Script
Monitors the trading bot and sends alerts if issues are detected
"""

import requests
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitor.log')
    ]
)
logger = logging.getLogger(__name__)

class Monitor:
    """Monitor the trading bot and send alerts"""
    
    def __init__(self, bot_url: str = "http://localhost:10000"):
        self.bot_url = bot_url
        self.last_status = None
        self.alert_threshold = 300  # 5 minutes
        self.last_alert_time = None

    def update_metrics(self, metrics_data):
    """Update system metrics (for compatibility with main bot)"""
    try:
        # Log key metrics for monitoring
        if 'current_price' in metrics_data:
            logger.info(f"Price: ₹{metrics_data['current_price']:.2f}")
        
        if 'pnl' in metrics_data:
            logger.info(f"P&L: ₹{metrics_data['pnl']:.2f}")
        
        if 'balance' in metrics_data:
            logger.info(f"Balance: ₹{metrics_data['balance']:.2f}")
            
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")
        
    def check_health(self) -> Dict[str, Any]:
        """Check bot health status"""
        try:
            response = requests.get(f"{self.bot_url}/health", timeout=10)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'data': response.json(),
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}",
                    'timestamp': datetime.now()
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def check_trading_status(self) -> Dict[str, Any]:
        """Check detailed trading status"""
        try:
            response = requests.get(f"{self.bot_url}/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking trading status: {e}")
            return None
    
    def check_for_issues(self, status: Dict[str, Any]) -> list:
        """Check for potential issues in bot status"""
        issues = []
        
        if status['status'] != 'healthy':
            issues.append(f"Bot unhealthy: {status.get('error', 'Unknown error')}")
            return issues
        
        # Get detailed status
        trading_status = self.check_trading_status()
        if not trading_status:
            issues.append("Unable to fetch trading status")
            return issues
        
        # Check for circuit breaker
        if trading_status.get('circuit_breaker', False):
            issues.append("Circuit breaker is active")
        
        # Check for significant losses
        daily_pnl = trading_status.get('daily_pnl', 0)
        if daily_pnl < -5000:  # Alert if daily loss > 5000
            issues.append(f"High daily loss: ₹{daily_pnl:.2f}")
        
        # Check if auto_trade is disabled
        if not trading_status.get('auto_trade', True):
            issues.append("Auto-trading is disabled")
        
        # Check for stuck positions (position open for >2 hours)
        current_position = trading_status.get('current_position')
        if current_position:
            # Note: This would require position entry time to be included in status
            pass
        
        return issues
    
    def send_alert(self, message: str):
        """Send alert (currently logs, can be extended for email/SMS)"""
        now = datetime.now()
        
        # Rate limiting - don't send alerts too frequently
        if (self.last_alert_time and 
            now - self.last_alert_time < timedelta(minutes=30)):
            return
        
        logger.warning(f"ALERT: {message}")
        self.last_alert_time = now
        
        # TODO: Implement email/SMS alerts
        # self.send_email_alert(message)
        # self.send_telegram_alert(message)
    
    def send_email_alert(self, message: str):
        """Send email alert (implement if needed)"""
        # Email configuration would go here
        pass

    def get_status(self):
    """Get current monitor status"""
    return {
        'last_check': self.last_status,
        'status': 'active',
        'last_alert': self.last_alert_time
    }
    
    def monitor_loop(self, check_interval: int = 60):
        """Main monitoring loop"""
        logger.info(f"Starting bot monitor - checking every {check_interval}s")
        
        while True:
            try:
                # Check bot health
                health_status = self.check_health()
                
                # Check for issues
                issues = self.check_for_issues(health_status)
                
                if issues:
                    alert_message = f"Bot Issues Detected:\n" + "\n".join(f"- {issue}" for issue in issues)
                    self.send_alert(alert_message)
                else:
                    logger.info("Bot status: OK")
                
                # Store last status
                self.last_status = health_status
                
                # Wait before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(30)  # Wait 30s before retrying

class PerformanceTracker:
    """Track bot performance metrics"""
    
    def __init__(self, bot_url: str = "http://localhost:10000"):
        self.bot_url = bot_url
        self.performance_log = []
    
    def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect performance metrics"""
        try:
            # Get trading status
            status_response = requests.get(f"{self.bot_url}/status", timeout=10)
            trades_response = requests.get(f"{self.bot_url}/trades", timeout=10)
            
            if status_response.status_code != 200 or trades_response.status_code != 200:
                return None
            
            status_data = status_response.json()
            trades_data = trades_response.json()
            
            # Calculate metrics
            trades = trades_data.get('trades', [])
            
            if not trades:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'balance': status_data.get('balance', 0)
                }
            
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'balance': status_data.get('balance', 0),
                'daily_pnl': status_data.get('daily_pnl', 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def log_performance(self):
        """Log performance metrics to file"""
        metrics = self.collect_metrics()
        if metrics:
            # Append to performance log
            with open('performance.jsonl', 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Print summary
            logger.info(f"Performance Update - Trades: {metrics['total_trades']}, "
                       f"Win Rate: {metrics['win_rate']:.1f}%, "
                       f"P&L: ₹{metrics['total_pnl']:.2f}, "
                       f"Balance: ₹{metrics['balance']:.2f}")
    
    def performance_loop(self, log_interval: int = 300):  # 5 minutes
        """Performance logging loop"""
        logger.info(f"Starting performance tracker - logging every {log_interval}s")
        
        while True:
            try:
                self.log_performance()
                time.sleep(log_interval)
            except KeyboardInterrupt:
                logger.info("Performance tracker stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in performance loop: {e}")
                time.sleep(60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bot Monitoring and Performance Tracking')
    parser.add_argument('--mode', choices=['monitor', 'performance', 'both'], 
                       default='both', help='Monitoring mode')
    parser.add_argument('--url', default='http://localhost:10000', 
                       help='Bot URL')
    parser.add_argument('--check-interval', type=int, default=60,
                       help='Health check interval in seconds')
    parser.add_argument('--perf-interval', type=int, default=300,
                       help='Performance logging interval in seconds')
    
    args = parser.parse_args()
    
    if args.mode in ['monitor', 'both']:
        monitor = BotMonitor(args.url)
        if args.mode == 'monitor':
            monitor.monitor_loop(args.check_interval)
        else:
            # Run monitor in separate thread for 'both' mode
            import threading
            monitor_thread = threading.Thread(
                target=monitor.monitor_loop, 
                args=(args.check_interval,),
                daemon=True
            )
            monitor_thread.start()
    
    if args.mode in ['performance', 'both']:
        tracker = PerformanceTracker(args.url)
        tracker.performance_loop(args.perf_interval)

if __name__ == "__main__":
    main()
