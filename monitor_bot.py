#!/usr/bin/env python3
"""
Monitor the Nifty Scalper Trading Bot
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import subprocess
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BotMonitor:
    def __init__(self):
        self.bot_process_name = "src/main.py"
        self.log_file = "logs/trading_bot.log"
        
    def is_bot_running(self) -> bool:
        """Check if bot is currently running"""
        try:
            result = subprocess.run(['pgrep', '-f', self.bot_process_name], 
                                   capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking bot status: {e}")
            return False
    
    def get_bot_pid(self) -> str:
        """Get bot process ID"""
        try:
            result = subprocess.run(['pgrep', '-f', self.bot_process_name], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return "Not running"
        except Exception as e:
            logger.error(f"Error getting bot PID: {e}")
            return "Unknown"
    
    def start_bot(self) -> bool:
        """Start the bot"""
        try:
            if self.is_bot_running():
                logger.info("Bot is already running")
                return True
            
            # Start bot in background
            subprocess.Popen(['nohup', 'python', 'src/main.py', '--mode', 'realtime', '--trade'], 
                           stdout=open('logs/trading_bot.log', 'a'), 
                           stderr=open('logs/trading_bot.log', 'a'), 
                           preexec_fn=os.setsid)
            
            time.sleep(3)  # Wait for startup
            
            if self.is_bot_running():
                logger.info("✅ Bot started successfully")
                return True
            else:
                logger.error("❌ Failed to start bot")
                return False
                
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False
    
    def stop_bot(self) -> bool:
        """Stop the bot"""
        try:
            if not self.is_bot_running():
                logger.info("Bot is not running")
                return True
            
            # Kill bot process
            subprocess.run(['pkill', '-f', self.bot_process_name])
            
            time.sleep(3)  # Wait for shutdown
            
            if not self.is_bot_running():
                logger.info("✅ Bot stopped successfully")
                return True
            else:
                logger.warning("Bot may still be running")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return False
    
    def get_recent_logs(self, lines: int = 20) -> str:
        """Get recent log entries"""
        try:
            result = subprocess.run(['tail', '-n', str(lines), self.log_file], 
                                   capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else "No logs available"
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return "Error reading logs"
    
    def get_system_status(self) -> dict:
        """Get system status"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bot_running': self.is_bot_running(),
            'bot_pid': self.get_bot_pid(),
            'log_file': self.log_file
        }
    
    def monitor_continuously(self, interval: int = 60):
        """Monitor bot continuously"""
        logger.info("Starting continuous monitoring...")
        
        try:
            while True:
                status = self.get_system_status()
                logger.info(f"📊 System Status: Bot Running: {status['bot_running']}, PID: {status['bot_pid']}")
                
                # Check if bot is running
                if not status['bot_running']:
                    logger.warning("⚠️  Bot is not running. Attempting to restart...")
                    if self.start_bot():
                        logger.info("✅ Bot restarted successfully")
                    else:
                        logger.error("❌ Failed to restart bot")
                
                # Wait before next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")

def main():
    """Main entry point"""
    monitor = BotMonitor()
    
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Monitor Nifty Scalper Trading Bot')
    parser.add_argument('--action', choices=['status', 'start', 'stop', 'restart', 'logs', 'monitor'], 
                       default='status', help='Action to perform')
    parser.add_argument('--lines', type=int, default=20, help='Number of log lines to show')
    args = parser.parse_args()
    
    try:
        if args.action == 'status':
            status = monitor.get_system_status()
            print(f"📊 Bot Status Report")
            print(f"   Timestamp: {status['timestamp']}")
            print(f"   Running: {'✅ YES' if status['bot_running'] else '❌ NO'}")
            print(f"   Process ID: {status['bot_pid']}")
            print(f"   Log File: {status['log_file']}")
            
        elif args.action == 'start':
            if monitor.start_bot():
                print("✅ Bot started successfully")
            else:
                print("❌ Failed to start bot")
                
        elif args.action == 'stop':
            if monitor.stop_bot():
                print("✅ Bot stopped successfully")
            else:
                print("❌ Failed to stop bot")
                
        elif args.action == 'restart':
            print("🔄 Restarting bot...")
            monitor.stop_bot()
            time.sleep(3)
            if monitor.start_bot():
                print("✅ Bot restarted successfully")
            else:
                print("❌ Failed to restart bot")
                
        elif args.action == 'logs':
            logs = monitor.get_recent_logs(args.lines)
            print(f"📜 Recent Logs ({args.lines} lines):")
            print(logs)
            
        elif args.action == 'monitor':
            monitor.monitor_continuously()
            
    except Exception as e:
        logger.error(f"Error in monitor: {e}")

if __name__ == "__main__":
    main()
