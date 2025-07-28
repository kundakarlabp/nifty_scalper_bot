#!/usr/bin/env python3
"""
Process Management Script - Check and manage all running bot processes
"""
import sys
import os
import psutil
import logging
from datetime import datetime
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
timezone = pytz.timezone('Asia/Kolkata')

def get_bot_processes():
    """Get all bot-related processes"""
    bot_processes = []
    
    try:
        # Iterate through all processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                # Check if process is related to our bot
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                if ('src/main.py' in cmdline or 
                    'web_dashboard' in cmdline or 
                    'websocket_client' in cmdline or 
                    'telegram_controller' in cmdline or
                    'nifty_scalper' in cmdline or
                    'kiteconnect' in cmdline):
                    
                    # Calculate uptime
                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                    uptime = datetime.now() - create_time
                    
                    bot_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline,
                        'create_time': create_time,
                        'uptime': str(uptime).split('.')[0],  # Remove microseconds
                        'status': proc.status()
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process no longer exists or access denied
                continue
            except Exception as e:
                logger.debug(f"Error checking process {proc.info['pid']}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error getting bot processes: {e}")
    
    return bot_processes

def display_process_status():
    """Display current process status"""
    try:
        print("=" * 60)
        print("🤖 NIFTY SCALPER BOT PROCESS STATUS")
        print("=" * 60)
        
        bot_processes = get_bot_processes()
        
        if not bot_processes:
            print("❌ No bot processes found")
            print("💡 Bot is not currently running")
            return False
        
        print(f"✅ Found {len(bot_processes)} bot-related processes:")
        print("-" * 60)
        
        for i, proc in enumerate(bot_processes, 1):
            print(f"{i}. PID: {proc['pid']}")
            print(f"   Name: {proc['name']}")
            print(f"   Status: {proc['status'].upper()}")
            print(f"   Uptime: {proc['uptime']}")
            print(f"   Command: {proc['cmdline'][:100]}...")
            print(f"   Started: {proc['create_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print()
        
        return True
        
    except Exception as e:
        logger.error(f"Error displaying process status: {e}")
        return False

def stop_bot_processes():
    """Stop all bot-related processes"""
    try:
        print("�� Stopping all bot processes...")
        
        bot_processes = get_bot_processes()
        
        if not bot_processes:
            print("✅ No bot processes found to stop")
            return True
        
        stopped_count = 0
        for proc in bot_processes:
            try:
                process = psutil.Process(proc['pid'])
                process.terminate()
                print(f"✅ Terminated process {proc['pid']}")
                stopped_count += 1
            except psutil.NoSuchProcess:
                print(f"⚠️  Process {proc['pid']} already stopped")
            except Exception as e:
                print(f"❌ Error stopping process {proc['pid']}: {e}")
        
        print(f"✅ Stopped {stopped_count} processes")
        return True
        
    except Exception as e:
        logger.error(f"Error stopping bot processes: {e}")
        return False

def start_bot_background():
    """Start bot in background"""
    try:
        print("🚀 Starting bot in background...")
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Start bot with proper background execution
        import subprocess
        
        # Start main bot
        bot_process = subprocess.Popen([
            sys.executable, 'src/main.py', '--mode', 'realtime', '--trade'
        ], stdout=open('logs/bot_stdout.log', 'w'), 
           stderr=open('logs/bot_stderr.log', 'w'),
           preexec_fn=os.setsid)
        
        print(f"✅ Bot started in background with PID: {bot_process.pid}")
        print("📊 Logs:")
        print("   Standard output: logs/bot_stdout.log")
        print("   Error output: logs/bot_stderr.log")
        print("🔧 To monitor: tail -f logs/bot_stdout.log")
        print("🛑 To stop: kill -TERM -{bot_process.pid}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error starting bot in background: {e}")
        return False

def monitor_bot_logs():
    """Monitor bot logs"""
    try:
        print("🔍 Monitoring bot logs (Press Ctrl+C to stop)...")
        print("=" * 60)
        
        import subprocess
        import time
        
        # Monitor stdout log
        try:
            subprocess.run(['tail', '-f', 'logs/bot_stdout.log'], check=True)
        except KeyboardInterrupt:
            print("\n🛑 Log monitoring stopped")
        except Exception as e:
            print(f"❌ Error monitoring logs: {e}")
            # Fallback to manual monitoring
            try:
                with open('logs/bot_stdout.log', 'r') as f:
                    # Go to end of file
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            print(line.strip())
                        else:
                            time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Log monitoring stopped")
            except Exception as e2:
                print(f"❌ Error in fallback log monitoring: {e2}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error monitoring bot logs: {e}")
        return False

def get_resource_usage():
    """Get resource usage of bot processes"""
    try:
        bot_processes = get_bot_processes()
        
        if not bot_processes:
            print("📊 No bot processes running")
            return
        
        print("📊 RESOURCE USAGE:")
        print("-" * 40)
        
        total_cpu = 0
        total_memory = 0
        
        for proc in bot_processes:
            try:
                process = psutil.Process(proc['pid'])
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                total_cpu += cpu_percent
                total_memory += memory_mb
                
                print(f"PID {proc['pid']}: {cpu_percent:.1f}% CPU, {memory_mb:.1f}MB RAM")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"PID {proc['pid']}: Access denied")
            except Exception as e:
                print(f"PID {proc['pid']}: Error getting usage - {e}")
        
        print(f"📈 Total: {total_cpu:.1f}% CPU, {total_memory:.1f}MB RAM")
        
        return True
        
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        return False

def main():
    """Main entry point"""
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Nifty Scalper Bot Process Manager')
        parser.add_argument('--action', choices=['status', 'start', 'stop', 'restart', 'monitor', 'resources'], 
                           default='status', help='Action to perform')
        args = parser.parse_args()
        
        if args.action == 'status':
            display_process_status()
        elif args.action == 'start':
            start_bot_background()
        elif args.action == 'stop':
            stop_bot_processes()
        elif args.action == 'restart':
            stop_bot_processes()
            time.sleep(3)
            start_bot_background()
        elif args.action == 'monitor':
            monitor_bot_logs()
        elif args.action == 'resources':
            get_resource_usage()
        else:
            display_process_status()
            
    except Exception as e:
        logger.error(f"Error in process manager: {e}")

if __name__ == "__main__":
    main()
