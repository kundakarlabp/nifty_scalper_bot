#!/usr/bin/env python3
"""
Simple Bot Control Script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_command(command: str):
    """Send command via Telegram"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ Telegram credentials not configured")
            print("💡 Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': command,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Command '{command}' sent to Telegram bot")
            print("📱 Check your Telegram app for the response")
            return True
        else:
            print(f"❌ Failed to send command: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending command: {e}")
        return False

def main():
    """Main control function"""
    if len(sys.argv) < 2:
        print("🔧 Nifty Scalper Bot Control")
        print("=" * 40)
        print("Usage: python bot_control.py <command>")
        print("")
        print("Available commands:")
        print("  /start     - Start trading system")
        print("  /stop      - Stop trading system")
        print("  /status    - Show system status")
        print("  /enable    - Enable trade execution")
        print("  /disable   - Disable trade execution")
        print("  /help      - Show all commands")
        print("")
        print("Example: python bot_control.py /status")
        return
    
    command = sys.argv[1]
    send_telegram_command(command)

if __name__ == "__main__":
    main()
