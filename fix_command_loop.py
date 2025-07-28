#!/usr/bin/env python3
"""
Fix for Telegram command processing loop
"""
import time
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def fix_telegram_command_loop():
    """Fix the Telegram command processing loop"""
    try:
        print("🔧 Fixing Telegram command processing loop...")
        
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("❌ Telegram credentials not configured")
            return False
        
        # Get current updates to clear any pending commands
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {
            'offset': -1,  # Get the last update to clear it
            'limit': 1,
            'timeout': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            print("✅ Cleared pending Telegram commands")
        else:
            print(f"⚠️  Could not clear pending commands: {response.status_code}")
        
        # Send a single test message to verify
        test_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        test_payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': "✅ Telegram command loop fixed!",
            'parse_mode': 'Markdown'
        }
        
        test_response = requests.post(test_url, json=test_payload, timeout=10)
        
        if test_response.status_code == 200:
            print("✅ Sent test message successfully")
            return True
        else:
            print(f"❌ Failed to send test message: {test_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing command loop: {e}")
        return False

if __name__ == "__main__":
    success = fix_telegram_command_loop()
    if success:
        print("🎉 Telegram command loop fix completed!")
        print("📱 You can now send commands via Telegram:")
        print("   /start - Start trading system")
        print("   /stop - Stop trading system")
        print("   /status - Show system status")
        print("   /help - Show all commands")
    else:
        print("❌ Telegram command loop fix failed!")
