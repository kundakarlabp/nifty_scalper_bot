import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test credentials
zerodha_key = os.getenv('ZERODHA_API_KEY')
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')

print("🔐 Configuration Test:")
print(f"Zerodha API Key: {'✅ SET' if zerodha_key else '❌ MISSING'}")
print(f"Telegram Bot Token: {'✅ SET' if telegram_token else '❌ MISSING'}")

# Don't print actual values for security
if zerodha_key:
    print(f"Zerodha Key Length: {len(zerodha_key)} characters")
if telegram_token:
    print(f"Telegram Token Length: {len(telegram_token)} characters")

print("✅ Configuration test completed!")
