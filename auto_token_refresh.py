#!/usr/bin/env python3
"""
Automatic Zerodha Token Refresh Script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import logging
from datetime import datetime
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/token_refresh.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ZerodhaTokenRefresher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ZERODHA_API_KEY')
        self.api_secret = os.getenv('ZERODHA_API_SECRET')
        self.access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
        self.kite = None
        self.setup_kite()
    
    def setup_kite(self):
        """Initialize Kite Connect"""
        try:
            if self.api_key:
                self.kite = KiteConnect(api_key=self.api_key)
                if self.access_token:
                    self.kite.set_access_token(self.access_token)
            logger.info("✅ Kite Connect initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing Kite Connect: {e}")
    
    def is_token_valid(self) -> bool:
        """Check if current token is valid"""
        try:
            if not self.kite:
                return False
            
            # Try to fetch user profile
            profile = self.kite.profile()
            logger.info(f"✅ Token is valid. User: {profile.get('user_name', 'Unknown')}")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Token is invalid or expired: {e}")
            return False
    
    def refresh_token(self) -> bool:
        """Refresh Zerodha access token"""
        try:
            logger.info("🔄 Starting token refresh process...")
            
            # This would require manual intervention for now
            # In a production environment, you'd need to implement
            # automatic token refresh using TOTP or other methods
            
            print("\n🔐 Zerodha Token Refresh Required")
            print("=" * 40)
            print("⚠️  Your Zerodha access token has expired!")
            print("🔄 Please run the interactive authentication:")
            print("   python authenticate_interactive.py")
            print("\n💡 Or manually update your .env file with a new access token")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error refreshing token: {e}")
            return False
    
    def monitor_and_refresh(self):
        """Monitor token and refresh when needed"""
        logger.info("🔍 Starting token monitoring...")
        
        while True:
            try:
                # Check token validity every hour
                if not self.is_token_valid():
                    logger.warning("⚠️  Token is invalid. Refresh required.")
                    self.refresh_token()
                else:
                    logger.info("✅ Token is still valid")
                
                # Wait for 1 hour before next check
                time.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                logger.info("🛑 Token monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in token monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retry

def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize token refresher
    refresher = ZerodhaTokenRefresher()
    
    # Check current token status
    if refresher.is_token_valid():
        print("✅ Your Zerodha access token is valid!")
        print("🚀 You can now run your trading bot!")
    else:
        print("❌ Your Zerodha access token is invalid or expired!")
        print("🔄 Please run interactive authentication:")
        print("   python authenticate_interactive.py")
    
    # Start monitoring (optional)
    import argparse
    parser = argparse.ArgumentParser(description='Zerodha Token Refresher')
    parser.add_argument('--monitor', action='store_true', 
                       help='Start continuous token monitoring')
    args = parser.parse_args()
    
    if args.monitor:
        print("🔍 Starting continuous token monitoring...")
        refresher.monitor_and_refresh()

if __name__ == "__main__":
    main()
