#!/usr/bin/env python3
"""
Interactive Zerodha Authentication Script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import webbrowser
import time
from kiteconnect import KiteConnect
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()

def authenticate_interactive():
    """Interactive authentication with Zerodha"""
    print("🔐 Zerodha Interactive Authentication")
    print("=" * 50)
    
    # Get credentials from .env
    api_key = os.getenv('ZERODHA_API_KEY')
    api_secret = os.getenv('ZERODHA_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Zerodha API credentials not found in .env file")
        print("Please update your .env file with ZERODHA_API_KEY and ZERODHA_API_SECRET")
        return False
    
    try:
        # Initialize Kite Connect
        kite = KiteConnect(api_key=api_key)
        
        # Generate login URL
        login_url = kite.login_url()
        print(f"🔗 Login URL: {login_url}")
        print("\n📝 Instructions:")
        print("1. Click the login URL above to open Zerodha authentication page")
        print("2. Log in to your Zerodha account")
        print("3. Complete the authentication process")
        print("4. You'll be redirected to a URL with 'request_token' parameter")
        print("5. Copy the complete URL from your browser address bar")
        print("6. Paste it below when prompted")
        
        # Try to open browser automatically
        try:
            webbrowser.open(login_url)
            print("\n🌐 Browser opened automatically with login URL")
        except:
            print("\n⚠️  Could not open browser automatically")
            print("   Please copy and paste the URL in your browser manually")
        
        # Get request token from user
        print("\n📋 After logging in, please paste the complete redirect URL:")
        redirect_url = input("   URL: ").strip()
        
        if not redirect_url:
            print("❌ No URL provided")
            return False
        
        # Extract request token from URL
        if 'request_token=' in redirect_url:
            # Parse request token from URL
            import urllib.parse
            parsed_url = urllib.parse.urlparse(redirect_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            request_token = query_params.get('request_token', [None])[0]
        else:
            # Assume user pasted just the request token
            request_token = redirect_url.split('/')[-1] if '/' in redirect_url else redirect_url
        
        if not request_token:
            print("❌ Could not extract request token from URL")
            return False
        
        print(f"\n✅ Request token extracted: {request_token[:10]}...")
        
        # Generate session
        print("🔄 Generating session...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Test the access token
        kite.set_access_token(access_token)
        profile = kite.profile()
        
        print(f"\n✅ Authentication successful!")
        print(f"👤 User: {profile.get('user_name', 'Unknown')}")
        print(f"🆔 User ID: {profile.get('user_id', 'Unknown')}")
        print(f"🔑 Access Token: {access_token}")
        
        # Save access token to .env file
        print("\n💾 Saving access token to .env file...")
        set_key('.env', 'ZERODHA_ACCESS_TOKEN', access_token)
        
        print("✅ Access token saved successfully!")
        print("\n🎉 Authentication completed successfully!")
        print("🚀 You can now run your trading bot!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return False

def main():
    """Main entry point"""
    print("🚀 Starting Zerodha Authentication Process...")
    
    success = authenticate_interactive()
    
    if success:
        print("\n✅ Authentication successful!")
        print("🔧 Next steps:")
        print("   1. Run your trading bot:")
        print("      python src/main.py --mode realtime --trade")
        print("   2. Or start in background:")
        print("      nohup python src/main.py --mode realtime --trade > logs/trading_bot.log 2>&1 &")
        print("   3. Control via Telegram commands")
    else:
        print("\n❌ Authentication failed!")
        print("💡 Please check your Zerodha credentials and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
