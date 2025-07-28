import os
import logging
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

logger = logging.getLogger(__name__)

class ZerodhaAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the request token from URL
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        if 'request_token' in query_params:
            request_token = query_params['request_token'][0]
            self.server.request_token = request_token
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            success_html = """
            <html>
                <body>
                    <h2>✅ Authentication Successful!</h2>
                    <p>You can close this window and return to your application.</p>
                    <p>Request token has been captured.</p>
                </body>
            </html>
            """
            self.wfile.write(success_html.encode())
        else:
            # Send error response
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            error_html = """
            <html>
                <body>
                    <h2>❌ Authentication Failed!</h2>
                    <p>No request token found in the callback.</p>
                </body>
            </html>
            """
            self.wfile.write(error_html.encode())

class ZerodhaAuthenticator:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = None
        self.request_token = None
    
    def get_login_url(self):
        """Get the login URL for Zerodha authentication"""
        return self.kite.login_url()
    
    def start_local_server(self, port=8000):
        """Start a local server to capture the callback"""
        try:
            server = HTTPServer(('localhost', port), ZerodhaAuthHandler)
            server.request_token = None
            
            logger.info(f"Starting local server on port {port}")
            logger.info("Please complete the authentication in your browser")
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            return server
        except Exception as e:
            logger.error(f"Failed to start local server: {e}")
            return None
    
    def authenticate_interactive(self):
        """Interactive authentication flow"""
        try:
            # Get login URL
            login_url = self.get_login_url()
            logger.info(f"Login URL: {login_url}")
            
            # Open browser automatically
            logger.info("Opening browser for authentication...")
            webbrowser.open(login_url)
            
            # Start local server to capture callback
            server = self.start_local_server(8000)
            if not server:
                return False
            
            # Wait for request token (with timeout)
            import time
            timeout = 120  # 2 minutes
            start_time = time.time()
            
            while not hasattr(server, 'request_token') or server.request_token is None:
                if time.time() - start_time > timeout:
                    logger.error("Authentication timeout")
                    server.shutdown()
                    return False
                time.sleep(1)
            
            # Get the request token
            self.request_token = server.request_token
            logger.info(f"Received request token: {self.request_token}")
            
            # Shutdown server
            server.shutdown()
            
            # Generate access token
            return self.generate_access_token()
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def generate_access_token(self):
        """Generate access token from request token"""
        try:
            if not self.request_token:
                logger.error("No request token available")
                return False
            
            # Generate session
            data = self.kite.generate_session(self.request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            logger.info("✅ Access token generated successfully!")
            logger.info(f"Access Token: {self.access_token}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate access token: {e}")
            return False
    
    def save_access_token(self, filepath=".env"):
        """Save access token to .env file"""
        try:
            # Read existing content
            lines = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    lines = f.readlines()
            
            # Update or add access token
            access_token_line = f"ZERODHA_ACCESS_TOKEN={self.access_token}\n"
            token_found = False
            
            for i, line in enumerate(lines):
                if line.startswith("ZERODHA_ACCESS_TOKEN="):
                    lines[i] = access_token_line
                    token_found = True
                    break
            
            if not token_found:
                lines.append(access_token_line)
            
            # Write back to file
            with open(filepath, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"✅ Access token saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save access token: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # This would be called from your main application
    API_KEY = os.getenv('ZERODHA_API_KEY')
    API_SECRET = os.getenv('ZERODHA_API_SECRET')
    
    if API_KEY and API_SECRET:
        authenticator = ZerodhaAuthenticator(API_KEY, API_SECRET)
        # authenticator.authenticate_interactive()  # Uncomment to run
    else:
        print("Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in environment variables")
