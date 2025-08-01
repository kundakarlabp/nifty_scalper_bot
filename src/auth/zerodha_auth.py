import os
import logging
import threading
import webbrowser
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class ZerodhaAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if 'request_token' in query_params:
            request_token = query_params['request_token'][0]
            self.server.request_token = request_token
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("""
            <html>
              <body>
                <h2>✅ Authentication Successful!</h2>
                <p>You can now return to your bot. This window can be closed.</p>
              </body>
            </html>
            """.encode("utf-8"))
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write("""
            <html>
              <body>
                <h2>❌ Authentication Failed</h2>
                <p>No request token found in URL.</p>
              </body>
            </html>
            """.encode("utf-8"))


class ZerodhaAuthenticator:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self.access_token = None
        self.request_token = None

    def get_login_url(self):
        return self.kite.login_url()

    def start_local_server(self, port=8000):
        try:
            server = HTTPServer(('localhost', port), ZerodhaAuthHandler)
            server.request_token = None
            logger.info(f"🌐 Local server started at http://localhost:{port}")
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            return server
        except Exception as e:
            logger.error(f"❌ Failed to start local server: {e}")
            return None

    def authenticate_interactive(self):
        try:
            login_url = self.get_login_url()
            logger.info(f"🔗 Login URL: {login_url}")
            webbrowser.open(login_url)

            server = self.start_local_server(8000)
            if not server:
                return False

            timeout = 120
            start = time.time()

            while not getattr(server, 'request_token', None):
                if time.time() - start > timeout:
                    logger.error("❌ Timeout waiting for request token")
                    server.shutdown()
                    return False
                time.sleep(1)

            self.request_token = server.request_token
            logger.info(f"📩 Received request token: {self.request_token}")
            server.shutdown()

            return self.generate_access_token()

        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False

    def generate_access_token(self):
        try:
            if not self.request_token:
                logger.error("❌ No request token to exchange")
                return False

            data = self.kite.generate_session(self.request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)

            logger.info("✅ Access token generated")
            logger.info(f"🔐 Access Token: {self.access_token}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to generate access token: {e}")
            return False

    def save_access_token(self, filepath=".env"):
        try:
            lines = []
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    lines = f.readlines()

            new_line = f"KITE_ACCESS_TOKEN={self.access_token}\n"
            updated = False

            for i, line in enumerate(lines):
                if line.startswith("KITE_ACCESS_TOKEN="):
                    lines[i] = new_line
                    updated = True
                    break

            if not updated:
                lines.append(new_line)

            with open(filepath, "w") as f:
                f.writelines(lines)

            logger.info(f"💾 Access token saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save token to {filepath}: {e}")
            return False


def get_kite_client() -> KiteConnect:
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not all([api_key, api_secret, access_token]):
        raise EnvironmentError("Missing API credentials. Set ZERODHA_API_KEY, ZERODHA_API_SECRET, and KITE_ACCESS_TOKEN.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


if __name__ == "__main__":
    API_KEY = os.getenv("ZERODHA_API_KEY")
    API_SECRET = os.getenv("ZERODHA_API_SECRET")

    if API_KEY and API_SECRET:
        auth = ZerodhaAuthenticator(API_KEY, API_SECRET)
        # Uncomment this to initiate login flow:
        # auth.authenticate_interactive()
    else:
        print("⚠️ Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in .env or environment variables.")
