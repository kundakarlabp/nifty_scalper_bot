#!/usr/bin/env python3
"""
Run the Nifty Scalper Web Dashboard
"""
import sys
import os

# Add the project root to Python path (go up 3 levels: web_dashboard -> src -> nifty_scalper_bot)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
from app import app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the web dashboard"""
    try:
        logger.info("🚀 Starting Nifty Scalper Web Dashboard...")
        logger.info("🌐 Visit: http://localhost:8000")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0', 
            port=8000, 
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate processes
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Web dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting web dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
