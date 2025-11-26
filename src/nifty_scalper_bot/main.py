#!/usr/bin/env python3
"""
Nifty Scalper Bot - Main Entry Point
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging before importing app components
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        logger.info("=" * 80)
        logger.info("Starting Nifty Scalper Bot")
        logger.info("=" * 80)
        
        # Import after logging setup to avoid issues
        from nifty_scalper_bot.core.app import NiftyScalperApp
        
        # Create and run app
        app = NiftyScalperApp()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("
👋 Bot stopped by user")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()