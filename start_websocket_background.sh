#!/bin/bash

echo "🚀 Starting WebSocket Connection in Background..."

# Create logs directory
mkdir -p logs

# Run WebSocket client in background
nohup python -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

import logging
from src.data_streaming.websocket_client import WebSocketClient
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/websocket.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        print('🧪 Starting WebSocket connection test...')
        
        # Initialize WebSocket client
        ws_client = WebSocketClient()
        
        # Set up callbacks
        def on_ticks(ticks):
            logger.info(f'📩 Received {len(ticks)} ticks')
        
        def on_connect(response):
            logger.info('✅ Connected to WebSocket')
        
        def on_close(code, reason):
            logger.info(f' WebSocket closed. Code: {code}, Reason: {reason}')
        
        def on_error(code, reason):
            logger.error(f' WebSocket error. Code: {code}, Reason: {reason}')
        
        ws_client.set_ticks_callback(on_ticks)
        ws_client.set_connect_callback(on_connect)
        ws_client.set_close_callback(on_close)
        ws_client.set_error_callback(on_error)
        
        # Initialize connection
        if ws_client.initialize_connection():
            logger.info('✅ WebSocket connection initialized')
            
            # Subscribe to Nifty 50 (token 256265)
            if ws_client.subscribe_tokens([256265]):
                logger.info('✅ Subscribed to Nifty 50')
                
                # Start streaming
                if ws_client.start_streaming():
                    logger.info('✅ WebSocket streaming started')
                    print('✅ WebSocket connection established successfully!')
                    
                    # Keep running
                    import time
                    while True:
                        time.sleep(60)
                else:
                    logger.error('❌ Failed to start streaming')
            else:
                logger.error('❌ Failed to subscribe to Nifty 50')
        else:
            logger.error('❌ Failed to initialize connection')
            
    except KeyboardInterrupt:
        logger.info('🛑 WebSocket client stopped by user')
    except Exception as e:
        logger.error(f'❌ Error in WebSocket client: {e}')

if __name__ == '__main__':
    main()
" > logs/websocket_client.log 2>&1 &

# Capture process ID
WS_PID=$!

echo "✅ WebSocket client started with PID: $WS_PID"
echo "📊 Logs: tail -f logs/websocket_client.log"
echo "🛑 To stop: kill $WS_PID"

# Wait a moment for startup
sleep 3

# Check if running
if ps -p $WS_PID > /dev/null; then
    echo "✅ WebSocket client is running in background!"
else
    echo "❌ WebSocket client failed to start"
    tail -n 10 logs/websocket_client.log
fi
