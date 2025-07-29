# src/data_streaming/data_processor.py
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StreamingDataProcessor:
    """
    Processes real-time market ticks, maintains data buffers,
    and generates OHLC data for technical analysis.
    """
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize the StreamingDataProcessor.

        Args:
            buffer_size (int): Maximum number of ticks to store per instrument.
        """
        self.buffer_size: int = buffer_size
        self.data_buffers: Dict[int, deque] = {}  # token -> deque of ticks
        self.ohlc_data: Dict[int, pd.DataFrame] = {}  # token -> DataFrame of OHLC data
        self.last_tick_time: Dict[int, Optional[datetime]] = {}  # token -> last tick timestamp

    def process_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single tick and update buffers.

        Args:
            tick (Dict): Raw tick data from Kite WebSocket.

        Returns:
            Optional[Dict]: Processed tick data, or None on error.
        """
        try:
            token = tick.get('instrument_token')
            if not token:
                logger.warning("Tick missing instrument token")
                return None

            # Initialize buffer for this token if not exists
            if token not in self.data_buffers:
                self.data_buffers[token] = deque(maxlen=self.buffer_size)
                self.ohlc_data[token] = pd.DataFrame()
                self.last_tick_time[token] = None

            # Process and add tick to buffer
            processed_tick = self._process_single_tick(tick)
            self.data_buffers[token].append(processed_tick)

            # Update last tick time
            self.last_tick_time[token] = processed_tick.get('timestamp')

            logger.debug(f"Processed tick for token {token}")
            return processed_tick

        except Exception as e:
            logger.error(f"Error processing tick for token {tick.get('instrument_token', 'unknown')}: {e}")
            return None

    def _process_single_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process individual tick data into a standardized format.

        Args:
            tick (Dict): Raw tick data.

        Returns:
            Dict: Standardized tick data.
        """
        try:
            # Extract core tick data
            processed = {
                'instrument_token': tick.get('instrument_token'),
                'timestamp': tick.get('timestamp', datetime.now()),
                'last_price': float(tick.get('last_price', 0.0)),
                'last_quantity': int(tick.get('last_quantity', 0)),
                'buy_quantity': int(tick.get('buy_quantity', 0)),
                'sell_quantity': int(tick.get('sell_quantity', 0)),
                'volume': int(tick.get('volume_traded', tick.get('volume', 0))),  # Kite uses volume_traded
                'oi': int(tick.get('oi', 0)),
                'net_change': float(tick.get('net_change', 0.0))
            }

            # Extract market depth data if available
            if 'depth' in tick and isinstance(tick['depth'], dict):
                depth = tick['depth']
                buy_levels = depth.get('buy', [])
                sell_levels = depth.get('sell', [])
                
                # Get best bid/ask
                processed['bid_price'] = float(buy_levels[0].get('price', 0.0)) if buy_levels else 0.0
                processed['bid_quantity'] = int(buy_levels[0].get('quantity', 0)) if buy_levels else 0
                processed['ask_price'] = float(sell_levels[0].get('price', 0.0)) if sell_levels else 0.0
                processed['ask_quantity'] = int(sell_levels[0].get('quantity', 0)) if sell_levels else 0

            return processed

        except Exception as e:
            logger.error(f"Error in _process_single_tick: {e}")
            # Return minimal valid tick on error
            return {
                'instrument_token': tick.get('instrument_token'),
                'timestamp': datetime.now(),
                'last_price': 0.0,
                'last_quantity': 0
            }

    def update_ohlc(self, token: int, timeframe: str = '1min') -> Optional[pd.DataFrame]:
        """
        Update OHLC data for a specific instrument token.

        Args:
            token (int): Instrument token.
            timeframe (str): Pandas resample frequency (e.g., '1min', '5min').

        Returns:
            Optional[pd.DataFrame]: Updated OHLC DataFrame, or None on error.
        """
        try:
            if token not in self.data_buffers or not self.data_buffers[token]:
                logger.debug(f"No data buffer for token {token}")
                return None

            # Convert buffer to DataFrame
            ticks_list = list(self.data_buffers[token])
            if not ticks_list:
                return None

            ticks_df = pd.DataFrame(ticks_list)
            
            # Validate required columns
            if 'timestamp' not in ticks_df.columns or 'last_price' not in ticks_df.columns:
                logger.warning(f"Missing required columns in ticks for token {token}")
                return None

            # Set timestamp as index and ensure it's datetime
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            ticks_df.set_index('timestamp', inplace=True)
            
            # Validate index
            if not isinstance(ticks_df.index, pd.DatetimeIndex):
                logger.error(f"Failed to create DatetimeIndex for token {token}")
                return None

            # Resample to OHLC
            ohlc = ticks_df['last_price'].resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Add volume data if available
            if 'volume' in ticks_df.columns:
                volume = ticks_df['volume'].resample(timeframe).last().fillna(0)
                ohlc['volume'] = volume

            # Update stored OHLC data
            if not ohlc.empty:
                if token not in self.ohlc_data or self.ohlc_data[token].empty:
                    self.ohlc_data[token] = ohlc
                else:
                    # Concatenate and remove duplicates
                    combined = pd.concat([self.ohlc_data[token], ohlc])
                    self.ohlc_data[token] = combined[~combined.index.duplicated(keep='last')].sort_index()

                logger.debug(f"Updated OHLC data for token {token} with {len(ohlc)} bars")
                return ohlc
            else:
                logger.debug(f"No new OHLC data generated for token {token}")
                return None

        except Exception as e:
            logger.error(f"Error updating OHLC data for token {token}: {e}")
            return None

    def get_latest_data(self, token: int, bars: int = 50) -> Optional[pd.DataFrame]:
        """
        Get the latest OHLC data for a specific instrument.

        Args:
            token (int): Instrument token.
            bars (int): Number of latest bars to retrieve.

        Returns:
            Optional[pd.DataFrame]: Latest OHLC data, or None if not available.
        """
        try:
            if token not in self.ohlc_data:
                logger.debug(f"No OHLC data buffer for token {token}")
                return None

            ohlc_data = self.ohlc_data[token]
            if ohlc_data.empty:
                return None

            # Return last N bars
            result = ohlc_data.tail(bars).copy()
            if result.empty:
                return None
                
            return result

        except Exception as e:
            logger.error(f"Error getting latest data for token {token}: {e}")
            return None

    def get_current_price(self, token: int) -> Optional[float]:
        """
        Get the most recent price for a specific instrument.

        Args:
            token (int): Instrument token.

        Returns:
            Optional[float]: Current price, or None if not available.
        """
        try:
            if token not in self.data_buffers:
                logger.debug(f"No buffer for token {token}")
                return None

            buffer = self.data_buffers[token]
            if not buffer:
                return None

            # Get latest tick
            latest_tick = buffer[-1]
            price = latest_tick.get('last_price')
            
            if price is not None:
                return float(price)
            return None

        except Exception as e:
            logger.error(f"Error getting current price for token {token}: {e}")
            return None

    def get_market_depth(self, token: int) -> Optional[Dict[str, float]]:
        """
        Get the current market depth for a specific instrument.

        Args:
            token (int): Instrument token.

        Returns:
            Optional[Dict]: Market depth data, or None if not available.
        """
        try:
            if token not in self.data_buffers:
                logger.debug(f"No buffer for token {token}")
                return None

            buffer = self.data_buffers[token]
            if not buffer:
                return None

            # Get latest tick with depth data
            latest_tick = buffer[-1]
            depth_keys = ['bid_price', 'bid_quantity', 'ask_price', 'ask_quantity']
            
            depth_data = {
                key: float(latest_tick.get(key, 0.0)) 
                for key in depth_keys 
                if key in latest_tick
            }
            
            return depth_data if depth_data else None

        except Exception as e:
            logger.error(f"Error getting market depth for token {token}: {e}")
            return None

    def clear_buffer(self, token: int) -> None:
        """
        Clear all data buffers for a specific instrument.

        Args:
            token (int): Instrument token.
        """
        try:
            if token in self.data_buffers:
                self.data_buffers[token].clear()
            if token in self.ohlc_data:
                self.ohlc_data[token] = pd.DataFrame()
            if token in self.last_tick_time:
                self.last_tick_time[token] = None

            logger.info(f"Cleared buffer for token {token}")

        except Exception as e:
            logger.error(f"Error clearing buffer for token {token}: {e}")

    def get_buffer_status(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the status of all data buffers.

        Returns:
            Dict: Status information for all tokens.
        """
        try:
            status = {}
            for token in self.data_buffers:
                status[token] = {
                    'buffer_size': len(self.data_buffers[token]),
                    'ohlc_size': len(self.ohlc_data.get(token, [])),
                    'last_tick': str(self.last_tick_time.get(token, 'N/A'))
                }
            return status

        except Exception as e:
            logger.error(f"Error getting buffer status: {e}")
            return {}


# Example usage (if run directly)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    processor = StreamingDataProcessor()

    # Simulate processing some ticks
    sample_tick = {
        'instrument_token': 256265,
        'timestamp': datetime.now(),
        'last_price': 18000.5,
        'last_quantity': 75,
        'buy_quantity': 1000,
        'sell_quantity': 800,
        'volume_traded': 100000,  # Kite's field name
        'depth': {
            'buy': [{'price': 18000.0, 'quantity': 75}],
            'sell': [{'price': 18001.0, 'quantity': 75}]
        }
    }

    processed = processor.process_tick(sample_tick)
    print(f"Processed tick: {processed}")

    # Update OHLC data
    ohlc = processor.update_ohlc(256265, '1min')
    print(f"Generated OHLC:\n{ohlc}")

    current_price = processor.get_current_price(256265)
    print(f"Current price: {current_price}")

    depth = processor.get_market_depth(256265)
    print(f"Market depth: {depth}")

    status = processor.get_buffer_status()
    print(f"Buffer status: {status}")