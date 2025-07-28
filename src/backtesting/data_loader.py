import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from kiteconnect import KiteConnect
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN

logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    def __init__(self):
        self.kite = self._initialize_kite()
    
    def _initialize_kite(self) -> Optional[KiteConnect]:
        """Initialize Kite Connect for historical data"""
        try:
            if ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN:
                kite = KiteConnect(api_key=ZERODHA_API_KEY)
                kite.set_access_token(ZERODHA_ACCESS_TOKEN)
                logger.info("✅ Kite Connect initialized for historical data")
                return kite
            else:
                logger.warning("⚠️  Zerodha credentials not available for historical data")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kite Connect for historical data: {e}")
            return None
    
    def load_historical_data(self, instrument_token: int, from_date: datetime, 
                           to_date: datetime, interval: str = "minute") -> Optional[pd.DataFrame]:
        """Load historical data from Zerodha"""
        try:
            if not self.kite:
                logger.warning("⚠️  Kite Connect not initialized")
                return None
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                logger.warning(f"⚠️  No historical data returned for token {instrument_token}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"⚠️  Missing column in historical data: {col}")
                    return None
            
            logger.info(f"✅ Loaded {len(df)} records for token {instrument_token}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading historical data for token {instrument_token}: {e}")
            return None
    
    def load_nifty_historical_data(self, days: int = 365, interval: str = "minute") -> Optional[pd.DataFrame]:
        """Load Nifty 50 historical data"""
        try:
            # Nifty 50 instrument token
            nifty_token = 256265
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self.load_historical_data(nifty_token, from_date, to_date, interval)
            
            if data is not None:
                logger.info(f"✅ Loaded Nifty 50 historical data: {len(data)} records")
            else:
                logger.warning("⚠️  Failed to load Nifty 50 historical data")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading Nifty 50 historical data: {e}")
            return None
    
    def resample_data(self, df: pd.DataFrame, new_interval: str) -> pd.DataFrame:
        """Resample data to different timeframes"""
        try:
            if df.empty:
                return df
            
            # Resample OHLCV data
            resampled = df.resample(new_interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"✅ Resampled data from {len(df)} to {len(resampled)} records ({new_interval})")
            return resampled
            
        except Exception as e:
            logger.error(f"❌ Error resampling data: {e}")
            return df
    
    def load_multi_timeframe_data(self, instrument_token: int, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes"""
        try:
            # Load base minute data
            base_data = self.load_historical_data(
                instrument_token, 
                datetime.now() - timedelta(days=days), 
                datetime.now(), 
                "minute"
            )
            
            if base_data is None:
                return {}
            
            # Create multi-timeframe data
            multi_tf_data = {
                '1min': base_data,
                '5min': self.resample_data(base_data, '5T'),
                '15min': self.resample_data(base_data, '15T'),
                '1H': self.resample_data(base_data, '1H'),
                '4H': self.resample_data(base_data, '4H'),
                '1D': self.resample_data(base_data, '1D')
            }
            
            logger.info(f"✅ Loaded multi-timeframe data for {len(multi_tf_data)} timeframes")
            return multi_tf_data
            
        except Exception as e:
            logger.error(f"❌ Error loading multi-timeframe data: {e}")
            return {}

# Example usage with sample data generator
class SampleDataGenerator:
    """Generate sample data for testing when real data is not available"""
    
    @staticmethod
    def generate_sample_nifty_data(days: int = 30, interval: str = "1min") -> pd.DataFrame:
        """Generate realistic sample Nifty data"""
        try:
            # Calculate number of periods (ensure integer)
            if interval == "1min":
                periods = int(days * 24 * 60)  # minutes
            elif interval == "5min":
                periods = int(days * 24 * 12)  # 5-minute periods
            elif interval == "15min":
                periods = int(days * 24 * 4)   # 15-minute periods
            elif interval == "1H":
                periods = int(days * 24)       # hourly periods
            elif interval == "1D":
                periods = int(days)            # daily periods
            else:
                periods = int(days * 24 * 60)  # default to minutes
            
            # Ensure minimum periods
            periods = max(100, periods)
            
            # Generate date range
            end_date = datetime.now()
            if interval == "1min":
                dates = pd.date_range(end=end_date, periods=periods, freq='1min')
            elif interval == "5min":
                dates = pd.date_range(end=end_date, periods=periods, freq='5min')
            elif interval == "15min":
                dates = pd.date_range(end=end_date, periods=periods, freq='15min')
            elif interval == "1H":
                dates = pd.date_range(end=end_date, periods=periods, freq='1H')
            elif interval == "1D":
                dates = pd.date_range(end=end_date, periods=periods, freq='1D')
            else:
                dates = pd.date_range(end=end_date, periods=periods, freq='1min')
            
            # Generate realistic price series (random walk with trend)
            base_price = 18000
            returns = np.random.normal(0.0001, 0.01, periods)  # Small drift, volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLC data
            opens = prices[:-1]
            closes = prices[1:]
            
            # Generate high/low with realistic spreads
            highs = []
            lows = []
            volumes = []
            
            for i in range(len(opens)):
                open_price = opens[i]
                close_price = closes[i]
                
                # High and low based on open/close
                high = max(open_price, close_price) + np.random.uniform(0, 50)
                low = min(open_price, close_price) - np.random.uniform(0, 50)
                
                # Ensure high >= low
                high = max(high, low + 1)
                
                highs.append(high)
                lows.append(low)
                
                # Generate volume (more volume during market hours)
                hour = dates[i].hour
                if 9 <= hour <= 15:  # Market hours
                    volume = np.random.randint(500000, 2000000)
                else:
                    volume = np.random.randint(100000, 500000)
                volumes.append(volume)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates[1:])  # Skip first date since we use it for initial price
            
            logger.info(f"✅ Generated sample data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error generating sample data: {e}")
            # Return minimal DataFrame as fallback
            dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
            return pd.DataFrame({
                'open': [18000] * 100,
                'high': [18050] * 100,
                'low': [17950] * 100,
                'close': [18000 + i for i in range(100)],
                'volume': [1000000] * 100
            }, index=dates)

# Example usage
if __name__ == "__main__":
    # Test with sample data
    generator = SampleDataGenerator()
    sample_data = generator.generate_sample_nifty_data(days=7, interval="5min")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data head:\n{sample_data.head()}")
    
    # Test data loader (if credentials available)
    loader = HistoricalDataLoader()
    if loader.kite:
        print("Historical data loader initialized with Zerodha connection")
    else:
        print("Historical data loader ready (sample data mode)")
