import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class SimpleScalpingStrategy:
    def __init__(self, stop_loss_points: int = 20, target_points: int = 40):
        self.stop_loss_points = stop_loss_points
        self.target_points = target_points
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
        # Simple Moving Average
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal"""
        if len(data) < 50:
            return None
        
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Buy conditions
        buy_conditions = [
            current['close'] > current['sma_20'],
            current['sma_20'] > current['sma_50'],
            current['rsi'] < 70,  # Not overbought
            current['close'] < current['bb_upper'],  # Below upper BB
        ]
        
        # Sell conditions
        sell_conditions = [
            current['close'] < current['sma_20'],
            current['sma_20'] < current['sma_50'],
            current['rsi'] > 30,  # Not oversold
            current['close'] > current['bb_lower'],  # Above lower BB
        ]
        
        if all(buy_conditions):
            return {
                'signal': 'BUY',
                'entry_price': current['close'],
                'stop_loss': current['close'] - self.stop_loss_points,
                'target': current['close'] + self.target_points,
                'confidence': 0.8
            }
        elif all(sell_conditions):
            return {
                'signal': 'SELL',
                'entry_price': current['close'],
                'stop_loss': current['close'] + self.stop_loss_points,
                'target': current['close'] - self.target_points,
                'confidence': 0.8
            }
        
        return None

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'close': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
        'high': [18010 + i*10 + np.random.randn()*50 for i in range(100)],
        'low': [17990 + i*10 + np.random.randn()*50 for i in range(100)],
        'open': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
        'volume': [1000000 + np.random.randint(-100000, 100000) for i in range(100)]
    })
    
    strategy = SimpleScalpingStrategy()
    signal = strategy.generate_signal(sample_data)
    print(f"Generated signal: {signal}")
