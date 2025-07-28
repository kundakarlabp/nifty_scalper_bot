import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_streaming.realtime_trader import RealTimeTrader

def monitor_system():
    """Monitor the real-time trading system"""
    print("🔍 Monitoring Real-time Trading System...")
    print("=" * 50)
    
    # This would normally access the running trader instance
    # For demo, we'll create a new instance to show status
    trader = RealTimeTrader()
    
    try:
        while True:
            status = trader.get_trading_status()
            
            print(f"\n📊 System Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Trading: {'✅ ACTIVE' if status['is_trading'] else '❌ STOPPED'}")
            print(f"   WebSocket: {'✅ CONNECTED' if status['streaming_status']['connected'] else '❌ DISCONNECTED'}")
            print(f"   Active Signals: {status['active_signals']}")
            print(f"   Trading Instruments: {status['trading_instruments']}")
            
            # Risk status
            risk_status = status['risk_status']
            print(f"\n💰 Risk Management:")
            print(f"   Account Size: ₹{risk_status['account_size']:,.2f}")
            print(f"   Daily P&L: ₹{risk_status['daily_pnl']:,.2f}")
            print(f"   Drawdown: {risk_status['drawdown_percentage']:.2f}%")
            print(f"   Positions: {risk_status['current_positions']}/{risk_status['max_positions']}")
            
            # Buffer status
            if status['processor_status']:
                print(f"\n💾 Data Buffers:")
                for token, buffer_info in status['processor_status'].items():
                    print(f"   Token {token}: {buffer_info['buffer_size']} ticks")
            
            print("\n" + "-" * 50)
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_system()
