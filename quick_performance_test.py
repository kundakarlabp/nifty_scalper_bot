import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta
from src.performance_reporting.report_generator import PerformanceReportGenerator, TradeRecord

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🚀 Quick Performance Reporting Test")
print("=" * 40)

try:
    # Initialize report generator
    print("🔧 Initializing performance report generator...")
    report_gen = PerformanceReportGenerator()
    
    # Generate sample trades
    print("📊 Generating sample trade data...")
    import random
    
    for i in range(10):
        trade = TradeRecord(
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 300)),
            symbol="NIFTY",
            direction=random.choice(["BUY", "SELL"]),
            entry_price=random.uniform(17800, 18200),
            exit_price=random.uniform(17800, 18200),
            quantity=random.choice([75, 150]),
            pnl=random.uniform(-2000, 5000),
            pnl_percentage=random.uniform(-2, 5),
            holding_period=random.randint(5, 60),
            stop_loss=random.uniform(17700, 18100),
            target=random.uniform(17900, 18300),
            confidence=random.uniform(0.7, 0.95),
            status="closed"
        )
        report_gen.add_trade_record(trade)
    
    print("✅ Generated 10 sample trades")
    
    # Add equity curve data
    base_equity = 100000
    for i in range(50):
        timestamp = datetime.now() - timedelta(minutes=i*10)
        equity = base_equity + random.uniform(-5000, 10000)
        report_gen.add_equity_point(timestamp, equity)
    
    print("✅ Added equity curve data")
    
    # Calculate metrics
    print("🧮 Calculating performance metrics...")
    metrics = report_gen.calculate_performance_metrics()
    
    # Generate report
    print("📋 Generating performance report...")
    report = report_gen.generate_performance_report()
    
    # Display key metrics
    print("\n" + "="*50)
    print("📈 PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total Trades:     {metrics.total_trades}")
    print(f"Win Rate:         {metrics.win_rate:.1%}")
    print(f"Total P&L:        ₹{metrics.total_pnl:,.2f}")
    print(f"Average P&L:      ₹{metrics.average_pnl:,.2f}")
    print(f"Max Drawdown:     {metrics.max_drawdown:.1%}")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"Profit Factor:    {metrics.profit_factor:.2f}")
    
    print("\n✅ Quick performance test completed!")
    print("�� Performance reporting system is working!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🔧 To run full performance reports:")
print("   python -m src.performance_reporting.report_cli --help")
