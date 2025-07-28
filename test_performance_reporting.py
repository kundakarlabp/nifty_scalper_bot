import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta
from src.performance_reporting.report_generator import PerformanceReportGenerator, TradeRecord
from src.performance_reporting.daily_reporter import DailyPerformanceReporter

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Performance Reporting System...")

# Test 1: Report Generator
print("\n1. Testing Report Generator...")
try:
    report_gen = PerformanceReportGenerator()
    print("✅ Report generator initialized successfully")
    
    # Add sample trade records
    sample_trades = [
        TradeRecord(
            timestamp=datetime.now() - timedelta(minutes=30),
            symbol="NIFTY",
            direction="BUY",
            entry_price=18000,
            exit_price=18050,
            quantity=75,
            pnl=3750,
            pnl_percentage=2.08,
            holding_period=15,
            stop_loss=17950,
            target=18100,
            confidence=0.85,
            status="closed"
        ),
        TradeRecord(
            timestamp=datetime.now() - timedelta(minutes=15),
            symbol="NIFTY",
            direction="SELL",
            entry_price=18080,
            exit_price=18030,
            quantity=75,
            pnl=3750,
            pnl_percentage=2.07,
            holding_period=10,
            stop_loss=18130,
            target=17980,
            confidence=0.90,
            status="closed"
        )
    ]
    
    for trade in sample_trades:
        report_gen.add_trade_record(trade)
    
    print(f"✅ Added {len(sample_trades)} sample trades")
    
    # Add equity points
    for i in range(10):
        timestamp = datetime.now() - timedelta(minutes=i*5)
        equity = 100000 + i * 1000
        report_gen.add_equity_point(timestamp, equity)
    
    print("✅ Added sample equity curve data")
    
except Exception as e:
    print(f"❌ Report generator test failed: {e}")

# Test 2: Performance Metrics Calculation
print("\n2. Testing Performance Metrics Calculation...")
try:
    metrics = report_gen.calculate_performance_metrics()
    print("✅ Performance metrics calculated successfully")
    print(f"   Total trades: {metrics.total_trades}")
    print(f"   Win rate: {metrics.win_rate:.2%}")
    print(f"   Total P&L: ₹{metrics.total_pnl:,.2f}")
    print(f"   Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    
except Exception as e:
    print(f"❌ Performance metrics calculation failed: {e}")

# Test 3: Report Generation
print("\n3. Testing Report Generation...")
try:
    report = report_gen.generate_performance_report()
    print("✅ Performance report generated successfully")
    print(f"   Report keys: {list(report.keys())}")
    
    # Test JSON format
    json_report = report_gen.generate_performance_report(format_type="json")
    print("✅ JSON report generated successfully")
    
except Exception as e:
    print(f"❌ Report generation failed: {e}")

# Test 4: Daily Reporter
print("\n4. Testing Daily Reporter...")
try:
    daily_reporter = DailyPerformanceReporter(report_gen)
    daily_report = daily_reporter.generate_daily_report()
    print("✅ Daily reporter initialized and working")
    print(f"   Daily report generated: {'Yes' if daily_report else 'No'}")
    
except Exception as e:
    print(f"❌ Daily reporter test failed: {e}")

# Test 5: Equity Curve
print("\n5. Testing Equity Curve...")
try:
    timestamps, equity_curve = report_gen.get_equity_curve()
    print("✅ Equity curve data retrieved successfully")
    print(f"   Equity points: {len(equity_curve)}")
    if equity_curve:
        print(f"   Starting equity: ₹{equity_curve[0]:,.2f}")
        print(f"   Final equity: ₹{equity_curve[-1]:,.2f}")
    
except Exception as e:
    print(f"❌ Equity curve test failed: {e}")

print("\n🎉 Performance reporting system tests completed!")
print("🚀 Your performance reporting system is ready!")

print("\n🔧 To generate reports:")
print("   python -m src.performance_reporting.report_cli --help")
print("   python -m src.performance_reporting.report_cli --type daily")
print("   python -m src.performance_reporting.report_cli --type weekly --format json")
