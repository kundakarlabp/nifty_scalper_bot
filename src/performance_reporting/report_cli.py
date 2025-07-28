import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
from datetime import datetime, timedelta
from src.performance_reporting.report_generator import PerformanceReportGenerator
from src.performance_reporting.daily_reporter import DailyPerformanceReporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class PerformanceReportCLI:
    """Command-line interface for performance reporting"""
    
    def __init__(self):
        self.report_generator = PerformanceReportGenerator()
        self.daily_reporter = DailyPerformanceReporter(self.report_generator)
    
    def generate_report(self, args):
        """Generate performance report based on arguments"""
        try:
            logger.info("Generating performance report...")
            
            # For demo purposes, we'll generate some sample data
            self._generate_sample_data()
            
            # Calculate metrics
            metrics = self.report_generator.calculate_performance_metrics()
            
            # Generate report
            if args.type == "daily":
                report = self.daily_reporter.generate_daily_report()
            elif args.type == "weekly":
                report = self.daily_reporter.generate_weekly_report()
            elif args.type == "period":
                report = self.daily_reporter.generate_performance_summary(args.days)
            else:
                report = self.report_generator.generate_performance_report()
            
            # Display results
            self._display_report(report, args.format)
            
            # Save if requested
            if args.output:
                self._save_report(report, args.output)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def _generate_sample_data(self):
        """Generate sample trade data for demonstration"""
        try:
            import random
            from src.performance_reporting.report_generator import TradeRecord
            
            # Generate 20 sample trades
            for i in range(20):
                trade = TradeRecord(
                    timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1000)),
                    symbol="NIFTY",
                    direction=random.choice(["BUY", "SELL"]),
                    entry_price=random.uniform(17500, 18500),
                    exit_price=random.uniform(17500, 18500),
                    quantity=random.choice([75, 150, 225]),
                    pnl=random.uniform(-5000, 10000),
                    pnl_percentage=random.uniform(-5, 15),
                    holding_period=random.randint(5, 120),
                    stop_loss=random.uniform(17400, 18400),
                    target=random.uniform(17600, 18600),
                    confidence=random.uniform(0.6, 0.95),
                    status="closed"
                )
                self.report_generator.add_trade_record(trade)
            
            # Add some equity points
            base_equity = 100000
            for i in range(100):
                timestamp = datetime.now() - timedelta(minutes=i*10)
                equity = base_equity + random.uniform(-5000, 15000)
                self.report_generator.add_equity_point(timestamp, equity)
            
            logger.info("✅ Generated sample data for demonstration")
            
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
    
    def _display_report(self, report: dict, format_type: str = "table"):
        """Display report in specified format"""
        try:
            if format_type == "json":
                print(json.dumps(report, indent=2, default=str))
            else:
                self._display_table_report(report)
        except Exception as e:
            logger.error(f"Error displaying report: {e}")
    
    def _display_table_report(self, report: dict):
        """Display report in table format"""
        try:
            print("\n" + "="*60)
            print("�� PERFORMANCE REPORT")
            print("="*60)
            
            # Summary section
            summary = report.get('summary', {})
            print(f"📈 Total Trades:     {summary.get('total_trades', 0)}")
            print(f"✅ Winning Trades:   {summary.get('winning_trades', 0)}")
            print(f"❌ Losing Trades:    {summary.get('losing_trades', 0)}")
            print(f"🎯 Win Rate:         {summary.get('win_rate', 0)}%")
            print(f"💰 Total P&L:        ₹{summary.get('total_pnl', 0):,.2f}")
            print(f"📊 Average P&L:      ₹{summary.get('average_pnl', 0):,.2f}")
            print(f"🚀 Max P&L:          ₹{summary.get('max_pnl', 0):,.2f}")
            print(f"�� Min P&L:          ₹{summary.get('min_pnl', 0):,.2f}")
            
            # Risk metrics
            risk_metrics = report.get('risk_metrics', {})
            print(f"\n⚖️  Max Drawdown:     {risk_metrics.get('max_drawdown', 0)}%")
            print(f"⚡ Sharpe Ratio:     {risk_metrics.get('sharpe_ratio', 0)}")
            print(f"🔥 Sortino Ratio:    {risk_metrics.get('sortino_ratio', 0)}")
            print(f"💰 Profit Factor:    {risk_metrics.get('profit_factor', 0)}")
            print(f"📊 Calmar Ratio:     {risk_metrics.get('calmar_ratio', 0)}")
            print(f"📈 Volatility:       {risk_metrics.get('volatility', 0)}")
            
            # Trading metrics
            trading_metrics = report.get('trading_metrics', {})
            print(f"\n⏱️  Max Consecutive Wins:  {trading_metrics.get('max_consecutive_wins', 0)}")
            print(f"😞 Max Consecutive Losses: {trading_metrics.get('max_consecutive_losses', 0)}")
            print(f"⏰ Avg Holding Period:     {trading_metrics.get('average_holding_period', 0)} mins")
            print(f"📦 Avg Position Size:      {trading_metrics.get('average_position_size', 0)}")
            print(f"🎯 Risk/Reward Ratio:      {trading_metrics.get('risk_reward_ratio', 0)}")
            print(f"💵 Total Volume:           ₹{trading_metrics.get('total_volume', 0):,.2f}")
            
            # Recent trades
            recent_trades = report.get('recent_trades', [])
            if recent_trades:
                print(f"\n📋 RECENT TRADES (last {len(recent_trades)}):")
                print("-" * 80)
                print(f"{'Time':<12} {'Symbol':<8} {'Dir':<4} {'Entry':<8} {'Exit':<8} {'P&L':<10}")
                print("-" * 80)
                
                for trade in recent_trades[:5]:  # Show first 5
                    timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%m-%d %H:%M') if isinstance(trade['timestamp'], str) else trade['timestamp'].strftime('%m-%d %H:%M')
                    print(f"{timestamp:<12} "
                          f"{trade['symbol']:<8} "
                          f"{trade['direction']:<4} "
                          f"{trade['entry_price']:<8.2f} "
                          f"{trade['exit_price']:<8.2f} "
                          f"₹{trade['pnl']:<9.2f}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error displaying table report: {e}")
    
    def _save_report(self, report: dict, output_file: str):
        """Save report to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✅ Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

def main():
    """Main entry point for performance reporting CLI"""
    parser = argparse.ArgumentParser(description='Performance Reporting Tool')
    parser.add_argument('--type', default='summary',
                       choices=['summary', 'daily', 'weekly', 'period'],
                       help='Type of report to generate')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for period report')
    parser.add_argument('--format', default='table',
                       choices=['table', 'json'],
                       help='Output format')
    parser.add_argument('--output', help='Output file for report (JSON format)')
    
    args = parser.parse_args()
    
    # Generate report
    cli = PerformanceReportCLI()
    cli.generate_report(args)

if __name__ == "__main__":
    main()
