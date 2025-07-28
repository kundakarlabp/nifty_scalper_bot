import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from src.performance_reporting.report_generator import PerformanceReportGenerator, TradeRecord

logger = logging.getLogger(__name__)

class DailyPerformanceReporter:
    """Generate daily performance reports"""
    
    def __init__(self, report_generator: PerformanceReportGenerator):
        self.report_generator = report_generator
    
    def generate_daily_report(self, date: datetime = None) -> Dict:
        """Generate daily performance report"""
        try:
            if date is None:
                date = datetime.now()
            
            # Filter trades for the day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            daily_trades = [
                trade for trade in self.report_generator.trade_records
                if start_of_day <= trade.timestamp <= end_of_day
            ]
            
            # Create temporary report generator for daily data
            daily_reporter = PerformanceReportGenerator()
            for trade in daily_trades:
                daily_reporter.add_trade_record(trade)
            
            # Generate daily report
            daily_metrics = daily_reporter.calculate_performance_metrics()
            daily_report = daily_reporter.generate_performance_report()
            
            logger.info(f"Daily report generated for {date.strftime('%Y-%m-%d')}")
            return daily_report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {}
    
    def generate_weekly_report(self, week_start: datetime = None) -> Dict:
        """Generate weekly performance report"""
        try:
            if week_start is None:
                # Get start of current week (Monday)
                today = datetime.now()
                week_start = today - timedelta(days=today.weekday())
            
            week_end = week_start + timedelta(days=6)
            
            # Filter trades for the week
            weekly_trades = [
                trade for trade in self.report_generator.trade_records
                if week_start <= trade.timestamp <= week_end
            ]
            
            # Create temporary report generator for weekly data
            weekly_reporter = PerformanceReportGenerator()
            for trade in weekly_trades:
                weekly_reporter.add_trade_record(trade)
            
            # Generate weekly report
            weekly_metrics = weekly_reporter.calculate_performance_metrics()
            weekly_report = weekly_reporter.generate_performance_report()
            
            logger.info(f"Weekly report generated for {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
            return weekly_report
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {}
    
    def generate_performance_summary(self, days: int = 30) -> Dict:
        """Generate performance summary for specified period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Filter trades for the period
            period_trades = [
                trade for trade in self.report_generator.trade_records
                if start_date <= trade.timestamp <= end_date
            ]
            
            # Create temporary report generator for period data
            period_reporter = PerformanceReportGenerator()
            for trade in period_trades:
                period_reporter.add_trade_record(trade)
            
            # Generate period report
            period_metrics = period_reporter.calculate_performance_metrics()
            period_report = period_reporter.generate_performance_report()
            
            logger.info(f"Period report generated for last {days} days")
            return period_report
            
        except Exception as e:
            logger.error(f"Error generating period report: {e}")
            return {}

class TelegramPerformanceReporter:
    """Send performance reports via Telegram"""
    
    def __init__(self, telegram_bot):
        self.telegram_bot = telegram_bot
    
    def send_daily_report(self, report: Dict, chat_id: str = None):
        """Send daily performance report via Telegram"""
        try:
            if not report:
                logger.warning("Empty report provided")
                return
            
            # Format the report as a readable message
            message = self._format_daily_report(report)
            
            # Send via Telegram
            if chat_id:
                self.telegram_bot.send_message(chat_id=chat_id, text=message)
            else:
                # Use default chat ID from telegram bot
                self.telegram_bot.send_message(text=message)
            
            logger.info("Daily performance report sent via Telegram")
            
        except Exception as e:
            logger.error(f"Error sending daily report via Telegram: {e}")
    
    def _format_daily_report(self, report: Dict) -> str:
        """Format daily report as Telegram message"""
        try:
            summary = report.get('summary', {})
            risk_metrics = report.get('risk_metrics', {})
            trading_metrics = report.get('trading_metrics', {})
            
            message = f"""
📊 **DAILY PERFORMANCE REPORT** 📊
📅 {datetime.now().strftime('%Y-%m-%d')}

💰 **Trading Summary**
• Trades: {summary.get('total_trades', 0)}
• Win Rate: {summary.get('win_rate', 0)}%
• Total P&L: ₹{summary.get('total_pnl', 0):,.2f}
• Avg P&L: ₹{summary.get('average_pnl', 0):,.2f}

⚖️ **Risk Metrics**
• Max Drawdown: {risk_metrics.get('max_drawdown', 0)}%
• Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0)}
• Sortino Ratio: {risk_metrics.get('sortino_ratio', 0)}
• Profit Factor: {risk_metrics.get('profit_factor', 0)}

📈 **Trading Metrics**
• Consecutive Wins: {trading_metrics.get('max_consecutive_wins', 0)}
• Consecutive Losses: {trading_metrics.get('max_consecutive_losses', 0)}
• Avg Holding: {trading_metrics.get('average_holding_period', 0)} mins
• Risk/Reward: {trading_metrics.get('risk_reward_ratio', 0)}

🚀 Keep up the great work!
            """
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting daily report: {e}")
            return "Daily performance report"

# Example usage
if __name__ == "__main__":
    print("Daily Performance Reporter ready!")
    print("Available classes:")
    print("- DailyPerformanceReporter: For daily/weekly reports")
    print("- TelegramPerformanceReporter: For Telegram notifications")
