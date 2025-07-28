import logging
from datetime import datetime
from typing import Dict, List, Optional
from database.models import db_manager, TradeModel, PerformanceMetricModel
from src.performance_reporting.report_generator import TradeRecord, PerformanceMetrics

logger = logging.getLogger(__name__)

class PerformanceDatabase:
    """Handle performance data persistence"""
    
    def __init__(self):
        self.db_manager = db_manager
    
    def save_trade_record(self, trade_record: TradeRecord) -> bool:
        """Save trade record to database"""
        session = self.db_manager.get_session()
        try:
            trade_model = TradeModel(
                symbol=trade_record.symbol,
                direction=trade_record.direction,
                quantity=trade_record.quantity,
                entry_price=trade_record.entry_price,
                exit_price=trade_record.exit_price,
                stop_loss=trade_record.stop_loss,
                target=trade_record.target,
                entry_time=trade_record.timestamp,
                exit_time=datetime.now(),  # This should be actual exit time
                pnl=trade_record.pnl,
                pnl_percentage=trade_record.pnl_percentage,
                status=trade_record.status
            )
            
            session.add(trade_model)
            session.commit()
            logger.info(f"Trade record saved to database: {trade_record.symbol}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade record: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def save_performance_metrics(self, metrics: PerformanceMetrics, period: str = "daily") -> bool:
        """Save performance metrics to database"""
        session = self.db_manager.get_session()
        try:
            # Convert metrics to database records
            metric_mappings = {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'total_pnl': metrics.total_pnl,
                'average_pnl': metrics.average_pnl,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'profit_factor': metrics.profit_factor
            }
            
            for metric_name, value in metric_mappings.items():
                metric_model = PerformanceMetricModel(
                    metric_name=metric_name,
                    value=float(value),
                    period=period,
                    timestamp=datetime.now()
                )
                session.add(metric_model)
            
            session.commit()
            logger.info(f"Performance metrics saved to database ({period})")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving performance metrics: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_recent_trades(self, limit: int = 50) -> List[TradeRecord]:
        """Get recent trades from database"""
        session = self.db_manager.get_session()
        try:
            trade_models = session.query(TradeModel).order_by(
                TradeModel.entry_time.desc()
            ).limit(limit).all()
            
            trade_records = []
            for trade_model in trade_models:
                trade_record = TradeRecord(
                    timestamp=trade_model.entry_time,
                    symbol=trade_model.symbol,
                    direction=trade_model.direction,
                    entry_price=trade_model.entry_price,
                    exit_price=trade_model.exit_price or 0,
                    quantity=trade_model.quantity,
                    pnl=trade_model.pnl or 0,
                    pnl_percentage=trade_model.pnl_percentage or 0,
                    holding_period=0,  # This would need to be calculated
                    stop_loss=trade_model.stop_loss,
                    target=trade_model.target,
                    confidence=0.8,  # Default confidence
                    status=trade_model.status or 'closed'
                )
                trade_records.append(trade_record)
            
            logger.info(f"Retrieved {len(trade_records)} recent trades from database")
            return trade_records
            
        except Exception as e:
            logger.error(f"Error retrieving recent trades: {e}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def get_performance_history(self, period: str = "daily", days: int = 30) -> List[Dict]:
        """Get performance history from database"""
        session = self.db_manager.get_session()
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            metrics = session.query(PerformanceMetricModel).filter(
                PerformanceMetricModel.period == period,
                PerformanceMetricModel.timestamp >= cutoff_date
            ).order_by(PerformanceMetricModel.timestamp.desc()).all()
            
            # Group by timestamp
            history = {}
            for metric in metrics:
                timestamp_str = metric.timestamp.strftime('%Y-%m-%d')
                if timestamp_str not in history:
                    history[timestamp_str] = {}
                history[timestamp_str][metric.metric_name] = metric.value
            
            logger.info(f"Retrieved performance history for {len(history)} periods")
            return list(history.values())
            
        except Exception as e:
            logger.error(f"Error retrieving performance history: {e}")
            return []
        finally:
            self.db_manager.close_session(session)

# Example usage
if __name__ == "__main__":
    print("Performance Database Integration ready!")
    print("Import and use: from src.performance_reporting.database_integration import PerformanceDatabase")
