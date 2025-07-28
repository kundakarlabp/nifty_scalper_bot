from src.database.models import db_manager, Trade, SignalRecord, PerformanceMetric
from sqlalchemy.orm import Session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseOperations:
    def __init__(self):
        self.db_manager = db_manager
    
    def save_trade(self, trade_data: dict) -> int:
        """Save a new trade to database"""
        session = self.db_manager.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            logger.info(f"Trade saved with ID: {trade.id}")
            return trade.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def update_trade_exit(self, trade_id: int, exit_price: float, exit_time: datetime = None) -> bool:
        """Update trade with exit information"""
        session = self.db_manager.get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_time = exit_time or datetime.utcnow()
                trade.status = 'CLOSED'
                
                # Calculate P&L
                if trade.direction == 'BUY':
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.quantity
                
                trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
                
                session.commit()
                logger.info(f"Trade {trade_id} updated with exit price: {exit_price}")
                return True
            else:
                logger.error(f"Trade {trade_id} not found")
                return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating trade exit: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def save_signal(self, signal_data: dict) -> int:
        """Save a signal record to database"""
        session = self.db_manager.get_session()
        try:
            signal = SignalRecord(**signal_data)
            session.add(signal)
            session.commit()
            session.refresh(signal)
            logger.info(f"Signal saved with ID: {signal.id}")
            return signal.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving signal: {e}")
            return None
        finally:
            self.db_manager.close_session(session)
    
    def get_open_trades(self) -> list:
        """Get all open trades"""
        session = self.db_manager.get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == 'OPEN').all()
            logger.info(f"Found {len(trades)} open trades")
            return trades
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def save_performance_metric(self, metric_name: str, value: float, period: str = None) -> bool:
        """Save a performance metric"""
        session = self.db_manager.get_session()
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                period=period
            )
            session.add(metric)
            session.commit()
            logger.info(f"Performance metric saved: {metric_name} = {value}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving performance metric: {e}")
            return False
        finally:
            self.db_manager.close_session(session)
    
    def get_recent_signals(self, hours: int = 24) -> list:
        """Get signals from recent hours"""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.utcnow().replace(tzinfo=None) - timedelta(hours=hours)
            signals = session.query(SignalRecord).filter(
                SignalRecord.timestamp >= cutoff_time
            ).all()
            return signals
        except Exception as e:
            logger.error(f"Error fetching recent signals: {e}")
            return []
        finally:
            self.db_manager.close_session(session)

# Import timedelta here to avoid circular imports
from datetime import timedelta
