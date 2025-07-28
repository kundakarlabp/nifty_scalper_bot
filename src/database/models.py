from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    strategy = Column(String(50))
    confidence = Column(Float)

class SignalRecord(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY/SELL
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    executed = Column(Boolean, default=False)
    execution_time = Column(DateTime)

class PerformanceMetric(Base):
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    period = Column(String(20))  # DAILY, WEEKLY, MONTHLY

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///trading_data.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
        logger.info("Database manager initialized")
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing database session: {e}")

# Global database manager instance
db_manager = DatabaseManager()
