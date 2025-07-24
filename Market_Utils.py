import pytz
from datetime import datetime, time
from config import Config

def is_market_open() -> bool:
    """Check if market is currently open"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE)
        market_end = time(Config.MARKET_END_HOUR, Config.MARKET_END_MINUTE)
        current_time = now.time()
        
        return market_start <= current_time <= market_end
        
    except Exception as e:
        return False

def get_market_status() -> str:
    """Get market status emoji and text"""
    if is_market_open():
        return "🟢 OPEN"
    else:
        return "🔴 CLOSED"

def time_until_market_open() -> str:
    """Get time until market opens"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        if now.weekday() >= 5:  # Weekend
            # Calculate days until Monday
            days_until_monday = 7 - now.weekday()
            return f"Opens Monday at 9:15 AM (in {days_until_monday} days)"
        
        market_start = now.replace(hour=Config.MARKET_START_HOUR, 
                                 minute=Config.MARKET_START_MINUTE, 
                                 second=0, microsecond=0)
        
        if now.time() < time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE):
            # Market opens today
            diff = market_start - now
            hours, remainder = divmod(diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            return f"Opens in {hours}h {minutes}m"
        else:
            # Market opens tomorrow
            return "Opens tomorrow at 9:15 AM"
            
    except Exception:
        return "Unknown"
