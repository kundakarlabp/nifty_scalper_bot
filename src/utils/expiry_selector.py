# src/utils/expiry_selector.py
import datetime
import calendar
import logging

logger = logging.getLogger(__name__)

def get_next_weekly_expiry(ref_date: datetime.date = None) -> datetime.date:
    """
    Calculates the next weekly expiry date (Thursday) relative to a reference date.
    If the reference date is a Thursday, it returns the *next* Thursday.

    Args:
        ref_date (datetime.date, optional): The date to calculate from.
                                           Defaults to today's date.

    Returns:
        datetime.date: The date of the next weekly expiry (Thursday).
    """
    if ref_date is None:
        ref_date = datetime.date.today()

    # Find the weekday (0=Monday, 6=Sunday). Thursday is 3.
    weekday = ref_date.weekday()
    
    # Calculate days until next Thursday
    # If today is Thursday, we want next Thursday (add 7 days)
    days_ahead = (3 - weekday + 7) % 7
    if days_ahead == 0: # If it was calculated as 0 (meaning today is Thursday)
        days_ahead = 7  # Set to next Thursday
        
    next_expiry = ref_date + datetime.timedelta(days=days_ahead)
    logger.info(f"📅 Reference Date: {ref_date}, Next Weekly Expiry (Thursday): {next_expiry}")
    return next_expiry

def get_monthly_expiry(year: int, month: int) -> datetime.date:
    """
    Returns the last Thursday of the given month (monthly expiry).

    Args:
        year (int): The year.
        month (int): The month (1-12).

    Returns:
        datetime.date: The date of the last Thursday of the month.
    """
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime.date(year, month, last_day)
    
    # Iterate backwards until we find a Thursday (weekday 3)
    while last_date.weekday() != 3:
        last_date -= datetime.timedelta(days=1)
        
    return last_date

# Optional: Function to get the "nearest" expiry (weekly or monthly)
def get_nearest_expiry(ref_date: datetime.date = None) -> datetime.date:
    """
    Determines the nearest expiry date, preferring the next weekly expiry.
    If the next weekly expiry is after the current month's monthly expiry,
    the monthly expiry is chosen instead.

    Args:
        ref_date (datetime.date, optional): The date to calculate from.
                                           Defaults to today's date.

    Returns:
        datetime.date: The date of the nearest expiry (weekly or monthly).
    """
    if ref_date is None:
        ref_date = datetime.date.today()

    next_weekly = get_next_weekly_expiry(ref_date)
    current_monthly = get_monthly_expiry(ref_date.year, ref_date.month)
    
    # Calculate days difference
    days_to_weekly = (next_weekly - ref_date).days
    days_to_monthly = (current_monthly - ref_date).days
    
    # If monthly expiry is in the past or today, get next month's
    if days_to_monthly < 0:
        # Move to next month, handling year rollover
        if ref_date.month == 12:
            next_month = 1
            next_year = ref_date.year + 1
        else:
            next_month = ref_date.month + 1
            next_year = ref_date.year
        current_monthly = get_monthly_expiry(next_year, next_month)
        days_to_monthly = (current_monthly - ref_date).days

    # Choose the expiry that is closer in days
    if days_to_monthly < days_to_weekly:
        logger.info(f"📅 Nearest expiry selected: Monthly ({current_monthly})")
        return current_monthly
    else:
        logger.info(f"📅 Nearest expiry selected: Weekly ({next_weekly})")
        return next_weekly

# Example Usage:
# if __name__ == "__main__":
#     today = datetime.date.today()
#     print(f"Today: {today}")
#     print(f"Next Weekly Expiry: {get_next_weekly_expiry(today)}")
#     print(f"Current Monthly Expiry: {get_monthly_expiry(today.year, today.month)}")
#     print(f"Nearest Expiry: {get_nearest_expiry(today)}")
