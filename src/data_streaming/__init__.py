"""Data streaming and real‑time trading logic."""
# Schedule the data fetching and processing task
# Adjust the frequency (e.g., '1' minute) as needed for your strategy
schedule.every(1).minutes.do(self.process_data_and_trade)
logger.info("Scheduled process_data_and_trade to run every 1 minute.")
