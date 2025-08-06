 # Schedule the data fetching and processing task
 # Adjust the frequency (e.g., '1' minute) according to your strategy's needs.
 # process_bar checks for Config.TIME_FILTER_START/END, so frequent checks are usually okay.
 schedule.every(1).minutes.do(self.fetch_and_process_data)
 logger.info("Scheduled fetch_and_process_data to run every 1 minute.")
