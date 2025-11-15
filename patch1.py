import sys
file_path = sys.argv[1]
with open(file_path, 'r') as f:
    content = f.read()

# Remove or make conditional the market hours check that blocks trading
content = content.replace(
    'if not self.is_market_hours() and not os.getenv("SESSION_ALLOW_OUT_OF_HOURS"):\n        return False',
    'if not self.is_market_hours() and not os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower() in ("1", "true"):\n        logger.debug(f"Market hours: {self.is_market_hours()}, override: {os.getenv(\'SESSION_ALLOW_OUT_OF_HOURS\')}")\n        if not self.is_market_hours():\n            return False'
)

with open(file_path, 'w') as f:
    f.write(content)
