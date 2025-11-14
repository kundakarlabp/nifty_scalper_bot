# NIFTY Options Sync (1-Year Rolling)

Downloads 1 year of NIFTY OPTIDX (CE & PE) from NSE F&O bhavcopies and keeps CSVs updated.

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_sync.sh   # builds data/*.csv
```

Outputs:

- data/nifty_options_all.csv
- data/nifty_options_weekly.csv
- data/nifty_options_monthly.csv


Rebuild / Verbose

```bash
# Force rebuild CSVs from cached zips
python scripts/sync_nifty_options.py --force-rebuild --outdir data

# Verbose logs
VERBOSE=1 bash scripts/run_sync.sh
```

Automate Daily (cron)

```bash
# Edit your crontab
crontab -e

# Run every weekday at 19:00 IST (adjust as needed)
0 19 * * 1-5 cd /path/to/your/repo && /usr/bin/env bash scripts/run_sync.sh >> sync.log 2>&1
```

Notes:

Uses NSE F&O bhavcopy zips located at /content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip.

Idempotent: cached zips live under data/cache/fo/YYYY/MON/.

Weekly vs Monthly classification: Monthly = last Thursday of the month.


## IMPLEMENTATION NOTES
- Be strict about CSV headers and sorting order.
- Handle missing days (holidays) silently.
- Ensure polite headers and sensible retries to reduce NSE blocking.
- Keep everything idempotent; never delete caches automatically.

## FINAL CHECK
After generating the files, run:
```bash
python3 scripts/sync_nifty_options.py --outdir data --days 365 --verbose
```

Ensure the three CSVs are produced with non-zero rows on normal trading periods.
