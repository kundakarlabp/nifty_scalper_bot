# Nifty Scalper Streamlit Monitor

A read-only dashboard for checking the bot from an Android phone or desktop browser.

## What it shows

- Bot API availability
- Trading engine readiness
- Whether live orders are armed
- Broker authentication and balance status
- Position reconciliation status
- Current safety blockers
- Raw health diagnostics
- One-click CSV export of all available logs from 09:15 to 15:30 IST for a selected date

The dashboard does not place, modify, or cancel orders.

## Production bot endpoint

The current bot admin page is:

`http://15.206.3.6:8080/admin`

For the Streamlit dashboard, use only the base URL:

```toml
BOT_API_URL = "http://15.206.3.6:8080"
```

Do not append `/admin`. The dashboard automatically calls:

- `/livez`
- `/readyz`
- `/health/trading`

## Deploy on Streamlit Community Cloud

1. Push or merge this folder to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select this repository and branch.
4. Set the main file path to:

   `dashboard/streamlit_app.py`

5. In the app's **Secrets** section, add:

```toml
BOT_API_URL = "http://15.206.3.6:8080"
BOT_ADMIN_PASSWORD = "YOUR_EXISTING_ADMIN_PASSWORD"
```

`BOT_ADMIN_PASSWORD` is used only server-side by the Streamlit app to authenticate to the existing admin log-download endpoint. It is not displayed in the dashboard or browser.

If a reverse proxy later protects the health endpoints with a bearer token, also add:

```toml
BOT_DASHBOARD_TOKEN = "replace-with-a-long-random-token"
```

6. Deploy the Streamlit app.

## Market-hours CSV download

1. Open the Streamlit dashboard.
2. Choose the trading date.
3. Tap **Prepare market-hours CSV**.
4. Tap **Download one CSV file**.

The generated UTF-8 CSV contains:

- `timestamp_ist`
- `message`

Only timestamped rows between **09:15:00 and 15:30:00 IST** are included. The dashboard requests up to the latest 20,000 available log lines from the existing admin service, then filters them by date and market hours.

## Android access

1. Open the generated Streamlit URL in Chrome.
2. Sign in if the Streamlit deployment is private.
3. Open Chrome's menu and tap **Add to Home screen**.
4. Launch it from the home-screen icon like an app.

## Local run

From the repository root:

```bash
python -m pip install -r dashboard/requirements.txt
BOT_API_URL=http://15.206.3.6:8080 BOT_ADMIN_PASSWORD='your-password' streamlit run dashboard/streamlit_app.py
```

On Windows PowerShell:

```powershell
$env:BOT_API_URL="http://15.206.3.6:8080"
$env:BOT_ADMIN_PASSWORD="your-password"
streamlit run dashboard/streamlit_app.py
```

## Security notes

- Keep the dashboard read-only.
- Never commit `BOT_ADMIN_PASSWORD` to GitHub; store it only in Streamlit Secrets.
- Do not expose broker API keys, access tokens, Telegram tokens, or `.env` contents.
- Prefer a private Streamlit deployment or place authentication in front of the app.
- The current production endpoint uses plain HTTP on a public IP; migrate it to HTTPS before broader exposure.
- The current bot health endpoints reveal operational status but not broker credentials.
