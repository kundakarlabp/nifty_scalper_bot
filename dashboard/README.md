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

The dashboard does not place, modify, or cancel orders.

## Deploy on Streamlit Community Cloud

1. Push/merge this folder to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select this repository and branch.
4. Set the main file path to:

   `dashboard/streamlit_app.py`

5. In the app's **Secrets** section, add:

```toml
BOT_API_URL = "https://YOUR-BOT-SERVICE.example.com"
```

The URL must be the public base URL of the running bot, without a trailing slash.

If a reverse proxy later protects the health endpoints with a bearer token, also add:

```toml
BOT_DASHBOARD_TOKEN = "replace-with-a-long-random-token"
```

6. Deploy the Streamlit app.

## Android access

1. Open the generated Streamlit URL in Chrome.
2. Sign in if the Streamlit deployment is private.
3. Open Chrome's menu and tap **Add to Home screen**.
4. Launch it from the home-screen icon like an app.

## Local run

From the repository root:

```bash
python -m pip install -r dashboard/requirements.txt
BOT_API_URL=http://localhost:8080 streamlit run dashboard/streamlit_app.py
```

On Windows PowerShell:

```powershell
$env:BOT_API_URL="http://localhost:8080"
streamlit run dashboard/streamlit_app.py
```

## Security notes

- Keep the dashboard read-only.
- Do not expose broker API keys, access tokens, Telegram tokens, or `.env` contents.
- Prefer a private Streamlit deployment or place authentication in front of the app.
- The current bot health endpoints reveal operational status but not broker credentials.
