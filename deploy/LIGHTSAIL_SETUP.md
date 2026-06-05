# Nifty Scalper Bot — Lightsail Setup (one time, then browser-only)

You only use the terminal **once**. After this, you manage everything from a web
dashboard in your browser — credentials, daily access token, logs, restart.

Your Lightsail static IP: **15.206.3.6** (this is what Zerodha must allowlist.)

---

## STEP 1 — Allowlist the IP on Zerodha (do this first, in your browser)

1. Go to https://developers.kite.trade
2. Open your app.
3. In the **Allowed IPs** field, enter: `15.206.3.6`
4. Save.

---

## STEP 2 — Open the firewall for the dashboard (in Lightsail website)

1. Lightsail → your instance **Ubuntu-2** → **Networking** tab.
2. Under **IPv4 Firewall**, click **Add rule** and add BOTH:
   - **HTTPS:** Custom / TCP / Port **443**  (secure dashboard — recommended)
   - **Dashboard:** Custom / TCP / Port **8080**  (plain fallback)
3. Save.

---

## STEP 3 — One-time install (paste ONCE into the Lightsail terminal)

Open the instance terminal ("Connect using SSH" button) and paste this whole block.
It installs Python, downloads the bot, sets up the dashboard, adds HTTPS, and starts
it as a background service that auto-restarts and survives reboots.

```bash
curl -fsSL https://raw.githubusercontent.com/kundakarlabp/nifty_scalper_bot/main/deploy/lightsail_setup.sh | bash
```

When it finishes it prints your dashboard URL and admin password.

---

## STEP 4 — Use the dashboard (browser, no terminal ever again)

Open the **secure** link the script printed: **https://15.206.3.6/admin**
(A one-time browser warning appears because it is your own server's certificate —
click **Advanced → Proceed**. Plain fallback: http://15.206.3.6:8080/admin)

- Sign in with the admin password the script printed.
- Enter your Zerodha API key, API secret, access token, Telegram details. Save.
- Click **Restart Bot**.

### Turning live trading ON/OFF (one click)
At the top of the dashboard there is a **Live Trading** banner:
- **OFF (SHADOW)** = analysing only, no real orders. You start here.
- Click **Turn ON Live Trading** (it asks you to confirm) to place real orders.
- Click **Switch to SHADOW** anytime to stop real trading instantly.
The toggle sets both required settings together and restarts the bot for you.

### Every morning (one step, in the browser)
- Log in to Kite and copy the **request token** from the redirect URL
  (the `request_token=...` part after you authorize).
- Open the dashboard → **Daily Token** → paste it into **Request Token** →
  **Update token & restart**. The bot fetches the access token automatically.
  (If you already have an access token, paste that in the second box instead.)

### Automatic updates (no terminal ever)
- Any change pushed to the GitHub repo is pulled to the server and the bot is
  restarted **automatically within ~2 minutes** (a background timer does this).
- You can also force it instantly: dashboard → **Update from GitHub**.

### Logs
- The **Logs** page auto-refreshes (every 5s by default; selectable 3/5/10s or off).
- Shows clean IST time + message only (host name, PID and UTC line removed).
- Filter by text, choose line count, and Download as .txt/.json/.csv.

---

## If something looks wrong
- Dashboard won't open → re-check STEP 2 (port 8080 firewall rule).
- Orders rejected with "No IPs configured" → re-check STEP 1 (IP `15.206.3.6` saved on Zerodha).
- Need to re-run setup → it is safe to run STEP 3 again; it updates in place.
