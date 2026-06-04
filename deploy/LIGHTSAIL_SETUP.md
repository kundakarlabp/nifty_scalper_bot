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
2. Under **IPv4 Firewall**, click **Add rule**:
   - Application: **Custom**
   - Protocol: **TCP**
   - Port: **8080**
3. Save.

---

## STEP 3 — One-time install (paste ONCE into the Lightsail terminal)

Open the instance terminal ("Connect using SSH" button) and paste this whole block.
It installs Python, downloads the bot, sets up the dashboard, and starts it as a
background service that auto-restarts and survives reboots.

```bash
curl -fsSL https://raw.githubusercontent.com/kundakarlabp/nifty_scalper_bot/main/deploy/lightsail_setup.sh | bash
```

When it finishes it prints your dashboard URL and admin password.

---

## STEP 4 — Use the dashboard (browser, no terminal ever again)

Open: **http://15.206.3.6:8080/admin**

- Sign in with the admin password the script printed.
- Enter your Zerodha API key, API secret, access token, Telegram details. Save.
- Click **Restart Bot**.

### Every morning
- Generate your fresh Zerodha access token (as you do now).
- Open the dashboard → **Daily Access Token** box → paste → **Update token & restart**.
- Done.

### Anytime
- **View Logs** — see live activity (like Railway logs).
- **Restart Bot** — one click.

---

## If something looks wrong
- Dashboard won't open → re-check STEP 2 (port 8080 firewall rule).
- Orders rejected with "No IPs configured" → re-check STEP 1 (IP `15.206.3.6` saved on Zerodha).
- Need to re-run setup → it is safe to run STEP 3 again; it updates in place.
