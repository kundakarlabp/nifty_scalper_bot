# Superlite operations

After this change is merged, run once on the Lightsail instance:

```bash
cd /home/ubuntu/nifty_scalper_bot
bash deploy/lightsail_release.sh --force
bash deploy/scripts/install_streamlit_console.sh
```

The admin controls then run on port 8081 and the read-only review console on port 8501. Existing settings are preserved unless a field is explicitly replaced. The existing auto-deployment timer continues updating the main branch and restarts the combined control service.
