# Deploying to Google Compute Engine

1. **Reserve static IP**
   - Navigate to *VPC network → IP addresses* and reserve an external static IP in the same region as the VM.

2. **Create VM**
   - Use an `e2-micro` instance with Ubuntu 22.04 LTS.
   - Attach the reserved static IP and enable SSH access.

3. **Timezone**
   - `sudo timedatectl set-timezone Asia/Kolkata`

4. **Install Docker and Git**
   - `sudo apt-get update && sudo apt-get -y install docker.io git && sudo usermod -aG docker $USER`
   - Log out and back in to apply the Docker group.

5. **Application directory**
   - `sudo mkdir -p /opt/niftybot && sudo chown $USER:$USER /opt/niftybot`

6. **Environment file**
   - Copy `.env.example` and fill values: `sudo nano /opt/niftybot/.env`

7. **First run**
   ```bash
   git clone https://github.com/<you>/<repo>.git ~/niftybot
   cd ~/niftybot && docker build -t niftybot:latest .
   sudo tee /etc/systemd/system/niftybot.service < deploy/gce/niftybot.service
   sudo systemctl daemon-reload && sudo systemctl enable niftybot && sudo systemctl start niftybot
   curl http://<VM_IP>:8000/health
   ```

8. **GitHub Actions**
   - Add repo secrets `GCE_HOST`, `GCE_USER`, `GCE_SSH_KEY`, `GCE_KNOWN_HOSTS`.
   - On push to `main`, the deploy workflow redeploys via SSH.

9. **Security**
   - Prefer no public firewall on port 8000; use an SSH tunnel:
     `ssh -L 8000:localhost:8000 user@VM` then `curl http://localhost:8000/health`.

10. **Cron (optional)**
    - `10 9 * * 1-5 systemctl restart niftybot` for a fresh pre-market start.
