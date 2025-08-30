#!/usr/bin/env bash
set -euo pipefail
cd ~/niftybot
git fetch --all
git reset --hard origin/main
docker build -t niftybot:latest .
sudo systemctl restart niftybot
sleep 2
sudo systemctl --no-pager status niftybot || true
