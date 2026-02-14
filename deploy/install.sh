#!/bin/bash

set -e

PROJECT_DIR="/home/ubuntu/nifty_scalper_bot"
VENV_DIR="$PROJECT_DIR/venv"

echo "===== Updating system ====="
apt update -y
apt install -y python3 python3-venv python3-pip git build-essential curl

echo "===== Creating swap if not exists ====="
bash deploy/setup_swap.sh

echo "===== Creating virtual environment ====="
cd $PROJECT_DIR
python3 -m venv venv
source venv/bin/activate

echo "===== Installing Python dependencies ====="
pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
pip install python-multipart

echo "===== Installing systemd services ====="
cp deploy/systemd/nifty-scalper.service /etc/systemd/system/
cp deploy/systemd/nifty-admin.service /etc/systemd/system/

systemctl daemon-reload
systemctl enable nifty-scalper
systemctl enable nifty-admin

echo "===== Starting services ====="
systemctl restart nifty-scalper
systemctl restart nifty-admin

echo "===== Installation complete ====="
systemctl status nifty-scalper --no-pager
systemctl status nifty-admin --no-pager
