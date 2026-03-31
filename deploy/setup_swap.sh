#!/bin/bash

if swapon --show | grep -q '/swapfile'; then
    echo "Swap already exists"
else
    echo "Creating 2GB swap"
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
