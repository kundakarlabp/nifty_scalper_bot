# Deployment Guide 🚀

This guide covers deploying the Nifty Scalper Bot to various platforms including Render.com, AWS, and local Docker setups.

## 🌐 Render.com Deployment (Recommended)

### Prerequisites
- GitHub account
- Render.com account (free tier available)
- Zerodha Kite API credentials

### Step-by-Step Deployment

1. **Fork the Repository**
   ```bash
   # Fork this repo on GitHub or create your own
   git clone https://github.com/yourusername/nifty-scalper-bot.git
   cd nifty-scalper-bot
   ```

2. **Push to Your GitHub Repository**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

3. **Create Render Web Service**
   - Go to [Render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name**: `nifty-scalper-bot`
     - **Environment**: `Docker`
     - **Region**: Choose closest to you
     - **Branch**: `main`
     - **Build Command**: (leave empty, handled by Dockerfile)
     - **Start Command**: `python nifty_scalper_bot.py`

4. **Set Environment Variables**
   
   In Render dashboard, add these environment variables:

   **Required Variables:**
   ```
   ZERODHA_API_KEY=your_api_key_here
   ZERODHA_API_SECRET=your_api_secret_here
   ZERODHA_CLIENT_ID=your_client_id_here
   ZERODHA_ACCESS_TOKEN=your_access_token_here
   ```

   **Optional but Recommended:**
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   TRADING_CAPITAL=100000
   AUTO_TRADE=true
   DRY_RUN=false
   MAX_DAILY_LOSS_PCT=5
   TRADE_LOT_SIZE=75
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Your bot will be available at `https://your-app-name.onrender.com`

### Render.com Configuration Tips

- **Free Tier Limitations**: 
  - Services sleep after 15 minutes of inactivity
  - 750 hours/month limit
  - Consider upgrading to paid plan for production

- **Health Checks**: 
  - Render automatically monitors `/health` endpoint
  - Bot will restart if health checks fail

- **Logs**:
  - View logs in Render dashboard
  - Set `LOG_LEVEL=DEBUG` for detailed logging

## 🐳 Local Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- `.env` file configured

### Quick Start
```bash
# Clone and setup
git clone https://github.com/yourusername/nifty-scalper-bot.git
cd nifty-scalper-bot

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run with Docker Compose
chmod +x run.sh
./run.sh start

# View logs
./run.sh logs

# Stop
./run.sh stop
```

### Docker Commands
```bash
# Build and run in foreground
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f nifty-scalper-bot

# Stop and remove containers
docker-compose down

# Clean everything
docker-compose down --volumes --remove-orphans
docker system prune -f
```

## ☁️ AWS Deployment

### Using AWS ECS (Elastic Container Service)

1. **Push Image to ECR**
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name nifty-scalper-bot

   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build and tag image
   docker build -t nifty-scalper-bot .
   docker tag nifty-scalper-bot:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/nifty-scalper-bot:latest

   # Push image
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nifty-scalper-bot:latest
   ```

2. **Create ECS Task Definition**
   ```json
   {
     "family": "nifty-scalper-bot",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "nifty-scalper-bot",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/nifty-scalper-bot:latest",
         "portMappings": [
           {
             "containerPort": 10000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {"name": "ZERODHA_API_KEY", "value": "your_api_key"},
           {"name": "ZERODHA_API_SECRET", "value": "your_api_secret"}
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/nifty-scalper-bot",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

### Using AWS EC2

1. **Launch EC2 Instance**
   - AMI: Amazon Linux 2
   - Instance Type: t3.micro (free tier)
   - Security Group: Allow port 10000

2. **Setup on EC2**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ec2-user@your-instance-ip

   # Install Docker
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose

   # Clone and run
   git clone https://github.com/yourusername/nifty-scalper-bot.git
   cd nifty-scalper-bot
   # Configure .env file
   docker-compose up -d --build
   ```

## 🔧 VPS Deployment (DigitalOcean, Linode, etc.)

### Setup Script for Ubuntu 20.04+
```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Clone project
git clone https://github.com/yourusername/nifty-scalper-bot.git
cd nifty-scalper-bot

# Setup environment
cp .env.example .env
echo "Please edit .env file with your credentials"
nano .env

# Run bot
docker-compose up -d --build

# Setup log rotation
sudo tee /etc/logrotate.d/nifty-scalper-bot > /dev/null <<EOF
/home/$(whoami)/nifty-scalper-bot/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}