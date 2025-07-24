# Nifty Scalper Bot 🚀

A production-ready algorithmic trading bot for NIFTY futures using Zerodha Kite API with Telegram notifications and comprehensive risk management.

## ✨ Features

- **Advanced Signal Generation**: Multi-indicator technical analysis (RSI, MACD, EMA, Bollinger Bands, ATR, VWAP)
- **Risk Management**: Circuit breakers, daily loss limits, adaptive position sizing
- **Telegram Integration**: Real-time notifications and remote control
- **ATR-based Stop Loss/Take Profit**: Dynamic risk management based on market volatility
- **Adaptive Thresholds**: Self-adjusting signal sensitivity based on performance
- **Health Monitoring**: Flask dashboard with REST API endpoints
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Render.com Ready**: Optimized for cloud deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │  Signal Engine  │    │   Kite Client   │
│                 │    │                 │    │                 │
│ • Commands      │    │ • Technical     │    │ • Order         │
│ • Notifications │    │   Indicators    │    │   Management    │
│ • Remote Control│    │ • ML Integration│    │ • Position      │
└─────────────────┘    └─────────────────┘    │   Tracking      │
         │                       │             └─────────────────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Main Bot Core  │
                    │                 │
                    │ • Trading Logic │
                    │ • Risk Manager  │
                    │ • Flask API     │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Zerodha Kite API credentials
- Telegram Bot Token (optional)
- Docker (for containerized deployment)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/nifty-scalper-bot.git
cd nifty-scalper-bot
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Bot

```bash
python nifty_scalper_bot.py
```

## 🐳 Docker Deployment

### Local Docker

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f nifty-scalper-bot
```

### Render.com Deployment

1. Fork this repository
2. Connect to Render.com
3. Create a new Web Service
4. Set environment variables in Render dashboard
5. Deploy!

```bash
# Render will automatically use:
# - Dockerfile for containerization
# - Port 10000 for health checks
# - /health endpoint for monitoring
```

## ⚙️ Configuration

### Essential Environment Variables

```bash
# Zerodha API (Required)
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_secret
ZERODHA_CLIENT_ID=your_client_id
ZERODHA_ACCESS_TOKEN=your_access_token

# Trading Settings
TRADING_CAPITAL=100000
MAX_DAILY_LOSS_PCT=5
TRADE_LOT_SIZE=75
AUTO_TRADE=true
DRY_RUN=false

# Risk Management
USE_ATR_SL=true
ATR_SL_MULT=1.2
ATR_TP_MULT=1.8
MAX_LOSS_STREAK=3

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start auto-trading |
| `/stop` | Stop auto-trading |
| `/status` | Show bot status |
| `/position` | Show current position |
| `/exit` | Force close position |
| `/trades` | Show recent trades |
| `/help` | Show help message |

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Bot Status
```http
GET /status
```

### Recent Trades
```http
GET /trades
```

### Control Bot
```http
POST /control
Content-Type: application/json

{
  "action": "start|stop|exit_position"
}
```

## 📊 Signal Generation

The bot uses a multi-factor signal generation system:

### Technical Indicators (70% weight)
- **EMA Crossover**: 9 vs 21 period (40% of technical)
- **MACD**: Line vs Signal crossover (30% of technical)  
- **RSI**: Momentum confirmation (20% of technical)
- **Bollinger Bands**: Mean reversion (10% of technical)

### Volume Analysis (20% weight)
- Volume spike confirmation
- VWAP alignment

### Adaptive Thresholds (10% weight)
- Performance-based threshold adjustment
- Win rate optimization

## 🛡️ Risk Management

### Circuit Breakers
- **Consecutive Losses**: Pause trading after N losses
- **Daily Loss Limit**: Stop trading if daily loss exceeds X%
- **Drawdown Protection**: Reduce position size during drawdowns

### Position Sizing
- Base lot size from configuration
- Reduced size during drawdowns (>10% = 75%, >15% = 50%)
- ATR-based stop losses for volatility adjustment

### ATR-Based Exits
```python
# For BUY positions
stop_loss = entry_price - (ATR * 1.2)
take_profit = entry_price + (ATR * 1.8)

# For SELL positions  
stop_loss = entry_price + (ATR * 1.2)
take_profit = entry_price - (ATR * 1.8)
```

## 📈 Performance Monitoring

### Key Metrics
- Total P&L and current balance
- Win rate and profit factor
- Maximum drawdown
- Average trade duration
- Daily/weekly performance

### Logging
- Comprehensive logging to files and console
- Trade execution logs
- Error tracking and debugging
- Performance analytics

## 🔧 Development

### Project Structure
```
nifty-scalper-bot/
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── kite_client.py         # Zerodha API wrapper
├── signal_generator.py    # Trading signal logic
├── telegram_bot.py        # Telegram integration
├── nifty_scalper_bot.py  # Main bot application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
└── README.md            # Documentation
```

### Adding New Indicators

```python
# In signal_generator.py
def calculate_new_indicator(self, df: pd.DataFrame) -> float:
    # Your indicator logic here
    return indicator_value

def generate_signal(self, indicators: Dict[str, float]) -> float:
    # Include your indicator in signal calculation
    signal += your_indicator_weight * your_logic
    return signal
```

## 🚨 Important Disclaimers

- **Risk Warning**: Trading involves substantial risk of loss
- **No Guarantees**: Past performance doesn't guarantee future results  
- **Test First**: Always test in paper trading mode initially
- **Monitor Actively**: Don't leave the bot unattended for extended periods
- **Legal Compliance**: Ensure compliance with local trading regulations

## 📞 Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this README and code comments
- **Community**: Join our discussions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Zerodha for the excellent Kite Connect API
- The Python trading community
- Contributors and testers

---

**⚠️ Risk Disclaimer**: This software is for educational purposes. Use at your own risk. The authors are not responsible for any financial losses.