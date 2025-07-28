import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from flask import Flask, render_template, jsonify, request
import logging
from datetime import datetime, timedelta
import json
import signal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

class MockTrader:
    """Mock trader for demonstration purposes"""
    def __init__(self):
        self.is_trading = True
        self.execution_enabled = False
        self.active_signals = 3
        self.active_positions = 1
    
    def get_trading_status(self):
        return {
            'is_trading': self.is_trading,
            'execution_enabled': self.execution_enabled,
            'active_signals': self.active_signals,
            'active_positions': self.active_positions,
            'streaming_status': {'connected': True, 'tokens': 1},
            'risk_status': {
                'account_size': 100000,
                'daily_pnl': 2500,
                'drawdown_percentage': 0.5,
                'current_positions': 1,
                'max_positions': 5
            }
        }

mock_trader = MockTrader()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("🛑 Received shutdown signal. Stopping web dashboard...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_system_status():
    """Get system status"""
    try:
        # In a real implementation, you'd get this from your actual trader
        status = mock_trader.get_trading_status()
        
        # Add timestamp
        status['timestamp'] = datetime.now().isoformat()
        status['uptime'] = "2 hours 35 minutes"
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance_data():
    """Get performance data"""
    try:
        # Generate sample performance data for demo
        import random
        
        # Generate sample trades
        sample_trades = []
        for i in range(15):
            trade = {
                'timestamp': (datetime.now() - timedelta(minutes=i*45)).isoformat(),
                'symbol': 'NIFTY',
                'direction': 'BUY' if random.random() > 0.5 else 'SELL',
                'entry_price': round(random.uniform(17800, 18200), 2),
                'exit_price': round(random.uniform(17800, 18200), 2),
                'quantity': random.choice([75, 150, 225]),
                'pnl': round(random.uniform(-3000, 6000), 2),
                'pnl_percentage': round(random.uniform(-3, 6), 2),
                'holding_period': random.randint(5, 120)
            }
            sample_trades.append(trade)
        
        # Generate sample metrics
        metrics = {
            'total_trades': 47,
            'winning_trades': 31,
            'losing_trades': 16,
            'win_rate': 66.0,
            'total_pnl': 47500,
            'average_pnl': 1010,
            'max_drawdown': 2.3,
            'sharpe_ratio': 1.8,
            'profit_factor': 2.1,
            'max_consecutive_wins': 5,
            'max_consecutive_losses': 2
        }
        
        # Generate sample equity curve
        equity_data = []
        base_equity = 100000
        for i in range(50):
            timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
            equity = base_equity + random.uniform(-5000, 15000) + (i * 200)
            equity_data.append({
                'timestamp': timestamp,
                'equity': round(equity, 2)
            })
        
        return jsonify({
            'trades': sample_trades,
            'metrics': metrics,
            'equity_curve': equity_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting performance  {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals')
def get_recent_signals():
    """Get recent trading signals"""
    try:
        # Generate sample signals for demo
        import random
        
        sample_signals = []
        for i in range(8):
            signal = {
                'timestamp': (datetime.now() - timedelta(minutes=i*30)).isoformat(),
                'symbol': 'NIFTY 50',
                'direction': 'BUY' if random.random() > 0.5 else 'SELL',
                'entry_price': round(random.uniform(17900, 18100), 2),
                'stop_loss': round(random.uniform(17850, 18050), 2),
                'target': round(random.uniform(17950, 18150), 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'volatility': round(random.uniform(0.8, 1.5), 2),
                'status': random.choice(['executed', 'pending', 'rejected'])
            }
            sample_signals.append(signal)
        
        return jsonify({
            'signals': sample_signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting signals  {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/controls', methods=['POST'])
def system_controls():
    """Handle system control commands"""
    try:
        data = request.get_json()
        command = data.get('command')
        
        # Handle different commands
        if command == 'start_trading':
            mock_trader.is_trading = True
            mock_trader.execution_enabled = True
            response = {'status': 'success', 'message': 'Trading started'}
        elif command == 'stop_trading':
            mock_trader.is_trading = False
            mock_trader.execution_enabled = False
            response = {'status': 'success', 'message': 'Trading stopped'}
        elif command == 'enable_execution':
            mock_trader.execution_enabled = True
            response = {'status': 'success', 'message': 'Execution enabled'}
        elif command == 'disable_execution':
            mock_trader.execution_enabled = False
            response = {'status': 'success', 'message': 'Execution disabled'}
        else:
            response = {'status': 'error', 'message': 'Unknown command'}
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error handling system controls: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/backtest')
def backtest():
    """Backtest results page"""
    return render_template('backtest.html')

@app.route('/logs')
def logs():
    """System logs page"""
    return render_template('logs.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Web Dashboard...")
    logger.info("🌐 Visit: http://localhost:8000")
    logger.info("💡 Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("🛑 Web dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running web dashboard: {e}")
