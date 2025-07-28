// Dashboard JavaScript functions

// Global variables
let equityChart = null;
let updateInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard loaded');
    initializeCharts();
    loadDashboardData();
    startAutoRefresh();
});

// Initialize charts
function initializeCharts() {
    const equityCtx = document.getElementById('equityChart');
    if (equityCtx) {
        equityChart = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity (₹)',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '₹' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
}

// Load dashboard data
function loadDashboardData() {
    loadSystemStatus();
    loadPerformanceData();
    loadRecentSignals();
}

// Load system status
function loadSystemStatus() {
    fetch('/api/status')
    .then(response => response.json())
    .then(data => {
        updateStatusCards(data);
    })
    .catch(error => {
        console.error('Error loading status:', error);
    });
}

// Update status cards
function updateStatusCards(data) {
    // Update system status indicators
    const statusCard = document.querySelector('.card.text-white.bg-primary');
    if (statusCard) {
        const statusText = data.is_trading ? 'Active' : 'Inactive';
        const statusClass = data.is_trading ? 'bg-success' : 'bg-secondary';
        statusCard.className = `card text-white ${statusClass}`;
        statusCard.querySelector('.card-text').textContent = statusText;
        statusCard.querySelector('small').textContent = data.execution_enabled ? 'Trading enabled' : 'Trading disabled';
    }
    
    // Update other cards
    const riskStatus = data.risk_status || {};
    updateCardValue('.card.text-white.bg-success', '₹' + riskStatus.account_size?.toLocaleString());
    updateCardValue('.card.text-white.bg-info', '₹' + riskStatus.daily_pnl?.toLocaleString());
}

// Update card value
function updateCardValue(selector, value) {
    const card = document.querySelector(selector);
    if (card) {
        card.querySelector('.card-text').textContent = value;
    }
}

// Load performance data
function loadPerformanceData() {
    fetch('/api/performance')
    .then(response => response.json())
    .then(data => {
        updatePerformanceCharts(data);
        updatePerformanceMetrics(data.metrics);
        updateTradesTable(data.trades);
    })
    .catch(error => {
        console.error('Error loading performance data:', error);
    });
}

// Update performance charts
function updatePerformanceCharts(data) {
    if (equityChart && data.equity_curve) {
        const labels = data.equity_curve.map(point => {
            const date = new Date(point.timestamp);
            return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        });
        const values = data.equity_curve.map(point => point.equity);
        
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = values;
        equityChart.update();
    }
}

// Update performance metrics
function updatePerformanceMetrics(metrics) {
    // Update metrics in the UI
    console.log('Performance metrics:', metrics);
}

// Update trades table
function updateTradesTable(trades) {
    const tableBody = document.getElementById('trades-table');
    if (tableBody && trades) {
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Add new rows (limit to 5)
        const displayTrades = trades.slice(0, 5);
        displayTrades.forEach(trade => {
            const row = document.createElement('tr');
            const tradeDate = new Date(trade.timestamp);
            row.innerHTML = `
                <td>${tradeDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
                <td>${trade.symbol}</td>
                <td><span class="badge bg-${trade.direction === 'BUY' ? 'success' : 'danger'}">${trade.direction}</span></td>
                <td class="${trade.pnl >= 0 ? 'text-success' : 'text-danger'}">₹${Math.abs(trade.pnl).toLocaleString()}</td>
                <td><span class="badge bg-primary">Closed</span></td>
            `;
            tableBody.appendChild(row);
        });
    }
}

// Load recent signals
function loadRecentSignals() {
    fetch('/api/signals')
    .then(response => response.json())
    .then(data => {
        updateSignalsTable(data.signals);
    })
    .catch(error => {
        console.error('Error loading signals:', error);
    });
}

// Update signals table
function updateSignalsTable(signals) {
    const tableBody = document.getElementById('signals-table');
    if (tableBody && signals) {
        // Clear existing rows
        tableBody.innerHTML = '';
        
        // Add new rows (limit to 5)
        const displaySignals = signals.slice(0, 5);
        displaySignals.forEach(signal => {
            const row = document.createElement('tr');
            const signalDate = new Date(signal.timestamp);
            let statusClass = 'secondary';
            let statusText = 'Unknown';
            
            switch(signal.status) {
                case 'executed':
                    statusClass = 'success';
                    statusText = 'Executed';
                    break;
                case 'pending':
                    statusClass = 'warning';
                    statusText = 'Pending';
                    break;
                case 'rejected':
                    statusClass = 'secondary';
                    statusText = 'Rejected';
                    break;
            }
            
            row.innerHTML = `
                <td>${signalDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
                <td>${signal.symbol}</td>
                <td><span class="badge bg-${signal.direction === 'BUY' ? 'success' : 'danger'}">${signal.direction}</span></td>
                <td>${(signal.confidence * 100).toFixed(0)}%</td>
                <td><span class="badge bg-${statusClass}">${statusText}</span></td>
            `;
            tableBody.appendChild(row);
        });
    }
}

// Send system command
function sendCommand(command) {
    // Show loading state
    const buttons = document.querySelectorAll('.btn-group .btn');
    buttons.forEach(btn => {
        btn.disabled = true;
        const originalText = btn.innerHTML;
        btn.setAttribute('data-original-text', originalText);
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ' + btn.textContent;
    });
    
    fetch('/api/controls', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({command: command})
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage(data.message, 'success');
            // Refresh status after a short delay
            setTimeout(() => {
                loadSystemStatus();
                restoreButtons();
            }, 1000);
        } else {
            showMessage('Error: ' + data.message, 'danger');
            restoreButtons();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showMessage('Error sending command', 'danger');
        restoreButtons();
    });
}

// Restore button states
function restoreButtons() {
    const buttons = document.querySelectorAll('.btn-group .btn');
    buttons.forEach(btn => {
        btn.disabled = false;
        const originalText = btn.getAttribute('data-original-text');
        if (originalText) {
            btn.innerHTML = originalText;
        }
    });
}

// Show message
function showMessage(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to body
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

// Start auto-refresh
function startAutoRefresh() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    updateInterval = setInterval(() => {
        loadDashboardData();
    }, 30000); // Refresh every 30 seconds
}

// Stop auto-refresh
function stopAutoRefresh() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
});
