# Gold Trading Expert Advisor with Machine Learning

A sophisticated Expert Advisor (EA) for MetaTrader 5 that combines machine learning predictions with technical analysis for gold (XAUUSD) trading. The system includes a web-based control panel for monitoring and managing trades.

## Features

- **Machine Learning Integration**: Uses a trained model to predict price movements
- **Technical Analysis**: Implements multiple technical indicators (RSI, MACD, Moving Averages, Stochastic)
- **Risk Management**: Includes position sizing and stop-loss management
- **Web Control Panel**: Real-time monitoring and control interface
- **Market Analysis**: Provides market insights, sentiment analysis, and correlation data
- **Martingale Strategy**: Optional position sizing strategy for recovery trades

## Prerequisites

- Python 3.8 or higher
- MetaTrader 5
- MetaTrader 5 Python package
- Flask
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gold-trading-ea.git
cd gold-trading-ea
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure MetaTrader 5:
   - Install MetaTrader 5
   - Enable AutoTrading in MT5
   - Update the login credentials in `app.py` with your MT5 account details

4. Run the application:
```bash
python app.py
```

## Project Structure

```
gold-trading-ea/
├── app.py                 # Main Flask application
├── gold_probability_ea.py # EA implementation
├── gold_rsi_martingale_ea.mq4  # MT4 EA version
├── ml_model.pkl          # Trained machine learning model
├── requirements.txt      # Python dependencies
├── static/              # Static files (charts, CSS)
├── templates/           # HTML templates
│   ├── index.html      # Main dashboard
│   └── control_panel.html  # EA control panel
└── README.md           # This file
```

## Configuration

### MT5 Connection Settings
Update the following in `app.py`:
```python
def initialize_mt5():
    if not mt5.initialize(server="YOUR_SERVER", login=YOUR_LOGIN, password="YOUR_PASSWORD"):
        print("Failed to connect:", mt5.last_error())
        return False
    return True
```

### EA Parameters
The EA can be configured through the web interface or by modifying the parameters in `gold_probability_ea.py`:
- Technical weight
- Market weight
- Risk weight
- Minimum probability threshold
- Strong signal threshold
- Indicator parameters (RSI, MACD, MA periods)

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface:
   - Open your browser and navigate to `http://localhost:5000`
   - Use the control panel to manage the EA

3. Monitor trades:
   - View open positions
   - Check performance statistics
   - Monitor technical indicators
   - Track market sentiment

## Features in Detail

### Machine Learning Model
- Trained on historical price data
- Predicts probability of bullish/bearish movements
- Updates predictions in real-time

### Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Moving Averages
- Stochastic Oscillator
- Support/Resistance Levels

### Risk Management
- Position sizing based on account balance
- Stop-loss and take-profit levels
- Trailing stop functionality
- Maximum drawdown protection

### Web Interface
- Real-time price chart
- Technical indicator display
- Position management
- Performance statistics
- Market insights

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 