from flask import Flask, render_template, jsonify, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
import time
import joblib
import numpy as np
import mplfinance as mpf
import yfinance as yf
import warnings
import threading
from gold_probability_ea import GoldProbabilityEA

app = Flask(__name__)

# Load ML model
model = joblib.load("ml_model.pkl")

# Global EA instance
ea_instance = None
ea_thread = None

def initialize_mt5():
    if not mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01"):
        print("Failed to connect:", mt5.last_error())
        return False
    return True

def get_live_data():
    if not initialize_mt5():
        return None
    
    # Fetch latest 1000 candles of XAUUSD
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
    
    if rates is None:
        print("Failed to get rates:", mt5.last_error())
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Get current price
    current_price = df.iloc[-1]['close']
    price_change = df.iloc[-1]['close'] - df.iloc[-2]['close']
    price_change_pct = (price_change / df.iloc[-2]['close']) * 100
    
    # Shutdown MT5 connection
    mt5.shutdown()
    return df, current_price, price_change, price_change_pct

def get_prediction(df):
    latest_data = df.iloc[-1][['open', 'high', 'low', 'close', 'tick_volume']].values.reshape(1, -1)
    prediction = model.predict_proba(latest_data)[0]
    bullish_prob = round(prediction[1] * 100, 2)  # Probability of bullish candle
    bearish_prob = round(prediction[0] * 100, 2)  # Probability of bearish candle
    return bullish_prob, bearish_prob

def get_correlation_data():
    # Get correlation with USD, S&P 500, and US Treasury Yields
    symbols = ['GC=F', 'DX-Y.NYB', '^GSPC', '^TNX']  # Gold, USD Index, S&P 500, 10Y Treasury
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1mo')
        data[symbol] = hist['Close']
    
    # Calculate correlations
    df_corr = pd.DataFrame(data)
    correlations = df_corr.corr()['GC=F'].drop('GC=F')
    return correlations

def calculate_sentiment(df):
    """Calculate market sentiment based on price action"""
    try:
        # Calculate sentiment based on recent price movements
        recent_prices = df['close'].tail(20)  # Last 20 periods
        price_changes = recent_prices.pct_change().dropna()
        
        # Calculate weighted sentiment
        weights = np.linspace(0.1, 1, len(price_changes))  # More weight to recent changes
        weighted_changes = price_changes * weights
        sentiment = np.mean(weighted_changes) * 100
        
        return round(sentiment, 2)
    except Exception as e:
        print(f"Error calculating sentiment: {str(e)}")
        return 0

def generate_chart(df):
    # Rename tick_volume to volume for mplfinance
    df = df.rename(columns={'tick_volume': 'volume'})
    
    # Set style for better visibility
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick='inherit',
            volume='inherit'
        ),
        gridstyle='dotted'
    )
    
    # Create the candlestick chart
    fig, axes = mpf.plot(df, type='candle', style=style,
                        title='XAUUSD Price Chart (5-Minute Interval)',
                        ylabel='Price (USD)',
                        volume=True,
                        returnfig=True,
                        figsize=(12, 8))
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.3)
    
    # Save the chart as an image
    plt.savefig("static/xauusd_chart.png", bbox_inches='tight', dpi=100)
    plt.close(fig)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_ma(prices, period=20):
    """Calculate Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_technical_indicators(df):
    """Calculate technical indicators for the latest data"""
    try:
        # Calculate RSI
        rsi = calculate_rsi(df['close'])
        
        # Calculate MACD
        macd, signal, hist = calculate_macd(df['close'])
        
        # Calculate Moving Average
        ma20 = calculate_ma(df['close'], 20)
        
        # Calculate Stochastic
        k, d = calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Get latest values
        latest_rsi = round(float(rsi.iloc[-1]), 1)
        latest_macd = round(float(hist.iloc[-1]), 2)
        latest_ma20 = round(float(ma20.iloc[-1]), 2)
        latest_stoch = round(float(k.iloc[-1]), 1)
        
        # Determine signals
        rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        macd_signal = "Bullish" if latest_macd > 0 else "Bearish"
        ma_signal = "Above Price" if latest_ma20 > df['close'].iloc[-1] else "Below Price"
        stoch_signal = "Overbought" if latest_stoch > 80 else "Oversold" if latest_stoch < 20 else "Neutral"
        
        return {
            "rsi": {"value": latest_rsi, "signal": rsi_signal},
            "macd": {"value": latest_macd, "signal": macd_signal},
            "ma20": {"value": latest_ma20, "signal": ma_signal},
            "stoch": {"value": latest_stoch, "signal": stoch_signal}
        }
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        return None

def calculate_support_resistance(df):
    """Calculate support and resistance levels"""
    try:
        # Get recent high and low prices
        recent_data = df.tail(100)
        
        # Calculate pivot points
        pivot = (recent_data['high'].iloc[-1] + recent_data['low'].iloc[-1] + recent_data['close'].iloc[-1]) / 3
        
        r1 = 2 * pivot - recent_data['low'].iloc[-1]
        r2 = pivot + (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1])
        s1 = 2 * pivot - recent_data['high'].iloc[-1]
        s2 = pivot - (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1])
        
        return {
            "strong_resistance": round(r2, 2),
            "resistance": round(r1, 2),
            "support": round(s1, 2),
            "strong_support": round(s2, 2)
        }
    except Exception as e:
        print(f"Error calculating support/resistance: {str(e)}")
        return None

def calculate_market_insights(df):
    """Calculate market insights including trend strength and volatility"""
    try:
        # Calculate trend strength using price momentum
        returns = df['close'].pct_change()
        momentum = returns.rolling(window=20).mean() * 100
        trend_strength = min(abs(float(momentum.iloc[-1] * 10)), 100)  # Scale and cap at 100
        
        # Calculate volatility using standard deviation
        volatility = min(float(returns.rolling(window=20).std() * 100 * 10), 100)  # Scale and cap at 100
        
        # Determine trend direction using simple moving averages
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        trend_direction = "Uptrend" if float(sma20.iloc[-1]) > float(sma50.iloc[-1]) else "Downtrend"
        
        # Create market summary
        if trend_strength > 25:
            strength_desc = "Strong" if trend_strength > 50 else "Moderate"
            trend_desc = f"{strength_desc} {trend_direction}"
        else:
            trend_desc = "Ranging Market"
            
        volatility_desc = "High" if volatility > 70 else "Moderate" if volatility > 30 else "Low"
        
        return {
            "trend_strength": trend_strength,
            "trend_description": trend_desc,
            "volatility": volatility,
            "volatility_description": volatility_desc
        }
    except Exception as e:
        print(f"Error calculating market insights: {str(e)}")
        return None

@app.route("/")
def home():
    data = get_live_data()
    if data is not None:
        df, current_price, price_change, price_change_pct = data
        generate_chart(df)
        bullish, bearish = get_prediction(df)
        
        # Get additional data
        correlations = get_correlation_data()
        sentiment = calculate_sentiment(df)
        
        # Calculate technical analysis data
        technical_indicators = calculate_technical_indicators(df)
        support_resistance = calculate_support_resistance(df)
        market_insights = calculate_market_insights(df)
        
        return render_template("index.html", 
                             bullish=bullish, 
                             bearish=bearish,
                             current_price=round(current_price, 2),
                             price_change=round(price_change, 2),
                             price_change_pct=round(price_change_pct, 2),
                             correlations=correlations,
                             sentiment=sentiment,
                             technical_indicators=technical_indicators,
                             support_resistance=support_resistance,
                             market_insights=market_insights)
    return render_template("index.html")

@app.route('/control-panel')
def control_panel():
    return render_template('control_panel.html')

@app.route("/update_chart")
def update_chart():
    data = get_live_data()
    if data is not None:
        df, current_price, price_change, price_change_pct = data
        generate_chart(df)
        bullish, bearish = get_prediction(df)
        
        # Get additional data
        correlations = get_correlation_data()
        sentiment = calculate_sentiment(df)
        
        # Calculate technical analysis data
        technical_indicators = calculate_technical_indicators(df)
        support_resistance = calculate_support_resistance(df)
        market_insights = calculate_market_insights(df)
        
        return jsonify({
            "status": "success",
            "bullish": bullish,
            "bearish": bearish,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "correlations": correlations.to_dict(),
            "sentiment": sentiment,
            "technical_indicators": technical_indicators,
            "support_resistance": support_resistance,
            "market_insights": market_insights
        })
    return jsonify({"status": "error"})

@app.route('/start_ea', methods=['POST'])
def start_ea():
    global ea_instance, ea_thread
    try:
        if ea_instance is None:
            ea_instance = GoldProbabilityEA()
            ea_thread = threading.Thread(target=run_ea)
            ea_thread.daemon = True
            ea_thread.start()
            return jsonify({"status": "success", "message": "EA started successfully"})
        return jsonify({"status": "error", "message": "EA is already running"})
    except Exception as e:
        print(f"Error starting EA: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_ea', methods=['POST'])
def stop_ea():
    global ea_instance, ea_thread
    try:
        if ea_instance is not None:
            ea_instance = None
            if ea_thread and ea_thread.is_alive():
                ea_thread.join(timeout=1)
            ea_thread = None
            return jsonify({"status": "success", "message": "EA stopped successfully"})
        return jsonify({"status": "error", "message": "EA is not running"})
    except Exception as e:
        print(f"Error stopping EA: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_settings', methods=['POST'])
def save_settings():
    global ea_instance
    try:
        settings = request.json
        if ea_instance is not None:
            ea_instance.technical_weight = float(settings['technical_weight']) / 100
            ea_instance.market_weight = float(settings['market_weight']) / 100
            ea_instance.risk_weight = float(settings['risk_weight']) / 100
            ea_instance.min_probability = float(settings['min_probability'])
            ea_instance.strong_signal_threshold = float(settings['strong_signal_threshold'])
            ea_instance.rsi_period = int(settings['rsi_period'])
            ea_instance.macd_fast = int(settings['macd_fast'])
            ea_instance.macd_slow = int(settings['macd_slow'])
            ea_instance.macd_signal = int(settings['macd_signal'])
            ea_instance.ma_short = int(settings['ma_short'])
            ea_instance.ma_long = int(settings['ma_long'])
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "EA is not running"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    try:
        if not initialize_mt5():
            return jsonify({"status": "error", "message": "Failed to connect to MT5"})
            
        trade_data = request.json
        symbol = "XAUUSD"
        point = mt5.symbol_info(symbol).point
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return jsonify({"status": "error", "message": "Failed to get price"})
            
        price = tick.ask if trade_data['type'] == 'BUY' else tick.bid
        
        # Calculate SL and TP
        sl = price - (float(trade_data['stop_loss']) * point) if trade_data['type'] == 'BUY' else price + (float(trade_data['stop_loss']) * point)
        tp = price + (float(trade_data['take_profit']) * point) if trade_data['type'] == 'BUY' else price - (float(trade_data['take_profit']) * point)
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(trade_data['lot_size']),
            "type": mt5.ORDER_TYPE_BUY if trade_data['type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "Manual Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({"status": "error", "message": f"Order failed: {result.comment}"})
            
        mt5.shutdown()
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_positions')
def get_positions():
    try:
        if not initialize_mt5():
            return jsonify({"status": "error", "message": "Failed to connect to MT5"})
            
        positions = mt5.positions_get(symbol="XAUUSD")
        if positions is None:
            return jsonify({"positions": []})
            
        positions_data = []
        for position in positions:
            positions_data.append({
                "ticket": position.ticket,
                "type": "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": position.volume,
                "price_open": position.price_open,
                "sl": position.sl,
                "tp": position.tp,
                "profit": position.profit
            })
            
        mt5.shutdown()
        return jsonify({"positions": positions_data})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/close_position', methods=['POST'])
def close_position():
    try:
        if not initialize_mt5():
            return jsonify({"status": "error", "message": "Failed to connect to MT5"})
            
        # Check if auto-trading is enabled
        if not mt5.terminal_info().trade_allowed:
            return jsonify({"status": "error", "message": "AutoTrading is disabled in MT5. Please enable it in the MT5 client."})
            
        # Get ticket from request data
        data = request.get_json()
        if not data or 'ticket' not in data:
            return jsonify({"status": "error", "message": "No ticket provided"})
            
        ticket = data['ticket']
            
        positions = mt5.positions_get(symbol="XAUUSD")
        if positions is None:
            return jsonify({"status": "error", "message": "No positions found"})
            
        position = None
        for pos in positions:
            if pos.ticket == ticket:
                position = pos
                break
                
        if position is None:
            return jsonify({"status": "error", "message": "Position not found"})
            
        # Get current price
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return jsonify({"status": "error", "message": "Failed to get current price"})
            
        # Prepare close request
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": tick.ask if position.type == mt5.POSITION_TYPE_BUY else tick.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close Position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close request
        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_message = f"Failed to close position: {result.comment}"
            if result.retcode == mt5.TRADE_RETCODE_TRADE_DISABLED:
                error_message = "AutoTrading is disabled in MT5. Please enable it in the MT5 client."
            return jsonify({"status": "error", "message": error_message})
            
        mt5.shutdown()
        return jsonify({"status": "success", "message": "Position closed successfully"})
        
    except Exception as e:
        print(f"Error closing position: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

def run_ea():
    global ea_instance
    try:
        while ea_instance is not None:
            ea_instance.run()
            time.sleep(1)  # Add a small delay to prevent CPU overuse
    except Exception as e:
        print(f"Error in EA thread: {str(e)}")
        ea_instance = None

if __name__ == "__main__":
    app.run(debug=True)
