import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gold_ea.log'
)

class GoldProbabilityEA:
    def __init__(self, symbol='XAUUSD', timeframe='5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = None
        self.last_signal = None
        self.stop_loss = None
        self.take_profit = None
        self.lot_size = 0.1  # Default lot size
        
        # Strategy Parameters
        self.technical_weight = 0.40
        self.market_weight = 0.35
        self.risk_weight = 0.25
        self.min_probability = 55
        self.strong_signal_threshold = 70
        
        # Initialize indicators
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ma_short = 20
        self.ma_long = 50
        
        # Initialize MT5 connection
        self.initialize_mt5()
        
    def initialize_mt5(self):
        """Initialize connection to MT5"""
        if not mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01"):
            logging.error(f"Failed to connect to MT5: {mt5.last_error()}")
            return False
        logging.info("Successfully connected to MT5")
        return True

    def calculate_rsi(self, data, periods=14):
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def place_order(self, order_type, price, sl=None, tp=None):
        """Place a trade order in MT5"""
        try:
            point = mt5.symbol_info(self.symbol).point
            price = float(price)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Gold Probability EA",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if sl:
                request["sl"] = float(sl)
            if tp:
                request["tp"] = float(tp)
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.comment}")
                return False
                
            logging.info(f"Order placed successfully: {order_type} {self.symbol} at {price}")
            return True
            
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return False

    def close_position(self, position):
        """Close an existing position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close Gold Probability EA",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to close position: {result.comment}")
                return False
                
            logging.info(f"Position closed successfully: {position.ticket}")
            return True
            
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            return False

    def get_positions(self):
        """Get all open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
            return positions
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return []

    def calculate_stop_loss_take_profit(self, order_type, price):
        """Calculate stop loss and take profit levels"""
        try:
            point = mt5.symbol_info(self.symbol).point
            atr = self.calculate_atr_from_mt5()
            
            if order_type == 'BUY':
                sl = price - (atr * 2)  # 2 ATR for stop loss
                tp = price + (atr * 4)  # 4 ATR for take profit
            else:
                sl = price + (atr * 2)
                tp = price - (atr * 4)
                
            return sl, tp
        except Exception as e:
            logging.error(f"Error calculating SL/TP: {e}")
            return None, None

    def calculate_atr_from_mt5(self, period=14):
        """Calculate ATR from MT5 data"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None:
                return 0
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            atr = self.calculate_atr(df['high'], df['low'], df['close'], period)
            return float(atr.iloc[-1])
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return 0

    def fetch_data(self):
        """Fetch latest market data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None

    def calculate_technical_score(self, df):
        """Calculate technical analysis score (40% weight)"""
        try:
            # RSI
            rsi = self.calculate_rsi(df['close'], self.rsi_period)
            rsi_score = 100 - abs(50 - rsi.iloc[-1])
            
            # MACD
            macd, signal = self.calculate_macd(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)
            macd_score = 50 + (macd.iloc[-1] / df['close'].iloc[-1] * 100)
            
            # Moving Averages
            ma_short = df['close'].rolling(window=self.ma_short).mean()
            ma_long = df['close'].rolling(window=self.ma_long).mean()
            ma_score = 50 + ((ma_short.iloc[-1] - ma_long.iloc[-1]) / df['close'].iloc[-1] * 100)
            
            # Combine technical scores
            technical_score = (rsi_score + macd_score + ma_score) / 3
            return technical_score
            
        except Exception as e:
            logging.error(f"Error calculating technical score: {e}")
            return 50

    def calculate_market_score(self, df):
        """Calculate market conditions score (35% weight)"""
        try:
            # Trend Analysis
            atr = self.calculate_atr(df['high'], df['low'], df['close'], 14)
            trend_strength = abs(df['close'].iloc[-1] - df['close'].iloc[-20]) / atr.iloc[-1]
            trend_score = min(100, trend_strength * 20)
            
            # Volatility
            volatility = atr.iloc[-1] / df['close'].iloc[-1] * 100
            volatility_score = 100 - min(100, volatility * 10)
            
            # Support/Resistance
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            sr_score = 50 + ((current_price - (recent_high + recent_low) / 2) / 
                            (recent_high - recent_low) * 50)
            
            # Combine market scores
            market_score = (trend_score + volatility_score + sr_score) / 3
            return market_score
            
        except Exception as e:
            logging.error(f"Error calculating market score: {e}")
            return 50

    def calculate_risk_score(self, df):
        """Calculate risk management score (25% weight)"""
        try:
            # Volatility-based risk
            atr = self.calculate_atr(df['high'], df['low'], df['close'], 14)
            volatility_risk = 100 - min(100, (atr.iloc[-1] / df['close'].iloc[-1] * 100) * 10)
            
            # Trend consistency
            price_changes = df['close'].pct_change()
            trend_consistency = abs(price_changes.iloc[-5:].mean() / price_changes.iloc[-5:].std())
            consistency_score = min(100, trend_consistency * 20)
            
            # Volume analysis
            volume_ma = df['tick_volume'].rolling(window=20).mean()
            volume_score = 50 + ((df['tick_volume'].iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1] * 50)
            
            # Combine risk scores
            risk_score = (volatility_risk + consistency_score + volume_score) / 3
            return risk_score
            
        except Exception as e:
            logging.error(f"Error calculating risk score: {e}")
            return 50

    def calculate_probabilities(self):
        """Calculate bullish and bearish probabilities"""
        try:
            df = self.fetch_data()
            if df is None or len(df) < 50:
                return None, None
            
            # Calculate component scores
            technical_score = self.calculate_technical_score(df)
            market_score = self.calculate_market_score(df)
            risk_score = self.calculate_risk_score(df)
            
            # Calculate weighted score
            weighted_score = (
                technical_score * self.technical_weight +
                market_score * self.market_weight +
                risk_score * self.risk_weight
            )
            
            # Convert to probabilities
            bullish_prob = weighted_score
            bearish_prob = 100 - weighted_score
            
            return bullish_prob, bearish_prob
            
        except Exception as e:
            logging.error(f"Error calculating probabilities: {e}")
            return None, None

    def check_entry_conditions(self):
        """Check if entry conditions are met and execute trades"""
        try:
            bullish_prob, bearish_prob = self.calculate_probabilities()
            if bullish_prob is None or bearish_prob is None:
                return None
            
            # Get current positions
            positions = self.get_positions()
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
                
            current_price = tick.ask if tick.ask > 0 else tick.bid
            
            # Strong signal conditions
            if bullish_prob > self.strong_signal_threshold:
                if not positions:  # No open positions
                    sl, tp = self.calculate_stop_loss_take_profit('BUY', current_price)
                    if self.place_order('BUY', current_price, sl, tp):
                        return 'BUY'
            elif bearish_prob > self.strong_signal_threshold:
                if not positions:  # No open positions
                    sl, tp = self.calculate_stop_loss_take_profit('SELL', current_price)
                    if self.place_order('SELL', current_price, sl, tp):
                        return 'SELL'
            
            # Moderate signal conditions
            elif bullish_prob > self.min_probability:
                if not positions:  # No open positions
                    sl, tp = self.calculate_stop_loss_take_profit('BUY', current_price)
                    if self.place_order('BUY', current_price, sl, tp):
                        return 'BUY'
            elif bearish_prob > self.min_probability:
                if not positions:  # No open positions
                    sl, tp = self.calculate_stop_loss_take_profit('SELL', current_price)
                    if self.place_order('SELL', current_price, sl, tp):
                        return 'SELL'
            
            return None
            
        except Exception as e:
            logging.error(f"Error checking entry conditions: {e}")
            return None

    def run(self):
        """Main EA loop"""
        logging.info("Starting Gold Probability EA")
        while True:
            try:
                # Check for new signals
                signal = self.check_entry_conditions()
                
                if signal and signal != self.last_signal:
                    logging.info(f"New signal detected: {signal}")
                    self.last_signal = signal
                
                # Wait for next interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    ea = GoldProbabilityEA()
    ea.run() 