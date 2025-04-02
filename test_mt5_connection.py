import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

def test_mt5_connection():
    # Initialize MT5 connection
    if not mt5.initialize(server="Exness-MT5Trial8", login=79700174, password="Botmudra.com@01"):
        print(f"Failed to connect to MT5: {mt5.last_error()}")
        return False
    
    print("Successfully connected to MT5")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return False
    
    print(f"Account Info:")
    print(f"Balance: {account_info.balance}")
    print(f"Equity: {account_info.equity}")
    print(f"Margin: {account_info.margin}")
    print(f"Free Margin: {account_info.margin_free}")
    
    # Get symbol info
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return False
    
    print(f"\nSymbol Info for {symbol}:")
    print(f"Bid: {symbol_info.bid}")
    print(f"Ask: {symbol_info.ask}")
    print(f"Point: {symbol_info.point}")
    
    # Test market data
    print("\nTesting market data retrieval...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
    if rates is None:
        print("Failed to get market data")
        print(f"Last error: {mt5.last_error()}")
        return False
    
    print("\nLatest Market Data:")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(df)
    
    # Test order placement
    print("\nTesting order placement...")
    point = symbol_info.point
    price = symbol_info.ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,  # Reduced volume for testing
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Test Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"Attempting to place test order at price: {price}")
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        print(f"Error code: {result.retcode}")
        return False
    
    print("Test order placed successfully!")
    print(f"Order ticket: {result.order}")
    print(f"Order volume: {result.volume}")
    print(f"Order price: {result.price}")
    
    # Wait a moment to ensure the order is processed
    time.sleep(2)
    
    # Get open positions
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        print("\nNo open positions")
        print(f"Last error: {mt5.last_error()}")
    else:
        print("\nOpen Positions:")
        for position in positions:
            print(f"Ticket: {position.ticket}")
            print(f"Type: {'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL'}")
            print(f"Volume: {position.volume}")
            print(f"Price: {position.price_open}")
            print(f"Current Price: {position.price_current}")
            print(f"Profit: {position.profit}")
    
    # Shutdown MT5 connection
    mt5.shutdown()
    return True

if __name__ == "__main__":
    print("Testing MT5 Connection...")
    success = test_mt5_connection()
    if success:
        print("\nMT5 Connection Test: SUCCESS")
    else:
        print("\nMT5 Connection Test: FAILED") 