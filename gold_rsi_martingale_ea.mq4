//+------------------------------------------------------------------+
//|                                              Gold RSI Martingale EA |
//|                                                                    |
//|                                             https://www.mql5.com    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

// Input Parameters
extern double InitialLotSize = 0.1;      // Initial Lot Size
extern double LotMultiplier = 2.0;       // Lot Size Multiplier for Losses
extern int RSI_Period = 14;              // RSI Period
extern int RSI_Overbought = 70;          // RSI Overbought Level
extern int RSI_Oversold = 30;            // RSI Oversold Level
extern int MaxTrades = 5;                // Maximum Number of Recovery Trades
extern bool UseTrailingStop = true;      // Use Trailing Stop
extern int TrailingStart = 20;           // Trailing Stop Start (pips)
extern int TrailingStep = 10;            // Trailing Stop Step (pips)

// Global Variables
datetime lastTradeTime = 0;
int magicNumber = 123456;
int currentTradeCount = 0;
double currentLotSize = InitialLotSize;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    currentLotSize = InitialLotSize;
    currentTradeCount = 0;
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Clean up
}

//+------------------------------------------------------------------+
//| Get Current RSI Value                                              |
//+------------------------------------------------------------------+
double GetRSI()
{
    return iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, 0);
}

//+------------------------------------------------------------------+
//| Manage Trailing Stop                                               |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
    if(!UseTrailingStop) return;
    
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderSymbol() == Symbol() && OrderMagicNumber() == magicNumber)
            {
                double orderOpenPrice = OrderOpenPrice();
                double currentSL = OrderStopLoss();
                
                if(OrderType() == OP_BUY)
                {
                    double newSL = Bid - TrailingStart * Point;
                    if(Bid - orderOpenPrice > TrailingStart * Point)
                    {
                        if(newSL > currentSL + TrailingStep * Point)
                        {
                            bool modified = OrderModify(OrderTicket(), orderOpenPrice, newSL, 0, 0, clrGreen);
                            if(!modified)
                                Print("Error modifying trailing stop: ", GetLastError());
                        }
                    }
                }
                else if(OrderType() == OP_SELL)
                {
                    double newSL = Ask + TrailingStart * Point;
                    if(orderOpenPrice - Ask > TrailingStart * Point)
                    {
                        if(newSL < currentSL - TrailingStep * Point || currentSL == 0)
                        {
                            bool modified = OrderModify(OrderTicket(), orderOpenPrice, newSL, 0, 0, clrRed);
                            if(!modified)
                                Print("Error modifying trailing stop: ", GetLastError());
                        }
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check for RSI Signals                                              |
//+------------------------------------------------------------------+
int CheckRSISignal()
{
    double currentRSI = GetRSI();
    
    // Check for oversold condition (buy signal)
    if(currentRSI < RSI_Oversold)
    {
        return 1;  // Buy signal
    }
    
    // Check for overbought condition (sell signal)
    if(currentRSI > RSI_Overbought)
    {
        return 0;  // Sell signal
    }
    
    return -1; // No signal
}

//+------------------------------------------------------------------+
//| Check for Trade Exit                                               |
//+------------------------------------------------------------------+
bool CheckTradeExit(int orderType)
{
    double currentRSI = GetRSI();
    
    if(orderType == OP_BUY)
    {
        // Exit buy when RSI becomes overbought
        return (currentRSI > RSI_Overbought);
    }
    else if(orderType == OP_SELL)
    {
        // Exit sell when RSI becomes oversold
        return (currentRSI < RSI_Oversold);
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if we're at the start of a new bar
    if(Time[0] == lastTradeTime)
        return;
        
    lastTradeTime = Time[0];
    
    // Manage trailing stop for existing positions
    ManageTrailingStop();
    
    // Check for trade exits
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if(OrderSymbol() == Symbol() && OrderMagicNumber() == magicNumber)
            {
                if(CheckTradeExit(OrderType()))
                {
                    bool result = OrderClose(OrderTicket(), OrderLots(), 
                                           OrderType() == OP_BUY ? Bid : Ask, 3);
                    if(!result)
                        Print("Error closing order: ", GetLastError());
                }
            }
        }
    }
    
    // Check for new entry signals
    int signal = CheckRSISignal();
    
    if(signal != -1)
    {
        // Close any existing orders
        for(int i = OrdersTotal() - 1; i >= 0; i--)
        {
            if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
            {
                if(OrderSymbol() == Symbol() && OrderMagicNumber() == magicNumber)
                {
                    bool result = OrderClose(OrderTicket(), OrderLots(), 
                                           OrderType() == OP_BUY ? Bid : Ask, 3);
                    if(!result)
                        Print("Error closing order: ", GetLastError());
                }
            }
        }
        
        if(signal == 1) // Buy signal
        {
            int ticket = OrderSend(Symbol(), OP_BUY, currentLotSize, Ask, 3, 0, 0, 
                                 "Gold RSI EA", magicNumber, 0, clrGreen);
            if(ticket > 0)
            {
                Print("Buy order placed successfully. Ticket: ", ticket);
                Print("Current Lot Size: ", currentLotSize);
                Print("Trade Count: ", currentTradeCount);
            }
            else
                Print("Buy order failed. Error: ", GetLastError());
        }
        else // Sell signal
        {
            int ticket = OrderSend(Symbol(), OP_SELL, currentLotSize, Bid, 3, 0, 0, 
                                 "Gold RSI EA", magicNumber, 0, clrRed);
            if(ticket > 0)
            {
                Print("Sell order placed successfully. Ticket: ", ticket);
                Print("Current Lot Size: ", currentLotSize);
                Print("Trade Count: ", currentTradeCount);
            }
            else
                Print("Sell order failed. Error: ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| Order close event handler                                          |
//+------------------------------------------------------------------+
void OnOrderClose()
{
    // Check if the closed order was a loss
    if(OrderProfit() < 0)
    {
        currentTradeCount++;
        if(currentTradeCount < MaxTrades)
        {
            currentLotSize *= LotMultiplier;
            Print("Loss detected. Increasing lot size to: ", currentLotSize);
            Print("Trade count: ", currentTradeCount);
        }
        else
        {
            // Reset to initial values after max trades
            currentLotSize = InitialLotSize;
            currentTradeCount = 0;
            Print("Maximum trades reached. Resetting to initial lot size: ", currentLotSize);
        }
    }
    else
    {
        // Reset to initial values after profit
        currentLotSize = InitialLotSize;
        currentTradeCount = 0;
        Print("Profit taken. Resetting to initial lot size: ", currentLotSize);
    }
} 