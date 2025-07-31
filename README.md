import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def compute_rsi(close, period=14):
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return pd.Series(np.nan, index=close.index)

def fetch_data(symbol, style, max_retries=3):
    if not symbol:
        print("❌ No symbol provided")
        return pd.DataFrame()
    
    # Remove .NS/.BO if already present to avoid duplication
    clean_symbol = symbol.split('.')[0].upper()
    exchange = '.NS' if '.' not in symbol else '.' + symbol.split('.')[-1]
    full_symbol = clean_symbol + exchange
    
    interval = '5m' if style == 'intraday' else '1d'
    period = '7d' if style == 'intraday' else '1y'
    
    for attempt in range(max_retries):
        try:
            print(f"⌛ Fetching {full_symbol} data ({style})...")
            data = yf.download(
                tickers=full_symbol,
                interval=interval,
                period=period,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for {full_symbol} (possibly delisted or wrong symbol)")
                
            # Ensure all required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = data['Close'] if col != 'Volume' else 0
            return data
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached for {full_symbol}")
                return pd.DataFrame()
            continue

def generate_signals(data):
    if data.empty:
        return data

    data = data.copy()
    
    # Calculate indicators
    data['EMA20'] = data['Close'].ewm(span=20, min_periods=20).mean()
    data['EMA50'] = data['Close'].ewm(span=50, min_periods=50).mean()
    data['RSI'] = compute_rsi(data['Close'])

    # Handle volume with proper alignment
    if 'Volume' not in data.columns or data['Volume'].isnull().all():
        data['Volume'] = data['Close'].mean()
        data['VolMA20'] = data['Volume']
        volume_filter = pd.Series(True, index=data.index)
    else:
        data['Volume'] = data['Volume'].ffill().fillna(0)
        data['VolMA20'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Ensure proper alignment before comparison
        vol_aligned, volma_aligned = data['Volume'].align(data['VolMA20'])
        volume_filter = (vol_aligned > (volma_aligned * 1.2)).reindex(data.index, fill_value=False)

    # Generate signals
    data['Signal'] = 0
    
    # Create aligned conditions
    ema20_gt_ema50 = data['EMA20'] > data['EMA50']
    rsi_lt_30 = data['RSI'] < 30
    close_gt_open = data['Close'] > data['Open']
    buy_condition = ema20_gt_ema50 & rsi_lt_30 & volume_filter & close_gt_open

    ema20_lt_ema50 = data['EMA20'] < data['EMA50']
    rsi_gt_70 = data['RSI'] > 70
    close_lt_open = data['Close'] < data['Open']
    sell_condition = ema20_lt_ema50 & rsi_gt_70 & volume_filter & close_lt_open

    # Apply signals
    data.loc[buy_condition, 'Signal'] = 1
    data.loc[sell_condition, 'Signal'] = -1
    
    # Remove consecutive duplicates
    data['Signal'] = data['Signal'].where(data['Signal'].diff().ne(0), 0)
    
    return data

def plot_chart(data, symbol, style):
    if data.empty:
        print("No data to plot")
        return

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(16, 12))
    
    # Plotting code remains the same as your original
    # ... [rest of your plotting code] ...

def analyze(symbol, style):
    print(f"\n{'='*50}")
    print(f"Analyzing {symbol} ({style} strategy)")
    print(f"{'='*50}")
    
    try:
        data = fetch_data(symbol, style)
        if data.empty:
            print(f"❌ Failed to fetch data for {symbol}")
            return
            
        print(f"📊 Data Range: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Data Points: {len(data)}")
        
        data = generate_signals(data)
        
        num_buy = len(data[data['Signal'] == 1])
        num_sell = len(data[data['Signal'] == -1])
        print(f"🔔 Signals: {num_buy} BUY | {num_sell} SELL")
        
        plot_chart(data, symbol, style)
        
        if not data.empty and 'Signal' in data:
            returns = data['Close'].pct_change()
            strategy_returns = returns * data['Signal'].shift(1)
            cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            print(f"📊 Strategy Return: {cumulative_return:.2%}")
            
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jarvis Trading Bot')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g.: RELIANCE, TATASTEEL.BO, AAPL)')
    parser.add_argument('--style', type=str, choices=['intraday', 'longterm'], 
                       default='intraday', help='Trading style')
    
    args = parser.parse_args()
    analyze(args.symbol, args.style)
