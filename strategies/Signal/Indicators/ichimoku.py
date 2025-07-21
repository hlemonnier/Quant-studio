import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
 
# Step 1: Import Data
def fetch_data(ticker, start_date, end_date, interval='1d'):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data
 
# Step 2: Calculate Ichimoku Components
def ichimoku(data):
    if len(data) < 52:
        return data
   
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    data['tenkan_sen'] = (high_9 + low_9) / 2
   
    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    data['kijun_sen'] = (high_26 + low_26) / 2
   
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
   
    high_52 = data['High'].rolling(window=52).max()
    low_52 = data['Low'].rolling(window=52).min()
    data['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
   
    data['chikou_span'] = data['Close'].shift(-26)
   
    return data
 
# Step 3: Generate Trading Signals
def generate_signals(data):
    buy_signals = []
    sell_signals = []
    for i in range(len(data)):
        if not pd.isna(data['senkou_span_a'][i]) and not pd.isna(data['senkou_span_b'][i]):
            if data['Close'][i] > data['senkou_span_a'][i] and data['Close'][i] > data['senkou_span_b'][i]:
                if data['tenkan_sen'][i] > data['kijun_sen'][i]:
                    buy_signals.append(data['Close'][i])
                    sell_signals.append(np.nan)
                else:
                    buy_signals.append(np.nan)
                    sell_signals.append(np.nan)
            elif data['Close'][i] < data['senkou_span_a'][i] and data['Close'][i] < data['senkou_span_b'][i]:
                if data['tenkan_sen'][i] < data['kijun_sen'][i]:
                    sell_signals.append(data['Close'][i])
                    buy_signals.append(np.nan)
                else:
                    buy_signals.append(np.nan)
                    sell_signals.append(np.nan)
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
   
    data['Buy_Signal'] = buy_signals
    data['Sell_Signal'] = sell_signals
    return data
 
# Step 4: Backtest the Strategy
def backtest_strategy(data):
    initial_balance = 1000000
    balance = initial_balance
    position = 0  # 1 means holding, 0 means not holding
    equity_curve = []
   
    for i in range(len(data)):
        if not pd.isna(data['Buy_Signal'][i]) and data['Buy_Signal'][i] > 0 and position == 0:
            position = balance / data['Close'][i]
            balance = 0
        elif not pd.isna(data['Sell_Signal'][i]) and data['Sell_Signal'][i] > 0 and position > 0:
            balance = position * data['Close'][i]
            position = 0
        equity = balance if balance > 0 else position * data['Close'][i]
        equity_curve.append(equity)
   
    data['Equity_Curve'] = equity_curve
    final_balance = equity_curve[-1] if equity_curve else initial_balance
    return data, final_balance
 
# Step 5: Performance Metrics
def performance_metrics(data, initial_balance):
    if len(data['Equity_Curve'].dropna()) == 0:
        return {
            "Total Return": 0,
            "Annualized Return": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0
        }
   
    total_return = (data['Equity_Curve'].iloc[-1] - initial_balance) / initial_balance
    num_days = (data.index[-1] - data.index[0]).days
    num_years = num_days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
   
    data['Daily_Return'] = data['Equity_Curve'].pct_change()
    avg_daily_return = data['Daily_Return'].mean()
    std_daily_return = data['Daily_Return'].std()
    sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0
   
    data['Rolling_Max'] = data['Equity_Curve'].cummax()
    data['Drawdown'] = data['Equity_Curve'] / data['Rolling_Max'] - 1
    max_drawdown = data['Drawdown'].min()
   
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
 
# Step 6: Visualization
def plot_ichimoku(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='black')
    plt.plot(data['tenkan_sen'], label='Tenkan-sen', color='blue', linestyle='--')
    plt.plot(data['kijun_sen'], label='Kijun-sen', color='red', linestyle='--')
    plt.plot(data['senkou_span_a'], label='Senkou Span A', color='green', linestyle='--')
    plt.plot(data['senkou_span_b'], label='Senkou Span B', color='orange', linestyle='--')
    plt.fill_between(data.index, data['senkou_span_a'], data['senkou_span_b'], where=data['senkou_span_a'] >= data['senkou_span_b'], color='green', alpha=0.3)
    plt.fill_between(data.index, data['senkou_span_a'], data['senkou_span_b'], where=data['senkou_span_a'] < data['senkou_span_b'], color='red', alpha=0.3)
    plt.scatter(data.index, data['Buy_Signal'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(data.index, data['Sell_Signal'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.legend(loc='best')
    plt.show()
 
def plot_equity_curve(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Equity_Curve'], label='Equity Curve', color='blue')
    plt.fill_between(data.index, data['Equity_Curve'], color='blue', alpha=0.3)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend(loc='best')
    plt.show()
 
# Main Execution
ticker = 'BTC-USD'  # Use 'BTC-USD' for Bitcoin
start_date = '2020-01-01'
end_date = '2024-05-01'
data = fetch_data(ticker, start_date, end_date, interval='1d')  # Use daily data
if not data.empty:
    data = ichimoku(data)
    data = generate_signals(data)
    data, final_balance = backtest_strategy(data)
    metrics = performance_metrics(data, 10000)
 
    print(f"Final balance: ${final_balance:.2f}")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Annualized Return: {metrics['Annualized Return']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
 
    plot_ichimoku(data)
    plot_equity_curve(data)
else:
    print("No data available for the given period and interval.")
 
 