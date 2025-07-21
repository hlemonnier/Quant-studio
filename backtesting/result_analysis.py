import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# Adjusting the project path and importing the runstrat function
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from parameters.config_file import Config
from backtesting.run_backtest import runstrat

# Function to get daily returns
def get_daily_returns(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    return data['Returns'].dropna()

# Assuming `returns` is a numpy array of your strategy's returns
def monte_carlo_simulation(returns, num_simulations=1000, num_days=365):
    # Initialize an array to store the results
    simulations = np.zeros((num_simulations, num_days))

    # Perform the simulations
    for i in range(num_simulations):
        # Generate a random sequence of returns
        simulated_returns = np.random.choice(returns, num_days)
        # Calculate the cumulative returns
        simulations[i, :] = np.cumprod(1 + simulated_returns) - 1

    return simulations

# Set up configuration and arguments
config = Config()
args = config.parse_args()

# Generate Monte Carlo simulations
# Parameters
ticker = 'BTC-USD'
start_date = args.fromdate
end_date = args.todate
num_simulations = 1000
num_days = 365

args.verbose = False
args.print_tearsheet = False
args.plot = False
args.write = False

# Get the daily returns from the strategy
daily_returns = runstrat(args)
strategy_simulations = monte_carlo_simulation(daily_returns, num_simulations, num_days)

# Calculate percentiles and IQR for your strategy
strategy_percentiles = np.percentile(strategy_simulations, [5, 50, 95], axis=0)
strategy_final_day_values = strategy_simulations[:, -1]
strategy_iqr = np.percentile(strategy_final_day_values, 75) - np.percentile(strategy_final_day_values, 25)
strategy_final_median = strategy_percentiles[1, -1]

# Get daily returns from BTC/USD
btc_returns = get_daily_returns(ticker, start_date, end_date)
btc_simulations = monte_carlo_simulation(btc_returns, num_simulations, num_days)

# Calculate percentiles and IQR for BTC/USD
btc_percentiles = np.percentile(btc_simulations, [5, 50, 95], axis=0)
btc_final_day_values = btc_simulations[:, -1]
btc_iqr = np.percentile(btc_final_day_values, 75) - np.percentile(btc_final_day_values, 25)
btc_final_median = btc_percentiles[1, -1]

# Compare the IQRs
print(f'IQR for the buy and hold BTC/USD strategy: {btc_iqr:.2f}%')
print(f'IQR for your strategy: {strategy_iqr:.2f}%')

# Plot 1: Monte Carlo Simulations
fig1, axs1 = plt.subplots(1, 2, figsize=(18, 6))

# Plot Monte Carlo simulations for BTC/USD
axs1[0].plot(btc_simulations.T, color='blue', alpha=0.05)
axs1[0].plot(btc_percentiles[1], color='red', label='Median', linewidth=2)
axs1[0].plot(btc_percentiles[0], color='green', linestyle='--', label='5th Percentile', linewidth=2)
axs1[0].plot(btc_percentiles[2], color='green', linestyle='--', label='95th Percentile', linewidth=2)
axs1[0].fill_between(range(num_days), btc_percentiles[0], btc_percentiles[2], color='green', alpha=0.1)
axs1[0].set_title('Monte Carlo Simulations of BTC/USD Performance (Buy and Hold)')
axs1[0].set_xlabel('Days')
axs1[0].set_ylabel('Cumulative Returns')
axs1[0].legend()
axs1[0].grid(True)
axs1[0].annotate(f'IQR (Final Day): {btc_iqr:.2f}%', xy=(num_days, btc_percentiles[1, -1]), 
                 xytext=(num_days * 0.75, btc_percentiles[1, -1] * 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
axs1[0].annotate(f'Median (Final Day): {btc_final_median:.2f}%', xy=(num_days, btc_final_median), 
                 xytext=(num_days * 0.75, btc_final_median * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# Plot Monte Carlo simulations for your strategy
axs1[1].plot(strategy_simulations.T, color='blue', alpha=0.05)
axs1[1].plot(strategy_percentiles[1], color='red', label='Median', linewidth=2)
axs1[1].plot(strategy_percentiles[0], color='green', linestyle='--', label='5th Percentile', linewidth=2)
axs1[1].plot(strategy_percentiles[2], color='green', linestyle='--', label='95th Percentile', linewidth=2)
axs1[1].fill_between(range(num_days), strategy_percentiles[0], strategy_percentiles[2], color='green', alpha=0.1)
axs1[1].set_title('Monte Carlo Simulations of Your Strategy Performance')
axs1[1].set_xlabel('Days')
axs1[1].set_ylabel('Cumulative Returns')
axs1[1].legend()
axs1[1].grid(True)
axs1[1].annotate(f'IQR (Final Day): {strategy_iqr:.2f}%', xy=(num_days, strategy_percentiles[1, -1]), 
                 xytext=(num_days * 0.75, strategy_percentiles[1, -1] * 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
axs1[1].annotate(f'Median (Final Day): {strategy_final_median:.2f}%', xy=(num_days, strategy_final_median), 
                 xytext=(num_days * 0.75, strategy_final_median * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.tight_layout()
plt.show()

# Plot 2: Strategy Growth Curve
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot the growth curve of your strategy
cumulative_returns = np.cumprod(1 + daily_returns) - 1
ax2.plot(cumulative_returns, label='Cumulative Return')
ax2.set_title('Strategy Growth Curve')
ax2.set_xlabel('Time (Days)')
ax2.set_ylabel('Cumulative Return')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.show()


'''
Explanation of the Plots:
- Monte Carlo Simulations of Strategy Performance:
  - Median Line: Shows the typical (median) performance of the strategy over time. It gives you an idea of the typical outcome.
  - Percentile Bands: The area between the 5th and 95th percentiles represents the range within which the majority of the simulated outcomes fall.
5th Percentile (Dashed Green Line): This line represents the performance of the strategy at the 5th percentile. It shows a pessimistic scenario, where only 5% of the simulations perform worse.
95th Percentile (Dashed Green Line): This line represents the performance of the strategy at the 95th percentile. It shows an optimistic scenario, where only 5% of the simulations perform better.
Ideally, Median Line is high and intercartile spread as little as possible so that less volatility

How to Use This Graph:
- Evaluate Risk and Return: By looking at the median, 5th percentile, and 95th percentile lines, you can get a sense of the risk and return profile of the strategy. If the 5th percentile line is close to zero or negative, it indicates that there is a significant risk of low or negative returns.
- Assess Strategy Robustness: The range between the 5th and 95th percentile lines gives you an idea of the variability in strategy performance. A wide range indicates higher uncertainty and variability in returns.
- Plan for Different Scenarios: The graph helps you plan for different market conditions. By understanding the best-case, worst-case, and median scenarios, you can make more informed decisions about position sizing, risk management, and potential adjustments to the strategy.
- Compare Strategies: If you have multiple strategies, you can run Monte Carlo simulations for each and compare their performance profiles. This can help you choose the strategy with the best risk-reward balance.

Understanding the Simulation Results:
- Steady Growth: If the median line shows steady growth, it indicates that the strategy generally performs well over time.
- Tail Risks: The 5th and 95th percentile lines help you understand the tail risks. If the 5th percentile line drops significantly, it indicates potential for substantial losses in adverse scenarios.


- Strategy Growth Curve:
  - Shows the cumulative return of the strategy over time, providing a visual representation of the strategy's growth.
High Volatility: If the individual simulation paths show a lot of variability, the strategy might be subject to high volatility.

'''
