## Enigma Labs Quant Trading Strategy

Welcome to the Enigma Labs Quantitative Trading Strategy, developed custmizing the BackTrader package, which is a framework designed to facilitate robust and realistic backtesting of trading strategies, providing comprehensive logs, plots, and reports to help traders analyze and refine their quantitative strategies.

## Comprehensive Backtesting Framework Capabilities: 
**Plots:** Representations of key metrics and strategy performance using BackTrader Observers (see screenshot in the backtesting_result folder)
  - Broker (cash vs value)
  - Trades (winning and losing trade)
  - Drawdown
  - Drawdown Length
  - Order Observer (to see when LIMIT orders expired)
  - Strategy vs Benchmark (Hold&Wait on BTC-USD)
  - BTC-USD Price with Moving Averages, Buy and Sell (+ volume) 

**In-depth generated Trading Journal Reports including:**
- Strategy Information : Strategy used, Start Date, End Date, Sizers type, Broker Comission, Slippage, Order_Type
- Key Performance Indicators (KPIs): PnL, Sharpe Ratio, Sortino Ratio, Volatility, MaxDrawdown, Cumulative Returns and more.
- Trade Analysis: Win/Loss Ratio, Trade Duration, PnL etc.
- Logs: Detailed logs of backtesting results.
- Daily Returns: Performance on a daily basis including : date, return (%), return in cash, nb_transactions, portefolio_values
- Daily Positions: Daily held positions including : date, nominal, cash, quantitiy
- Transactions: All transactions executed including : transaction_id, date, symbol, quantity, price, nominal, order_type (Market, Limit, Stop, StopTrail..), trade_type (entry, close), broker_commission

**Comparative Analysis**
The framework integrates the Quantstats and Pyfolio packages to generate an HTML report, providing additional KPIs and a deeper comparison between the strategy used and a Hold & Wait strategy on BTC-USD.


## Quantitative Trading Strategies
We include several strategies including a Buy&Hold Strategy, a crossover Long/Short Strategy, a machine learning + signal Long/Short Strategy. There is also the beginning of a grid strategy.


## Risk Management

## Money Management
### Position Sizer
The Strategy includes sevral build-in sizers to determine the size of the next trade. In this regards, we'v built a Risk Adjusted Sizer that adjust the size of our entry order by taking the volatility into account.

### Position Management
The trading strategy integrates a Position Management system designed to dynamically scale in and out of positions based on current market conditions, volatility, and momentum. (We try to incorporate a grid logic here)

## Order Management
### Trailing Strategy
We have built a Trailing Strategy within our framework that implements a dynamic trailing strategy designed to safeguard profits and manage risks. It automatically adjusts stop loss and take profit levels based on two primary functionalities: initial threshold-based adjustments to secure our gain profit and volatility-driven adjustments (Chandelier Exit).

##  Signal
We use several classical indicators such as EMA, Bollinger Bands and Stochastic Oscillator as well as a Machine Learning signal. All these signal are called in a Signal Geenerator class used in our strategy.
### Machine Learning
We created a stacking classifier for price direciton prediction. We also tried to do a price prediction using Time Series algorithm like ARIMA and TimeGPT (transoformers trained to time series).

## Data
## Database Creation
We created a PostgreSQL database with Coinbase OHLCV data of BTC-USD using CoinAPI with different timeframe (1MIN, 5MIN, 15MIN, 30MIN, 1HRS, 4HRS). We included the script the creat this database and a the script to validate the data. Each time the Script is run, the database will be updated until tooday 00h00.

##  Data Simulation
This module generates synthetic hourly/daily OHLCV data to support trading strategy testing and validation.

**Hybrid Model:** based on market data with a random factor on the volatility.

- Logarithmic Returns: Utilizes cumulative logarithmic returns to create realistic price trajectories.
- Volatility models: Offers four volatility calculations (historical, adjusted, moving average, exponential) to adjust price simulations.
- Random Shocks: Introduces randomness to reflect sudden market fluctuations.
- Correlation checks (targeting correlations in the range of 60-70% to maintain realism). 

**Crisis model:** implements crisis data to real BTC data.
- Financial crisis : Defines periods of financial crises (e.g., "Bulle Internet", "Crise Financière 2008", "COVID-19") with their start and end dates.
- Percentage changes : Applies the percentage changes of the NASDAQ during the crisis at every interval to the BTC data
- Then use the previously introduced Hybrid Model 


## Optimization Module
The Trading Strategy Optimization Module is an essential tool for evaluating the robustness and potential success of trading strategies over time. By employing various optimization techniques, this module helps chosse our strategy's hyperparameter selection.

Features and Optimization Techniques
This module integrates four distinct optimization techniques:

- **Grid Search Optimization:** Exhaustively explore all possible combinations of parameters within a defined grid.
- **Random Search Optimization:** To efficiently search the parameter space by sampling combinations randomly.
- **Bayesian Optimization:** Utilizes a probabilistic model (Gaussian processes) to predict and select the most promising parameters based on past evaluations.
- **Genetic Algorithm Optimization:**  Mimics natural selection through operations like mutation and crossover to evolve solutions iteratively.

The module track the best parameter, the KPI, the time taken for each optimization process and quantifies the number of iterations or evaluations performed. This data is crucial for comparing the efficiency and effectiveness of each optimization technique. We have included plots that compare the Sharpe ratio, across different optimization methods, providing a clear visual representation of each method's efficacy.

## Result Analysis
We included tools to analyze our results such as a MonteCarlo Simulation and cumulative return plots.

## Getting Started
### Prerequisites
Ensure you have the following packages installed:
- BackTrader
- Quantstats
- Pyfolio

