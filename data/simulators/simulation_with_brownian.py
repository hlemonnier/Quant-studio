import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import kurtosis
from datetime import datetime

'''
This script simulates the price movement of an asset using Brownian motion for OHLCV data.
The algorithm is based on a randomly chosen cycle length (between 1 hour and 10945 hours in the example), to which we associate a bullish, bearish or neutral trend.
The whole process follows a Brownian motion. 
The data generated undergoes 2 statistical tests and a condition on the kurtosis to ensure its consistency, and is only recorded if it meets the following criteria
We add a probability of incident, which if it occurs will cause the price to crash or rise.
We also enter the volumes traded over each period, chosen at random, again with a risk of overshooting or crushing.
'''

def generate_hourly_holcv_data(start_date, end_date, base_csv_file, volatility, min_incident, max_incident, slippage, volume_crush_probability, min_crush_ratio, max_crush_ratio, eps, high_kurtosis=False):
    attempts = 0
    cycle_start_dates = []  # List to store the start dates of each cycle
    cycle_trends = []  # List to store the trends of each cycle
    while attempts < 100:  # We limit the number of iteration to avoid an endless loop if no data meets the conditions
        attempts += 1
        date_range = pd.date_range(start=start_date, end=end_date, freq='h') # Frequency of creation of the OHLCV on the selected period. (Here, one bar equals one hour)
        data = []

        open_price = 69610.46
        cycle_duration = 0
        cycle_remaining = 0
        current_cycle_amplitude = 0
        current_cycle_trend = 0

        for i, current_date in enumerate(date_range):
            if cycle_remaining == 0:
                "The effectiveness of the trend will not always be perfect, as the data is generated based on a random walk"
                cycle_start_dates.append(current_date)  # Records the start date of the new cycle
                cycle_duration = np.random.randint(1, 10945)
                current_cycle_amplitude = 5 # Or np.random.uniform(1, 10) for a random amplitude
                current_cycle_trend = np.random.choice([-1, 0, 1])
                cycle_trends.append(current_cycle_trend)  # Records the trend of the new cycle
                cycle_remaining = cycle_duration

            cycle_remaining -= 1

            incident_impact = 0
            if np.random.uniform() < 1/10945: # Probability of getting a crash is one on the number of bars during the selected period
                print('Incident at ', current_date)
                incident_impact = np.random.uniform(low=min_incident, high=max_incident)

            if high_kurtosis:
                brownian_increment = np.random.standard_t(df=3) * volatility 
            else:
                brownian_increment = np.random.normal(loc=0, scale=volatility)

            "We first calculate a high price and a low price with the condition low < high by construction and we choose a value for the close randomly between high and low"

            high_price = max(open_price * (1 + np.clip(np.random.uniform(0, volatility), -eps, eps)), 0) # Function np.clip ensures the values do not go too low high or too low too fast
            low_price = max(open_price * (1 - np.clip(np.random.uniform(0, volatility), -eps, eps)), 0) # Same here for the low price
            close_price = np.random.uniform(low=low_price, high=high_price) * (1 + brownian_increment)
            close_price *= (1 + incident_impact)
            close_price += current_cycle_trend * current_cycle_amplitude

            
            volume = np.random.uniform(low=0.00044775, high= 13161.7501) # Choosen according to real hourly BTC data on the year 23/24
            "Here we also simulate the volume traded on each bar, with a probality of increase or crash"
            if np.random.uniform() < volume_crush_probability:
                crush_ratio = np.random.uniform(min_crush_ratio, max_crush_ratio)
                volume *= crush_ratio

            data.append({
                'time_period_start': current_date,
                'time_period_end': current_date + pd.Timedelta(hours=1),
                'time_open' : current_date,
                'time_close' : current_date,
                'price_open': open_price,
                'price_high': high_price,
                'price_low': low_price,
                'price_close': close_price,
                'volume_traded': volume,
                'trades_count': volume
            })

            open_price = close_price * (1 + np.random.uniform(-slippage, slippage))

        asset_data = pd.DataFrame(data)

        # Perform statistical tests
        adf_result = adfuller(asset_data['price_close'])
        adf_statistic = adf_result[0]
        """ADF's test : Checks the stationarity of the time series."""

        acf_result = acf(asset_data['price_close'])
        lb_statistic = np.sum(np.square(acf_result[1:]))
        """Ljung-Box's test : Checks the absence of autocorrelation in the residuals of a time series."""

        # Calculate the kurtosis value
        kurtosis_statistic = kurtosis(asset_data['price_close'])
        """Kurtosis : Checks that kurtosis is less than or equal to 0."""

        # Checks conditions to get consistent data
        if adf_statistic > 0 and lb_statistic < 40 and kurtosis_statistic <= 0:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            csv_file = f"{base_csv_file}/browninan_model_ohlcv_btc_usd.csv"
            asset_data.to_csv(csv_file, index=False)
            print(f"Data successfully saved to {csv_file} with ADF statistic: {adf_statistic}, Ljung-Box statistic: {lb_statistic}, Kurtosis: {kurtosis_statistic}")
            return asset_data, cycle_start_dates, cycle_trends
        else:
            print(f"Attempt {attempts}: Failed to meet all conditions. ADF: {adf_statistic}, Ljung-Box: {lb_statistic}, Kurtosis: {kurtosis_statistic}")

    print("Failed to generate a dataset that meets all conditions within 100 attempts.")
    return asset_data, cycle_start_dates, cycle_trends

def calculate_annualized_volatility(price_series):
    "Function used to determine the volatility of the stock, editable with the parameter volatility"
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    volatility = log_returns.std() * np.sqrt(8760)  # Annualizing the hourly volatility
    return volatility

# Exemple d'utilisation
start_date = '2024-05-20 00:00'
end_date = '2025-01-31 00:00'

base_csv_file = "C:/Users/SolalDanan/enigma-labs-quant-strategy/data/data_simulation/Bownian_Model/brownian_model_csv"
hourly_holcv_data, cycle_start_dates, cycle_trends = generate_hourly_holcv_data(
    start_date=start_date,
    end_date=end_date,
    base_csv_file=base_csv_file,
    volatility=0.001, # 0.001 corresponds to a volatility of 17% more or less
    min_incident=-0.05,
    max_incident=0.05,
    slippage=0.001, # We try to recreate the real market fluctuation by adding slippage
    volume_crush_probability=0.2,
    min_crush_ratio=0.1,
    max_crush_ratio=0.5,
    high_kurtosis=True,
    eps=0.1
)

if hourly_holcv_data is not None:
    print(hourly_holcv_data.head())

    annualized_volatility = calculate_annualized_volatility(hourly_holcv_data['price_close'])
    print(f"Annualized Volatility: {annualized_volatility}")

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(hourly_holcv_data['time_period_start'], hourly_holcv_data['price_close'], label='Close Price', color='green')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    for cycle_start_date, cycle_trend in zip(cycle_start_dates, cycle_trends):
        ax1.axvline(x=cycle_start_date, color='red', linestyle='--', linewidth=0.7)
        trend_text = {1: 'Bullish', 0: 'Neutral', -1: 'Bearish'}[cycle_trend]
        ax1.text(cycle_start_date, ax1.get_ylim()[1], trend_text, rotation=90, verticalalignment='bottom', fontsize=8, color='red')

    plt.title('Close Prices and Traded Volumes Over Time')
    plt.suptitle(f'Annualized Volatility: {annualized_volatility:.2%}', fontsize=14, color='green')
    plt.show()
else:
    print("No valid dataset was generated.")
