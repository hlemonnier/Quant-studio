import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_stock_data(ticker, start, end, interval):
    try:
        stock_data = yf.download(ticker, start=start, end=end, interval=interval)
        stock_data.index = pd.to_datetime(stock_data.index)  # Convert the index to datetime
        return stock_data['Close']
    except Exception as e:
        print(f"Erreur lors du téléchargement des données pour {ticker} avec l'intervalle {interval}: {e}")
        return pd.Series()

def get_stock_data_local(path):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='time_period_start', parse_dates=True)
            df.index = pd.to_datetime(df.index)  # Assurez-vous que les dates sont des objets Timestamp
            return df['price_close']
        else:
            print(f"Le fichier {path} n'existe pas.")
            return pd.Series()
    except Exception as e:
        print(f"Erreur lors du chargement des données locales depuis {path}: {e}")
        return pd.Series()

def shift_data_from_date(data, x, start_date):
    shifted_data = data.copy()
    start_index = shifted_data.index.get_loc(start_date)
    shifted_data.iloc[start_index:] += x
    return shifted_data

crises = {
    "Bulle Internet": {"start": '2002-03-10', "end": '2002-10-09'},
    "Crise Financière 2008": {"start": '2007-11-09', "end": '2009-01-09'},
    "COVID-19": {"start": '2020-02-19', "end": '2020-03-23'}
}

use_local_data = True
bitcoin_data_path = '/Users/dnn/Enigma Securities/enigma-labs-quant-strategy/data/1years_ohlcv_btc_usd_spot/1years_ohlcv_btc_usd_spot_1HRS.csv'

selected_crisis = "Crise Financière 2008"
crisis_start = pd.to_datetime(crises[selected_crisis]["start"])
crisis_end = pd.to_datetime(crises[selected_crisis]["end"])

crisis_data = get_stock_data('^IXIC', crisis_start, crisis_end, '1d')

if crisis_data.empty:
    print("Données de crise non disponibles, veuillez vérifier l'intervalle et la période.")
    exit()

current_data = get_stock_data_local(bitcoin_data_path)

if current_data.empty:
    print("Les données locales ne sont pas disponibles. Téléchargement des données en ligne.")
    current_stock = 'BTC-USD'
    current_start = '2022-10-01'
    current_end = '2024-07-10'
    current_data = get_stock_data(current_stock, current_start, current_end, '1d')

if current_data.empty:
    print("Données actuelles non disponibles, veuillez vérifier l'intervalle et la période.")
    exit()

if len(current_data) <= len(crisis_data):
    print("Les données actuelles sont insuffisantes pour appliquer la crise sélectionnée.")
    exit()

crisis_returns = crisis_data.pct_change().dropna()
print(crisis_returns)

seed = np.random.randint(1000, high=10000)
np.random.seed(seed)
print('Graine pour ce graphe :', seed)
crisis_length = len(crisis_returns)
# Generate random start index ensuring it is within valid range
max_random_start = max(0, len(current_data) - len(crisis_returns))
random_start = np.random.randint(0, max_random_start) if max_random_start > 0 else 0

random_start_date = current_data.index[random_start]

simulated_prices = list(current_data.iloc[:random_start].values)
for i in range(crisis_length):
    if len(simulated_prices) > 0:
        simulated_price = simulated_prices[-1] * (1 + crisis_returns.iloc[i])
    else:
        # Gérer le cas où simulated_prices est vide, par exemple en initialisant à partir de current_data
        simulated_price = current_data.iloc[random_start] * (1 + crisis_returns.iloc[i])
    
    simulated_prices.append(simulated_price)

remaining_period = len(current_data) - len(simulated_prices)
if remaining_period > 0:
    last_simulated_price = simulated_prices[-1]
    random_returns = np.random.normal(0, crisis_returns.std(), remaining_period)
    for ret in random_returns:
        last_simulated_price *= (1 + ret)
        simulated_prices.append(last_simulated_price)

simulated_data = pd.DataFrame(simulated_prices, index=current_data.index, columns=['Simulated_Close'])

shift_start_date = pd.to_datetime(simulated_data.index[random_start + crisis_length])
amplitude_x = simulated_data.loc[shift_start_date, 'Simulated_Close'] - current_data.loc[shift_start_date]
shifted_simulated_data = shift_data_from_date(current_data, amplitude_x, shift_start_date)

shift_start_value = shifted_simulated_data.loc[shift_start_date]
simulated_data.loc[shift_start_date, 'Simulated_Close'] = shift_start_value

combined_simulated_data = pd.concat([
    simulated_data.loc[:shift_start_date],
    shifted_simulated_data.loc[shift_start_date:]
])

shifted_simulated_data = shifted_simulated_data.to_frame(name='Close')

def simulate_hybrid_prices(dataset, volatility_model='historical', bis=False, adjustment_factor=0.85, seed=1, deviation=1000, initial_value=None):
    dataset = dataset.copy()  # Ensure we are working with a copy to avoid SettingWithCopyWarning
    
    if not bis:
        dataset.loc[:, 'Log_Return'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
    else:
        dataset.loc[:, 'Log_Return'] = np.log(dataset['Simulated_Close'] / dataset['Simulated_Close'].shift(1))

    def historical_volatility(log_returns, window=20):
        return log_returns.rolling(window=window).std()

    def adjusted_volatility(log_returns, window=20):
        return historical_volatility(log_returns, window)

    def moving_average_volatility(log_returns, window=50):
        return log_returns.rolling(window=window).std()

    def exponential_volatility(log_returns, span=20):
        return log_returns.ewm(span=span).std()

    def select_volatility_model(log_returns, model='exponential'):
        if model == 'historical':
            return historical_volatility(log_returns)
        elif model == 'adjusted':
            return adjusted_volatility(log_returns)
        elif model == 'moving_average':
            return moving_average_volatility(log_returns)
        elif model == 'exponential':
            return exponential_volatility(log_returns)
        else:
            raise ValueError("Unknown volatility model")

    volatility = select_volatility_model(dataset['Log_Return'], model=volatility_model)
    adjusted_volatility = volatility * adjustment_factor

    np.random.seed(seed)
    random_shocks = np.random.normal(0, 1, len(adjusted_volatility))
    new_log_returns = random_shocks * adjusted_volatility

    dataset.loc[:, 'Hybrid_Close'] = initial_value * np.exp(np.cumsum(new_log_returns))
    dataset.loc[dataset.index[0], 'Hybrid_Close'] = initial_value

    dataset.dropna(inplace=True)

    dataset.loc[:, 'Hybrid_Log_Return'] = np.log(dataset['Hybrid_Close'] / dataset['Hybrid_Close'].shift(1))

    realized_volatility = dataset['Log_Return'].std() * np.sqrt(360)
    hybrid_volatility = dataset['Hybrid_Log_Return'].std() * np.sqrt(360)

    if not bis:
        closing_price_correlation = dataset[['Close', 'Hybrid_Close']].corr().iloc[0, 1]
    else:
        closing_price_correlation = dataset[['Simulated_Close', 'Hybrid_Close']].corr().iloc[0, 1]

    dataset.loc[:, 'Hybrid_Open'] = dataset['Hybrid_Close'].shift(1)
    dataset.loc[dataset.index[0], 'Hybrid_Open'] = dataset['Hybrid_Close'].iloc[0]

    fluctuation_high = np.random.normal(np.random.uniform(0, deviation), 1, size=len(dataset))
    fluctuation_low = np.random.normal(np.random.uniform(0, deviation), 1, size=len(dataset))

    dataset.loc[:, 'Hybrid_High'] = dataset[['Hybrid_Open', 'Hybrid_Close']].max(axis=1) + fluctuation_high
    dataset.loc[:, 'Hybrid_Low'] = dataset[['Hybrid_Open', 'Hybrid_Close']].min(axis=1) - fluctuation_low

    dataset.loc[:, 'Hybrid_High'] = dataset[['Hybrid_Open', 'Hybrid_Close', 'Hybrid_High']].max(axis=1)
    dataset.loc[:, 'Hybrid_Low'] = dataset[['Hybrid_Open', 'Hybrid_Close', 'Hybrid_Low']].min(axis=1)

    dataset.loc[:, 'volume_traded'] = np.random.uniform(100, 15000, size=len(dataset))

    dataset_ohlc = dataset[['Hybrid_Open', 'Hybrid_High', 'Hybrid_Low', 'Hybrid_Close', 'volume_traded']]

    return dataset_ohlc

volatility_model = 'adjusted'
if volatility_model == 'historical':
    window = 20
elif volatility_model == 'adjusted':
    window = 20
elif volatility_model == 'moving_average':
    window = 50
elif volatility_model == 'exponential':
    window = 20
else:
    raise ValueError("Unknown volatility model")

# Ensure that shifted_simulated_data has a DatetimeIndex
shifted_simulated_data.index = pd.to_datetime(shifted_simulated_data.index)

# Find the closest available date if shift_start_date is not in the index
if shift_start_date not in shifted_simulated_data.index:
    shift_start_date = shifted_simulated_data.index[shifted_simulated_data.index.searchsorted(shift_start_date)]
adjusted_start_index = max(0, shifted_simulated_data.index.get_loc(shift_start_date) - window - 1)
adjusted_start_date = shifted_simulated_data.index[adjusted_start_index]

# Ensure shift_start_date is a Timestamp
shift_start_date = pd.to_datetime(shift_start_date)
next_date = shift_start_date + pd.Timedelta(days=1)
if next_date not in shifted_simulated_data.index:
    next_date = shifted_simulated_data.index[shifted_simulated_data.index.searchsorted(next_date)]

initial_value = shifted_simulated_data.loc[next_date, 'Close']
print('Valeur initiale', initial_value)

hybrid_data = simulate_hybrid_prices(
    dataset=shifted_simulated_data.loc[shift_start_date:], 
    volatility_model=volatility_model,
    initial_value=initial_value
)

# Exclure la partie orange après le début de la période hybride
orange_end_date = shift_start_date

# S'assurer que random_start_date est un Timestamp
random_start_date = pd.to_datetime(random_start_date)

# Trouver la date de fin ajustée
if shift_start_date not in shifted_simulated_data.index:
    # Si shift_start_date n'est pas dans l'index, trouver la date la plus proche
    shift_start_date = shifted_simulated_data.index[shifted_simulated_data.index.searchsorted(shift_start_date)]
adjusted_start_index = max(0, shifted_simulated_data.index.get_loc(shift_start_date) - window - 1)
adjusted_start_date = shifted_simulated_data.index[adjusted_start_index]

# Assurer la conversion de shift_start_date en Timestamp
shift_start_date = pd.to_datetime(shift_start_date)
next_date = shift_start_date + pd.Timedelta(days=1)
if next_date not in shifted_simulated_data.index:
    next_date = shifted_simulated_data.index[shifted_simulated_data.index.searchsorted(next_date)]

initial_value = shifted_simulated_data.loc[next_date, 'Close']
print('Valeur initiale', initial_value)

hybrid_data = simulate_hybrid_prices(
    dataset=shifted_simulated_data.loc[shift_start_date:], 
    volatility_model=volatility_model,
    initial_value=initial_value
)

# S'assurer que random_start_date est un Timestamp
random_start_date = pd.to_datetime(random_start_date)

# Vérifier que random_start_date - pd.Timedelta(days=1) existe dans l'index
if (random_start_date - pd.Timedelta(days=1)) not in combined_simulated_data.index:
    random_start_date = combined_simulated_data.index[combined_simulated_data.index.searchsorted(random_start_date - pd.Timedelta(days=1))]

first_dataset = combined_simulated_data.loc[:random_start_date - pd.Timedelta(days=1)]
second_dataset = combined_simulated_data.loc[random_start_date : shift_start_date - pd.Timedelta(days=1)]
third_dataset = hybrid_data['Hybrid_Close']

# S'assurer de la continuité des données
third_dataset.iloc[0] = second_dataset['Simulated_Close'].iloc[-1]
third_dataset.index = pd.date_range(start=shift_start_date, periods=len(third_dataset), freq='1d')

# Exclure la partie orange après le début de la période hybride
simulated_data = simulated_data.loc[:orange_end_date]

concatenation = pd.concat([first_dataset['Simulated_Close'], second_dataset['Simulated_Close'], third_dataset], axis=0)
concatenation.name = 'price_close'

# Afficher les résultats
plt.figure(figsize=(14, 8))
plt.plot(current_data.index, current_data, color='blue', label='Real Market Data (BTC-USD)')
plt.plot(concatenation.index, concatenation, color='red', label='Hybrid Model Data')
plt.plot(simulated_data.index, simulated_data, color='orange', label='Data Using Crisis Model')

plt.axvline(x=random_start_date, color='red', linestyle='--', label='Crisis Random Start Date')
plt.axvline(x=adjusted_start_date, color='red', linestyle='--', label='Crisis End Date')

plt.xlabel('Date')
plt.ylabel('Prix de l\'Action')
plt.legend()
plt.show()

# Recalculate OHLC prices
ohlc_data = pd.DataFrame(index=concatenation.index)
ohlc_data['price_close'] = concatenation
ohlc_data['price_open'] = ohlc_data['price_close'].shift(1)
ohlc_data.loc[ohlc_data.index[0], 'price_open'] = ohlc_data['price_close'].iloc[0]

deviation = 1000

# Small fluctuations for high and low prices
fluctuation_high = np.random.uniform(np.random.uniform(0, deviation), 1, size=len(ohlc_data))
fluctuation_low = np.random.uniform(np.random.uniform(0, deviation), 1, size=len(ohlc_data))

ohlc_data['price_high'] = ohlc_data[['price_open', 'price_close']].max(axis=1) + fluctuation_high
ohlc_data['price_low'] = ohlc_data[['price_open', 'price_close']].min(axis=1) - fluctuation_low

ohlc_data['price_high'] = ohlc_data[['price_open', 'price_close', 'price_high']].max(axis=1)
ohlc_data['price_low'] = ohlc_data[['price_open', 'price_close', 'price_low']].min(axis=1)

ohlc_data['volume_traded'] = np.random.uniform(100, 15000, size=len(ohlc_data))

btc_data_ohlc = ohlc_data[['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']]
print(ohlc_data.head())

btc_data_ohlc.index.name = 'time_period_start'
btc_data_ohlc.reset_index(inplace=True)


csv_file_path = '/Users/dnn/Enigma Securities/enigma-labs-quant-strategy/data/data_simulation/Crisis_Model/crisis_data_simulation.csv'
btc_data_ohlc.to_csv(csv_file_path, index=True)
