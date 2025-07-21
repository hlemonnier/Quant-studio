import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance, kendalltau, spearmanr

# Paramètres
interval = '15m'
csv_file_in = "/home/hlemonnier/Bureau/Coding/enigma-quant/1years_ohlcv_btc_usd_spot_15MIN.csv"
csv_file_out = f"/home/hlemonnier/Bureau/Coding/enigma-quant/data/simulators/hybrid_model_ohlcv_btc_usd_{interval}.csv"

# Lecture des données historiques
btc_data = pd.read_csv(csv_file_in)
btc_data = btc_data.reset_index()
btc_data = btc_data.rename(columns={
    'timestamp': 'time_period_start',
    'open': 'price_open',
    'high': 'price_high',
    'low': 'price_low',
    'close': 'Close',
    'volume': 'volume_traded'
})
btc_data = btc_data[['time_period_start', 'price_open', 'price_high', 'price_low', 'Close', 'volume_traded']]

# Calcul du log-return réel
btc_data['Log_Return'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))

# Définition de la volatilité (historique par défaut)
def historical_volatility(log_returns, window=20):
    return log_returns.rolling(window=window).std()

def select_volatility_model(log_returns, model='historical'):
    if model == 'historical':
        return historical_volatility(log_returns)
    else:
        raise ValueError("Modèle de volatilité non reconnu")

selected_model = 'historical'
volatility = select_volatility_model(btc_data['Log_Return'], model=selected_model)
adjustment_factor = np.random.uniform(0.3, 1.5, size=len(volatility))
adjusted_volatility = volatility * adjustment_factor

# Génération des nouveaux log-returns hybrides
np.random.seed(1)
random_shocks = np.random.normal(0, 1, len(adjusted_volatility))
new_log_returns = random_shocks * adjusted_volatility

# Génération du prix hybride
close_price = btc_data['Close'].iloc[0]
btc_data['Hybrid_Close'] = close_price * np.exp(np.cumsum(np.nan_to_num(new_log_returns, nan=0.0)))
btc_data.loc[btc_data.index[0], 'Hybrid_Close'] = close_price

# Calcul du log-return hybride
btc_data['Hybrid_Log_Return'] = np.log(btc_data['Hybrid_Close'] / btc_data['Hybrid_Close'].shift(1))

# Pour les analyses de log-return : on garde uniquement les lignes non-NaN dans les deux séries
logrets_df = btc_data[['Log_Return', 'Hybrid_Log_Return']].dropna()

# Volatilités annualisées
realized_volatility = logrets_df['Log_Return'].std() * np.sqrt(360)
hybrid_volatility = logrets_df['Hybrid_Log_Return'].std() * np.sqrt(360)

# Corrélation des prix de clôture
closing_price_correlation = btc_data[['Close', 'Hybrid_Close']].corr().iloc[0, 1]

# Distances et corrélations des log-returns
if not logrets_df.empty:
    wasserstein_dist = wasserstein_distance(logrets_df['Log_Return'], logrets_df['Hybrid_Log_Return'])
    kendall_corr, _ = kendalltau(logrets_df['Log_Return'], logrets_df['Hybrid_Log_Return'])
    spearman_corr, _ = spearmanr(logrets_df['Log_Return'], logrets_df['Hybrid_Log_Return'])
else:
    wasserstein_dist = np.nan
    kendall_corr = np.nan
    spearman_corr = np.nan

print(f"Wasserstein's distance between real and hybrid log yields: {wasserstein_dist:.2f}")
print(f"Kendall correlation between real and hybrid log yields: {kendall_corr:.2f}")
print(f"Spearman correlation between real and hybrid log yields: {spearman_corr:.2f}")

# OHLCV hybrides
btc_data['Hybrid_Open'] = btc_data['Hybrid_Close'].shift(1)
btc_data.loc[btc_data.index[0], 'Hybrid_Open'] = btc_data.loc[btc_data.index[0], 'Hybrid_Close']

deviation = 1000
mean_high = np.random.uniform(0, deviation)
mean_low = np.random.uniform(0, deviation)
fluctuation_high = np.random.normal(mean_high, 1, size=len(btc_data))
fluctuation_low = np.random.normal(mean_low, 1, size=len(btc_data))

btc_data['Hybrid_High'] = btc_data[['Hybrid_Open', 'Hybrid_Close']].max(axis=1) + fluctuation_high
btc_data['Hybrid_Low'] = btc_data[['Hybrid_Open', 'Hybrid_Close']].min(axis=1) - fluctuation_low
btc_data['Hybrid_High'] = btc_data[['Hybrid_Open', 'Hybrid_Close', 'Hybrid_High']].max(axis=1)
btc_data['Hybrid_Low'] = btc_data[['Hybrid_Open', 'Hybrid_Close', 'Hybrid_Low']].min(axis=1)
btc_data['Hybrid_Volume'] = np.random.uniform(100, 15000, size=len(btc_data))

btc_data_ohlc = btc_data[['time_period_start', 'Hybrid_Open', 'Hybrid_High', 'Hybrid_Low', 'Hybrid_Close', 'Hybrid_Volume']]
btc_data_ohlc = btc_data_ohlc.rename(columns={
    'Hybrid_Open': 'price_open',
    'Hybrid_High': 'price_high',
    'Hybrid_Low': 'price_low',
    'Hybrid_Close': 'price_close',
    'Hybrid_Volume': 'volume_traded'
})
btc_data_ohlc.to_csv(csv_file_out, index=True)

print(f"Annualized Volatility of Real Prices: {realized_volatility:.2%}")
print(f"Annualized Volatility of Hybrid Prices: {hybrid_volatility:.2%}")
print(f"Correlation between Real and Hybrid Closing Prices: {closing_price_correlation:.2f}")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(btc_data['Close'], label='Prix Réel')
plt.plot(btc_data['Hybrid_Close'], label='Prix Hybride', linestyle='-')
plt.text(0.05, 0.95, f'Correlation: {closing_price_correlation:.2%}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.text(0.05, 0.90, f'Real volatility: {realized_volatility:.2%}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.text(0.05, 0.85, f'Hybrid Volatility: {hybrid_volatility:.2%}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
plt.legend(loc='upper right')
plt.title('Cours réel vs cours hybride du Bitcoin')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.show()
