import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time),
        "endTime": int(end_time),
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_all_ohlcv(symbol, interval, start_date, end_date, filename):
    ms_per_request = 15 * 60 * 1000 * 1000  # 1000 candles of 15min = 10_000_000 ms (6.94 jours)
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    all_klines = []

    while start_time < end_time:
        req_end_time = min(start_time + ms_per_request, end_time)
        klines = get_binance_klines(symbol, interval, start_time, req_end_time)
        if not klines:
            break
        all_klines.extend(klines)
        # Next request starts after the last candle in this batch
        start_time = klines[-1][0] + 1
        time.sleep(0.5)  # Respect API rate limits

    # Convert to DataFrame
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "15m"
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)
    filename = "1years_ohlcv_btc_usd_spot_15MIN.csv"
    fetch_all_ohlcv(symbol, interval, start_date, end_date, filename)
