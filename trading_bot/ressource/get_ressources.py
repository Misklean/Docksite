import pandas as pd
import numpy as np
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import talib
from datetime import datetime, timedelta
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('default.config')

# Initialize Alpaca client
client = CryptoHistoricalDataClient(
    api_key=config['alpaca']['API_KEY'],
    secret_key=config['alpaca']['API_SECRET']
)

def fetch_last_year_org(symbol):
    # Initialize client (no keys required for crypto data)
    client = CryptoHistoricalDataClient()
    
    # Create the request with multiple symbols
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,  # Changed to 5-minute intervals
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1)  # Corrected to call the function to get the current time
    )
    
    # Get the data
    bars = client.get_crypto_bars(request_params)
    
    # Convert to DataFrame and also get individual symbol data
    return bars.df

def fetch_last_year_hour(symbol):
    # Initialize client (no keys required for crypto data)
    client = CryptoHistoricalDataClient()
    
    # Create the request with multiple symbols
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,  # Changed to 5-minute intervals
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1)  # Corrected to call the function to get the current time
    )
    
    # Get the data
    bars = client.get_crypto_bars(request_params)
    
    # Convert to DataFrame and also get individual symbol data
    df = bars.df
    
    return compute_indicators(df)

def compute_indicators(df):
    # Compute indicators for all symbols and preprocess data
    if 'close' in df.columns:

        # Calculate SMA
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        
        # Calculate EMA
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['BBL_20_2.0'] = lower
        df['BBM_20_2.0'] = middle
        df['BBU_20_2.0'] = upper
        
        # Calculate RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # Normalize indicators and close price using Min-Max scaling
        columns_to_normalize = ['close', 'SMA_20', 'SMA_50', 'EMA_20', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
        for col in columns_to_normalize:
            df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        # Keep only the relevant columns
        columns_to_keep = [
            'volume',
            'trade_count',
            'close_normalized',
            'SMA_20_normalized', 
            'SMA_50_normalized',
            'EMA_20_normalized',
            'BBL_20_2.0_normalized',
            'BBM_20_2.0_normalized', 
            'BBU_20_2.0_normalized',
            'RSI_normalized',
            'MACD_normalized',
            'MACD_signal_normalized',
            'MACD_hist_normalized'
        ]

        df = df[columns_to_keep]

        # Remove any rows with NaN values that may have been created during indicator calculation
        df = df.dropna()

        df = df.reset_index()  # Reset index to access 'symbol'
        df = df.rename(columns={'timestamp': 'ds', 'close_normalized': 'y', 'symbol': 'unique_id'})

        # Ensure 'ds' is a datetime object
        df['ds'] = pd.to_datetime(df['ds'])

        return df
    
symbol = 'BTC/USD'
df = fetch_last_year_org(symbol)
df.to_csv('ressource/dataset/data_2023_original.csv', index=True)

df = fetch_last_year_hour(symbol)
df.to_csv('ressource/dataset/data_2023_hour_preprocessed.csv', index=True)

