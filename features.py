import pandas as pd
import numpy as np

def create_features(stock_data, sma_short_window=10, sma_long_window=50, rsi_window=14):
    """
    Creates features from the stock data.

    Args:
        stock_data (pandas.DataFrame): The stock data.
        sma_short_window (int): The short SMA window.
        sma_long_window (int): The long SMA window.
        rsi_window (int): The RSI window.

    Returns:
        pandas.DataFrame: The stock data with features.
    """
    stock_data['Price Change'] = stock_data['Close'].diff()
    stock_data['Target'] = np.where(stock_data['Price Change'].shift(-1) > 0, 1, 0)

    for lag in range(1, 6):
        stock_data[f'Return_Lag_{lag}'] = stock_data['Close'].pct_change(periods=lag)

    stock_data[f'SMA_{sma_short_window}'] = stock_data['Close'].rolling(window=sma_short_window).mean()
    stock_data[f'SMA_{sma_long_window}'] = stock_data['Close'].rolling(window=sma_long_window).mean()
    stock_data['SMA_Crossover'] = np.where(stock_data[f'SMA_{sma_short_window}'] > stock_data[f'SMA_{sma_long_window}'], 1, 0)

    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    stock_data.loc[:, 'RSI'] = stock_data['RSI'].replace([np.inf, -np.inf], 100)  # Use .loc
    stock_data.loc[:, 'RSI'] = stock_data['RSI'].fillna(50)  # Use .loc

    stock_data['Volume_Change_Pct'] = stock_data['Volume'].pct_change()

    stock_data.dropna(inplace=True)
    return stock_data

def prepare_data(stock_data):
    """
    Prepares the data for modeling.

    Args:
        stock_data (pandas.DataFrame): The stock data with features.

    Returns:
        tuple: X (features), y (target), feature_columns (list of feature names)
    """
    feature_columns = [
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_4', 'Return_Lag_5',
        f'SMA_10', f'SMA_50', 'SMA_Crossover',
        'RSI', 'Volume_Change_Pct'
    ]
    X = stock_data[feature_columns].copy()
    y = stock_data['Target'].copy()

    if X.isnull().values.any() or y.isnull().values.any():
        X.fillna(method='ffill', inplace=True)
        X.fillna(method='bfill', inplace=True)
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

    return X, y, feature_columns