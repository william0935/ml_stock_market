import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    """
    Loads stock data from yfinance.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.

    Returns:
        pandas.DataFrame: The stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)  # Explicitly set auto_adjust
    if stock_data.empty:
        print(f"No data found for ticker symbol: {ticker} in the specified date range.")
        return None
    return stock_data