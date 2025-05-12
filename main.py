import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Get the Data ---
# Define the ticker symbol for the stock you want to predict (e.g., Apple - AAPL)
# You can change this to any valid ticker symbol.
ticker_symbol = "AAPL"

# Define the date range for the historical data
# Let's get a good amount of data, e.g., 5 years
start_date = "2019-01-01"
end_date = "2024-01-01" # Use a date in the past to have a clear future for testing

print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")
try:
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    print("Data fetched successfully.")
except Exception as e:
    print(f"Error fetching data: {e}")
    # Exit if data fetching fails
    exit()

# Check if data was downloaded
if stock_data.empty:
    print(f"No data found for ticker symbol: {ticker_symbol} in the specified date range.")
    exit()

# Display the first few rows of the fetched data
print("\nSample of fetched data:")
print(stock_data.head())

# --- 2. Prepare the Data for Linear Regression ---
# For this simple model, we'll use the 'Adj Close' price as the feature (X)
# and the next day's 'Adj Close' price as the target (y).
df = stock_data[['Adj Close']].copy()

# Create the target variable by shifting the 'Adj Close' column by one day.
# This means row 'i' will have the 'Adj Close' price from row 'i+1'.
df['Prediction_Target'] = df['Adj Close'].shift(-1)

# Drop the last row because it will have a NaN (Not a Number) in the 'Prediction_Target' column
# as there is no next day's price available.
df.dropna(inplace=True)

# Display the prepared data
print("\nSample of prepared data with target variable:")
print(df.head())
print(f"\nPrepared data shape: {df.shape}")


# Define features (X) and target (y)
# X is the current day's 'Adj Close' price
X = df[['Adj Close']]
# y is the next day's 'Adj Close' price (the target)
y = df['Prediction_Target']

# --- 3. Splitting the Data (60/20/20 Chronological Split) ---
# We need to split the data chronologically to simulate real-world prediction,
# where you train on past data and predict future data.
# We'll use the index (which is the date) to perform the split.

total_size = len(df)
train_size = int(total_size * 0.6)
val_size = int(total_size * 0.2)
# The remaining data will be the test set
test_size = total_size - train_size - val_size

print(f"\nTotal data points: {total_size}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {val_size}")
print(f"Test set size: {test_size}")

# Perform the chronological split
# Training data: first 60% of the data
X_train, y_train = X[:train_size], y[:train_size]

# Temporary data for validation and test: the remaining 40%
X_temp, y_temp = X[train_size:], y[train_size:]

# Validation data: the first half of the temporary data (20% of total)
X_val, y_val = X_temp[:val_size], y_temp[:val_size]

# Test data: the second half of the temporary data (remaining 20% of total)
X_test, y_test = X_temp[val_size:], y_temp[val_size:]

print(f"\nShape of training data (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Shape of validation data (X_val, y_val): {X_val.shape}, {y_val.shape}")
print(f"Shape of test data (X_test, y_test): {X_test.shape}, {y_test.shape}")


# --- 4. Building and Training the Linear Regression Model ---
print("\nBuilding and training the Linear Regression model...")
# Create a Linear Regression model instance
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)
print("Model training complete.")

# Print the learned coefficients
print(f"\nModel Coefficient (slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

# --- 5. Making Predictions and Evaluating the Model ---

# Make predictions on the validation set
print("\nEvaluating model on validation set...")
y_val_pred = model.predict(X_val)

# Evaluate the model's performance on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Validation Set Evaluation:")
print(f"Mean Squared Error (MSE): {mse_val}")
print(f"R-squared (R2): {r2_val}")

# Make predictions on the test set
print("\nEvaluating model on test set...")
y_test_pred = model.predict(X_test)

# Evaluate the model's performance on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test Set Evaluation:")
print(f"Mean Squared Error (MSE): {mse_test}")
print(f"R-squared (R2): {r2_test}")

# --- Optional: Visualize Predictions vs Actual ---
print("\nGenerating plot of actual vs predicted prices on the test set...")
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_test_pred, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'Stock Price Prediction ({ticker_symbol}) - Linear Regression (Test Set)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)
plt.show()

print("\nScript finished.")
