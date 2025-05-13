import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. getting the data
ticker = "AAPL"
start_date = "2019-01-01"
end_date = "2024-01-01"

print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
stock_data = yf.download(ticker, start=start_date, end=end_date)
if stock_data.empty:
    print(f"No data found for ticker symbol: {ticker} in the specified date range.")
    exit()

# 2. preprocessing the data
close_data = stock_data[['Close']].copy()
close_data['Target Prediction'] = close_data['Close'].shift(-1)
close_data.dropna(inplace=True)

X = close_data[['Close']]
y = close_data['Target Prediction']

# 3. split the data into 60/20/20 for training, validation, and testing
data_size = len(close_data)
train_size = int(data_size * 0.6)
validation_size = int(data_size * 0.2)
test_size = data_size - train_size - validation_size

X_train, y_train = X[:train_size], y[:train_size]
X_temp, y_temp = X[train_size:], y[train_size:]
X_val, y_val = X_temp[:validation_size], y_temp[:validation_size]
X_test, y_test = X_temp[validation_size:], y_temp[validation_size:]

# 4. using linear regression model to predict next day's price
model = LinearRegression()
model.fit(X_train, y_train)

# 5. outputting model statistics
model_slope = model.coef_[0]
model_intercept = model.intercept_

## 5.5 evaluate on validation set
print(f'\n--- Validation Set Performance ---')
y_val_pred = model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
print(f'Validation Mean Squared Error (MSE): {mse_val:.4f}')
print(f'Validation R-squared (R2): {r2_val:.4f}')

# 6. make predictions
y_test_pred = model.predict(X_test)

## 6.5 Evaluate the model on the test set
print(f'\n--- Test Set Performance ---')
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'Test Mean Squared Error (MSE): {mse_test:.4f}')
print(f'Test R-squared (R2): {r2_test:.4f}\n')

# 7. generate the side by side plots
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# plot 1 showing the linear regression line
ax1 = axes[0]
ax1.scatter(X_test, y_test, alpha=0.7, c='blue', label='Actual Price Today vs Actual Price Tomorrow')
x_line = np.linspace(X_test['Close'].min(), X_test['Close'].max(), 100)
y_line = model_slope * x_line + model_intercept
line_label = f'Model Prediction Line (y = {model_slope:.4f}x + {model_intercept:.4f})'
ax1.plot(x_line, y_line, color='black', linestyle='--', linewidth=2, label=line_label)
ax1.set_title(f'Actual Prices vs Model Prediction Line ({ticker}) - Test Set', fontsize=14)
ax1.set_xlabel('Price Today', fontsize=12)
ax1.set_ylabel('Price Tomorrow', fontsize=12)
ax1.legend()
ax1.grid(True)

# plot 2 showing the progression of data with time
ax2 = axes[1]
ax2.plot(y_test.index, y_test, label='Actual Prices', color='blue', linewidth=2)
ax2.plot(y_test.index, y_test_pred, label='Predicted Prices (Model Output)', color='red', linestyle='--', linewidth=2)
ax2.set_title(f'Stock Price Prediction ({ticker}) - Time Series (Test Set)', fontsize=14)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Close Price', fontsize=12)
ax2.legend()
ax2.grid(True)

# 8. make prettier and show plots
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.1, top=0.9, wspace=0.25, hspace=0.2)
plt.show()
