import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Fetching ---
ticker = "AAPL"
start_date = "2022-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d') # Use current date

print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if stock_data.empty:
    print(f"No data found for ticker symbol: {ticker} in the specified date range.")
    exit()

# --- 2. Feature Engineering ---
print("\n--- Feature Engineering ---")
stock_data['Price Change'] = stock_data['Close'].diff()
# Predict NEXT day's price up (1) or down/same (0) based on CURRENT day's features
stock_data['Target'] = np.where(stock_data['Price Change'].shift(-1) > 0, 1, 0)

for lag in range(1, 6):
    stock_data[f'Return_Lag_{lag}'] = stock_data['Close'].pct_change(periods=lag)

sma_short_window = 10
sma_long_window = 50
stock_data[f'SMA_{sma_short_window}'] = stock_data['Close'].rolling(window=sma_short_window).mean()
stock_data[f'SMA_{sma_long_window}'] = stock_data['Close'].rolling(window=sma_long_window).mean()
stock_data['SMA_Crossover'] = np.where(stock_data[f'SMA_{sma_short_window}'] > stock_data[f'SMA_{sma_long_window}'], 1, 0)

rsi_window = 14
delta = stock_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
rs = gain / loss # Ensure loss is not zero to avoid inf
stock_data['RSI'] = 100 - (100 / (1 + rs))
stock_data['RSI'].replace([np.inf, -np.inf], 100, inplace=True) # Handle cases where loss is 0 initially (RSI -> 100)
stock_data['RSI'].fillna(50, inplace=True) # Fill initial NaNs with a neutral RSI

stock_data['Volume_Change_Pct'] = stock_data['Volume'].pct_change()

stock_data.dropna(inplace=True) # Critical: drop NaNs from features AND target alignment

feature_columns = [
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_4', 'Return_Lag_5',
    f'SMA_{sma_short_window}', f'SMA_{sma_long_window}', 'SMA_Crossover',
    'RSI', 'Volume_Change_Pct'
]
X = stock_data[feature_columns].copy()
y = stock_data['Target'].copy()

if X.isnull().values.any() or y.isnull().values.any():
    print("Warning: NaNs found in X or y after initial dropna. Further cleaning needed.")
    X.fillna(method='ffill', inplace=True)
    X.fillna(method='bfill', inplace=True) # Catch any NaNs at the very start
    # Re-align y if X was modified by fillna and index changed (though it shouldn't if copy() was used)
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]


print(f"Number of features: {len(X.columns)}")
print(f"Features: {X.columns.tolist()}")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
if X.empty or y.empty or X.shape[0] != y.shape[0]:
    print("X or y is empty or shapes mismatch after feature engineering and NaN handling. Exiting.")
    exit()

# --- 3. Data Splitting (Chronological) ---
print("\n--- Data Splitting ---")
train_size_pct = 0.6
val_size_pct = 0.2
train_idx = int(len(X) * train_size_pct)
val_idx = int(len(X) * (train_size_pct + val_size_pct))

X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

print(f"Train set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

if X_train.empty or X_val.empty or X_test.empty:
    print("One of the data splits is empty. Adjust data range or split percentages. Exiting.")
    exit()

# --- 4. Feature Scaling ---
print("\n--- Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 5. Logistic Regression Model ---
print("\n--- Model Training ---")
model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- 6. Model Evaluation ---
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# Combined Confusion Matrices Plot
print("\n--- Combined Confusion Matrices ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1 row, 2 columns

# Validation Confusion Matrix
cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=['Down/Same', 'Up'], yticklabels=['Down/Same', 'Up'], ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Validation Confusion Matrix', y=1.02) # Adjust title position

# Test Confusion Matrix
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=['Down/Same', 'Up'], yticklabels=['Down/Same', 'Up'], ax=axes[1])
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')
axes[1].set_title('Test Confusion Matrix', y=1.02) # Adjust title position

plt.tight_layout(pad=3.0) # Add some padding between subplots and title
plt.suptitle('Model Confusion Matrices', fontsize=16, y=0.95) # Overall title for the figure, move it down slightly
plt.show()

# Classification Reports (printed after the plots)
print(f'\n--- Validation Set Performance ---')
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy_val:.4f}')
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

print(f'\n--- Test Set Performance ---')
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {accuracy_test:.4f}')
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))


# --- 7. Feature Importance (Coefficients) ---
if hasattr(model, 'coef_'):
    print("\n--- Feature Importance (Model Coefficients) ---")
    
    # Ensure X_train (used for scaler and columns) has same columns as X (used for model)
    feature_names_for_plot = X_train.columns.tolist()
    coefficients_for_plot = model.coef_[0]

    if len(feature_names_for_plot) == len(coefficients_for_plot):
        importance_df = pd.DataFrame({
            'Feature': feature_names_for_plot,
            'Coefficient': coefficients_for_plot
        }).sort_values(by='Coefficient', ascending=False)

        print("Top 5 important features head:")
        print(importance_df.head())
        print(f"\nPlotting barplot with {len(importance_df)} features.")

        if not importance_df.empty:
            print("\n--- Barplot Debug Info ---")
            print("Features to plot (first 5):", importance_df['Feature'].head().tolist())
            print("Type of 'Feature' column:", type(importance_df['Feature']))
            if not importance_df['Feature'].empty:
                 print("Type of first element in 'Feature' column:", type(importance_df['Feature'].iloc[0]))
            is_tuple_list = [isinstance(item, tuple) for item in importance_df['Feature']]
            if any(is_tuple_list):
                print("WARNING: 'Feature' column contains tuples!")
            else:
                print("'Feature' column does not appear to contain tuples directly.")
            
            try:
                plt.figure(figsize=(12, max(6, len(importance_df) * 0.4))) # Dynamic height
                sns.barplot(
                    x='Coefficient',
                    y='Feature',
                    data=importance_df,
                    order=importance_df['Feature'].tolist(), # Explicitly set order
                    palette='coolwarm'
                )
                plt.title('Feature Importance (Logistic Regression Coefficients)')
                plt.xlabel('Coefficient Value (Log-Odds)')
                plt.ylabel('Feature Name')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            except ValueError as e:
                print(f"\nERROR plotting feature importance barplot: {e}")
                print("This can sometimes happen due to unexpected feature name structures or internal seaborn/pandas issues.")
                print("The feature importance data is still available in 'importance_df':")
                print(importance_df)
            except Exception as e_gen: # Catch any other potential plotting errors
                print(f"\nAn unexpected error occurred during feature importance plotting: {e_gen}")
                print(importance_df)

        else:
            print("Importance DataFrame is empty, skipping feature importance plot.")
    else:
        print(f"Mismatch between number of feature names ({len(feature_names_for_plot)}) and coefficients ({len(coefficients_for_plot)}). Skipping feature importance plot.")

# --- Check Class Distribution ---
print("\n--- Class Distribution ---")
print("Training Set Class Distribution:")
print(y_train.value_counts(normalize=True))
print("\nValidation Set Class Distribution:")
print(y_val.value_counts(normalize=True))
print("\nTest Set Class Distribution:")
print(y_test.value_counts(normalize=True))