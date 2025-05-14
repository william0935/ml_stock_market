import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import local modules
from data_loader import load_stock_data
from features import create_features, prepare_data
from model import train_model, evaluate_model
from visualizations import plot_confusion_matrices, plot_classification_reports, show_plot

# --- 1. Data Fetching ---
ticker = "AAPL"
start_date = "2022-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

stock_data = load_stock_data(ticker, start_date, end_date)
if stock_data is None:
    exit()

# --- 2. Feature Engineering ---
stock_data = create_features(stock_data)
X, y, feature_columns = prepare_data(stock_data)

if X.empty or y.empty or X.shape[0] != y.shape[0]:
    exit()

# --- 3. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=False) # 0.25 x 0.8 = 0.2

if X_train.empty or X_val.empty or X_test.empty:
    exit()

# --- 4. Hyperparameter Tuning and Model Training ---
best_C = None
best_model = None
best_scaler = None
best_val_accuracy = 0.0
best_train_sizes = None
best_train_scores_mean = None
best_val_scores_mean = None

C_values = [0.001, 0.01, 0.1, 1, 10, 100]  # Example C values

for C in C_values:
    model, scaler, train_sizes, train_scores_mean, val_scores_mean = train_model(X_train, y_train, X_val, y_val, C=C)

    # Evaluate on validation set
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"C: {C}, Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_C = C
        best_model = model
        best_scaler = scaler
        best_train_sizes = train_sizes
        best_train_scores_mean = train_scores_mean
        best_val_scores_mean = val_scores_mean

print(f"\nBest C: {best_C}, Best Validation Accuracy: {best_val_accuracy:.4f}")

# --- 5. Plot Learning Curves ---
plt.figure(figsize=(10, 6))
plt.plot(best_train_sizes, best_train_scores_mean, label='Training Accuracy')
plt.plot(best_train_sizes, best_val_scores_mean, label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Model Evaluation ---
y_val_pred, y_test_pred, val_report, test_report, X_val_scaled, X_test_scaled = evaluate_model(best_model, best_scaler, X_val, y_val, X_test, y_test)

# --- 7. Visualization ---
fig, axes = plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred)
plot_classification_reports(axes, val_report, test_report)
show_plot(fig)