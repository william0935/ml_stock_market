import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier  # Changed import
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight  # Import compute_class_weight

# Import local modules
from data_loader import load_stock_data
from features import create_features, prepare_data
from model import evaluate_model  # Assuming evaluate_model doesn't depend on the training process
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

# --- 4. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 5. Calculate Class Weights ---
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))  # Create a dictionary

# --- 6. Model Training with SGD and Loss Tracking ---
model = SGDClassifier(
    loss='log_loss',  # Specify log loss for Logistic Regression
    #class_weight='balanced',  # Remove 'balanced'
    class_weight=class_weight_dict,  # Pass the dictionary
    random_state=42,
    max_iter=1000,
    tol=1e-3,  # Tolerance for stopping criteria
    learning_rate='adaptive',
    eta0=0.1
)

loss_values = []  # Store training loss values
val_loss_values = []  # Store validation loss values
X_train_scaled = np.asarray(X_train_scaled)
y_train = np.asarray(y_train)
X_val_scaled = np.asarray(X_val_scaled)
y_val = np.asarray(y_val)

for i in range(model.max_iter):
    model.partial_fit(X_train_scaled, y_train, classes=np.unique(y_train))  # Train one iteration

    # Calculate training loss
    y_train_pred = model.predict_proba(X_train_scaled)
    train_loss = -np.mean(y_train * np.log(y_train_pred[:, 1]) + (1 - y_train) * np.log(1 - y_train_pred[:, 1]))
    loss_values.append(train_loss)

    # Calculate validation loss
    y_val_pred = model.predict_proba(X_val_scaled)
    val_loss = -np.mean(y_val * np.log(y_val_pred[:, 1]) + (1 - y_val) * np.log(1 - y_val_pred[:, 1]))
    val_loss_values.append(val_loss)

    if i > 1 and abs(loss_values[-1] - loss_values[-2]) < model.tol:
        break

# --- 7. Plot Loss Curve ---
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)
plt.legend()  # Show legend to distinguish between lines
plt.show()

# --- 8. Model Evaluation ---
y_val_pred, y_test_pred, val_report, test_report, X_val_scaled, X_test_scaled = evaluate_model(model, scaler, X_val, y_val, X_test, y_test)

# --- 9. Visualization ---
fig, axes = plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred)
plot_classification_reports(axes, val_report, test_report)
show_plot(fig)