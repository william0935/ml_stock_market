from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def train_model(X_train, y_train, X_val, y_val, C=1.0):  # Added X_val, y_val, and C
    """
    Trains the Logistic Regression model and generates learning curves.

    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The training target.
        X_val (pandas.DataFrame): The validation features.  # Added
        y_val (pandas.Series): The validation target.  # Added
        C (float): Regularization parameter. # Added

    Returns:
        tuple: The trained model, scaler, and learning curve data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Scale validation data

    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000, C=C)  # Use C
    model.fit(X_train_scaled, y_train)

    # Generate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        np.concatenate((X_train_scaled, X_val_scaled)),  # Combine training and validation data
        np.concatenate((y_train, y_val)),  # Combine training and validation labels
        train_sizes=np.linspace(0.1, 1.0, 5),  # Use 5 points for the curve
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        shuffle=True,
        random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    return model, scaler, train_sizes, train_scores_mean, val_scores_mean

def evaluate_model(model, scaler, X_val, y_val, X_test, y_test):
    """
    Evaluates the model on the validation and test sets.

    Args:
        model (sklearn.linear_model.LogisticRegression): The trained model.
        scaler (sklearn.preprocessing.StandardScaler): The scaler used for feature scaling.
        X_val (pandas.DataFrame): The validation features.
        y_val (pandas.Series): The validation target.
        X_test (pandas.DataFrame): The test features.
        y_test (pandas.Series): The test target.

    Returns:
        tuple: y_val_pred, y_test_pred, val_report, test_report
    """
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    val_report = classification_report(y_val, y_val_pred, zero_division=0)
    test_report = classification_report(y_test, y_test_pred, zero_division=0)

    return y_val_pred, y_test_pred, val_report, test_report, X_val_scaled, X_test_scaled