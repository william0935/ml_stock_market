import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred):
    """
    Plots the confusion matrices for the validation and test sets.

    Args:
        y_val (pandas.Series): The validation target.
        y_val_pred (numpy.ndarray): The validation predictions.
        y_test (pandas.Series): The test target.
        y_test_pred (numpy.ndarray): The test predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    # Validation Confusion Matrix
    cm_val = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=['Down/Same', 'Up'], yticklabels=['Down/Same', 'Up'], ax=axes[0])  # Use axes[0]
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title('Validation Confusion Matrix')

    # Test Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='viridis', xticklabels=['Down/Same', 'Up'], yticklabels=['Down/Same', 'Up'], ax=axes[1])  # Use axes[1]
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')
    axes[1].set_title('Test Confusion Matrix')

    plt.suptitle('Model Performance', fontsize=16)
    return fig, axes

def plot_classification_reports(axes, val_report, test_report):
    pass

def show_plot(fig):
    plt.show()