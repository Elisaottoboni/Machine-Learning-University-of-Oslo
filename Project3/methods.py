from matplotlib import pyplot as plt
import numpy as np

def training_results_graph(model, model_code):
    epochs = range(1, len(model.history['loss']) + 1)
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    accuracy = model.history['accuracy']
    val_accuracy = model.history['val_accuracy']

    plt.figure(figsize=(14, 10))

    # Plot for Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, label='Training Loss', color='blue', linewidth=3)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linestyle='--', linewidth=3)
    plt.title(f'Training and Validation Loss - Model {model_code}', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot for Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy', color='blue', linewidth=3)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange', linestyle='--', linewidth=3)
    plt.title(f'Training and Validation Accuracy - Model {model_code}', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig(f'images/{model_code}_training_results.png', dpi=300, bbox_inches='tight')

    plt.show()



def plot_roc_curve(model, X_test, y_test, model_code):
    """
    Plots the ROC curve for a given model:
    Parameters:
    - model: Trained model (must have predict_proba or predict method)
    - X_test: Test data features
    - y_test: True labels for the test data
    """
    # Get predicted probabilities (for binary classification)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue', linewidth=3)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random Classifier')
    plt.title(f'ROC Curve - Model {model_code}', fontsize=15)
    plt.xlabel('False Positive Rate (FPR)', fontsize=15)
    plt.ylabel('True Positive Rate (TPR)', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Save the figure with high resolution
    plt.savefig(f'images/{model_code}_roc_curve.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, name, epochs=25):
    # Plotting
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_loss_list, label="Train Loss", color='blue', linewidth=3)
    plt.plot(range(1, epochs + 1), test_loss_list, label="Test Loss", color='orange', linestyle='--', linewidth=3)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title(f"Loss vs. Epochs")
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_acc_list, label="Train Accuracy", color='blue', linewidth=3)
    plt.plot(range(1, epochs + 1), test_acc_list, label="Test Accuracy", color='orange', linestyle='--', linewidth=3)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title(f"Accuracy vs. Epochs")
    
    plt.tight_layout()
    plt.savefig(f'images/{name}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()