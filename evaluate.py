import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch

macro = "macro avg"
weighted = "weighted avg"

def plot_training_curves(loss_history, acc_history, val_loss_history, val_acc_history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(all_test_labels, all_test_preds, unique_labels):
    """
    Plot the confusion matrix for predictions and true labels.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    _, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Test Set Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    plt.show()

def evaluate_model(model, test_loader, device, unique_labels):
    all_test_preds = []
    all_test_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(all_test_labels, all_test_preds, unique_labels)
    classification_metrics(all_test_labels, all_test_preds, unique_labels)
    

def classification_metrics(all_test_labels, all_test_preds, unique_labels):
    report = classification_report(all_test_labels, all_test_preds, target_names=unique_labels, digits=3, zero_division=0, output_dict=True)
    log_classification_table(report, unique_labels)

    
    
def log_classification_table(report, unique_labels):
    """Log classification metrics as a table to MLflow."""
    import pandas as pd
    
    # Create table data
    table_data = []
    for label in unique_labels:
        table_data.append({
            'Class': label,
            'Precision': round(report[label]['precision'], 3),
            'Recall': round(report[label]['recall'], 3),
            'F1-Score': round(report[label]['f1-score'], 3),
            'Support': report[label]['support']
        })
    
    # Add macro and weighted averages
    table_data.append({
        'Class': macro,
        'Precision': round(report[macro]['precision'], 3),
        'Recall': round(report[macro]['recall'], 3),
        'F1-Score': round(report[macro]['f1-score'], 3),
        'Support': report[macro]['support']
    })
     # Add accuracy row
    table_data.append({
        'Class': 'accuracy',
        'Precision': '',  # Not applicable for accuracy
        'Recall': '',     # Not applicable for accuracy
        'F1-Score': round(report['accuracy'], 3),
        'Support': report['macro avg']['support']  # Total support
    })
    
    table_data.append({
        'Class': weighted,
        'Precision': round(report[weighted]['precision'], 3),
        'Recall': round(report[weighted]['recall'], 3),
        'F1-Score': round(report[weighted]['f1-score'], 3),
        'Support': report[weighted]['support']
    })
    
    # Create DataFrame and log as table
    df = pd.DataFrame(table_data)
    
    # Log as MLflow table
    mlflow.log_table(data=df, artifact_file="artifacts/classification_report.json")
    
    # Also save as CSV artifact
    df.to_csv("artifacts/classification_report.csv", index=False)
    mlflow.log_artifact("artifacts/classification_report.csv")