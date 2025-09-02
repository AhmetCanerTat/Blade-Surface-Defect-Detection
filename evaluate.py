from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch


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



def evaluate_model(model, test_loader, device, unique_labels):
    """
    Evaluate the model on the test set, plot confusion matrix, and print classification report.
    """
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

    # Confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Test Set Confusion Matrix")
    plt.show()

    # Classification report for per-label metrics
    report = classification_report(all_test_labels, all_test_preds, target_names=unique_labels, digits=3, zero_division=0)
    print("Classification Report:\n")
    print(report)


