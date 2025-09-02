
import torch


def _run_epoch(model, data_loader, criterion, optimizer, device, train=True):
    """Run one epoch of training or validation."""
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(train):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model_with_val(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, early_stopper=None):
    """
    Train a model with validation, early stopping, and optional LR scheduler.
    Returns: loss and accuracy histories for train and validation.
    """
    loss_history, acc_history = [], []
    val_loss_history, val_acc_history = [], []
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_epoch_loss, val_epoch_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.4f}")
        stop = early_stopper.step(val_epoch_loss, model)
        if stop:
            print("Early stopping triggered")
            break
    return loss_history, acc_history, val_loss_history, val_acc_history