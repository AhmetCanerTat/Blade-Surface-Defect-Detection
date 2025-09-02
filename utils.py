import torch

class EarlyStoppingWithLR:
    """
    Combines early stopping and ReduceLROnPlateau scheduler.
    """
    def __init__(self, optimizer, patience=5, lr_patience=2, factor=0.5, min_lr=1e-7, verbose=True):
        self.patience = patience
        self.lr_patience = lr_patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        self.counter = 0
        self.lr_counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.optimizer = optimizer

    def step(self, val_loss, model):
        stop = False
        # Early stopping logic
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.lr_counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            self.lr_counter += 1
            if self.lr_counter >= self.lr_patience:
                self._reduce_lr()
                self.lr_counter = 0
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                if self.best_state is not None:
                    model.load_state_dict(self.best_state)
                stop = True
        return stop

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")


# Function to save the best model state
def save_best_model_state(best_state, filepath):
    """
    Save the best model state_dict to a file.
    Args:
        best_state: state_dict of the best model (from EarlyStoppingWithLR.best_state)
        filepath: path to save the .pth file
    """
    torch.save(best_state, filepath)
    print(f"Best model state saved to {filepath}")
