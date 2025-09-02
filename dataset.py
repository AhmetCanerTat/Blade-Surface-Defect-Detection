import torch
from torch.utils.data import Dataset

class MergedImagesDataset(Dataset):
    """
    PyTorch Dataset for merged images and their labels.
    Args:
        merged_images (list of np.ndarray): List of merged images (H, W, 3).
        labels (list of str): List of string labels for each image.
        label_to_idx (dict): Mapping from label string to integer index.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, merged_images, labels, label_to_idx, transform=None):
        self.merged_images = merged_images
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.merged_images)

    def __getitem__(self, idx):
        image = self.merged_images[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        if self.transform:
            image = self.transform(image)
        else:
            # Convert numpy image (H, W, C) to torch tensor (C, H, W) and normalize to [0,1]
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, label_idx