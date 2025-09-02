import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



def apply_grad_cam_efficientnetv2s(model, img_array, target_class, device, mean=None, std=None):
    """
    Apply Grad-CAM to EfficientNetV2-S on Apple Silicon (MPS) or CPU.
    Args:
        model: Trained EfficientNetV2-S PyTorch model.
        img_array: Numpy array (H, W, C) in BGR or RGB format.
        target_class: Class index for which to compute Grad-CAM.
        device: torch.device('mps' or 'cpu').
        input_size: Tuple for resizing the image (default: (224, 224)).
        mean: Normalization mean (default: ImageNet mean if None).
        std: Normalization std (default: ImageNet std if None).
    Returns:
        visualization: Numpy array of the image with Grad-CAM overlay.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform(img_array).unsqueeze(0).to(device)

    # For EfficientNetV2-S, the last feature block is usually model.features[-1]
    target_layer = model.features[-1]

    # Grad-CAM (use_cuda=False for MPS/CPU)
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay CAM on image (ensure same size)
    rgb_img = np.array(img_array).astype(np.float32) / 255.0
    # Make Grad-CAM less visible (alpha=0.3 makes original image more visible)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.9)
    return visualization


# Show Grad-CAM for n random images per label
import random
import matplotlib.pyplot as plt

def show_grad_cam_for_random_images_per_label(model, images, labels, label_to_idx, device, n_per_label=2):
    """
    Show Grad-CAM results for n random images from every label.
    Args:
        model: Trained model.
        images: MergedImagesDataset or list/array of images (H, W, C) or torch.Tensor.
        labels: List of labels (same order as images) or None if using dataset.
        label_to_idx: Dict mapping label names to class indices.
        device: torch.device.
        n_per_label: Number of images per label to show.
    """
    # If images is a dataset, extract labels and indices
    if hasattr(images, '__getitem__') and hasattr(images, '__len__') and hasattr(images, 'labels'):
        # images is a dataset (e.g., MergedImagesDataset)
        dataset = images
        # If dataset.labels is a list of label names, use as is; if ints, map to names
        if isinstance(dataset.labels[0], str):
            all_labels = dataset.labels
        else:
            # Try to invert label_to_idx
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            all_labels = [idx_to_label[lbl] for lbl in dataset.labels]
        label_to_indices = {label: [] for label in label_to_idx}
        for idx, label in enumerate(all_labels):
            label_to_indices[label].append(idx)
        get_image = lambda idx: dataset[idx][0]  # returns (image, label)
    else:
        # images is a list/array/tensor, labels must be provided
        label_to_indices = {label: [] for label in label_to_idx}
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        get_image = lambda idx: images[idx]

    for label, idx_list in label_to_indices.items():
        print(f"Processing label: {label}, number of images: {len(idx_list)}")
        if len(idx_list) == 0:
            continue
        selected_indices = random.sample(idx_list, min(n_per_label, len(idx_list)))
        class_idx = label_to_idx[label]
        _, axes = plt.subplots(1, len(selected_indices), figsize=(6*len(selected_indices), 6))
        if len(selected_indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, selected_indices):
            img = get_image(idx)
            # If tensor, convert to numpy
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
                # If shape is (C, H, W), transpose to (H, W, C)
                if img.shape[0] in [1, 3]:
                    img = np.transpose(img, (1, 2, 0))
                # If single channel, repeat to 3 channels
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                # If values are in [0,1], scale to [0,255] for visualization
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            cam_image = apply_grad_cam_efficientnetv2s(model, img, class_idx, device)
            ax.imshow(cam_image)
            ax.set_title(f'Label: {label}')
            ax.axis('off')

        plt.suptitle(f'Grad-CAM for label: {label}')
        plt.show()
