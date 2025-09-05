# Debug function: Show 2 random CLAHE images from every label
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from torchvision import transforms

def preprocess_image_cv2(img, resize_shape=(256, 256)):

    # Resize
    img_resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
    # Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img_clahe = clahe.apply(img_gray)
    return img_clahe



def get_augmentation_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(), # Converts to [0,1] and (C,H,W)
    ])
    
    
    # If using CLAHE images (single channel), convert to 3 channels for compatibility
def expand_channel(img):
    if img.ndim == 2:
        return np.repeat(img[..., np.newaxis], 3, axis=-1)  # (H, W, 3)
    elif img.shape[2] == 1:
        return np.repeat(img, 3, axis=-1)
    return img


def expand_channels_for_split(x_train, x_val, x_test):
    x_train_exp = [expand_channel(img) for img in x_train]
    x_val_exp = [expand_channel(img) for img in x_val]
    x_test_exp = [expand_channel(img) for img in x_test]
    return x_train_exp, x_val_exp, x_test_exp



def preprocess_all_images(images_cv2):
    processed_results = [preprocess_image_cv2(img) for img in images_cv2]
    clahe_images = processed_results
    return clahe_images

def show_random_clahe_images_per_label(clahe_images, labels, n_per_label=2):
    from collections import defaultdict
    label_to_images = defaultdict(list)
    for img, label in zip(clahe_images, labels):
        label_to_images[label].append(img)
    for label, img_list in label_to_images.items():
        if len(img_list) == 0:
            continue
        selected_imgs = random.sample(img_list, min(n_per_label, len(img_list)))
        _, axes = plt.subplots(1, len(selected_imgs), figsize=(5*len(selected_imgs), 5))
        if len(selected_imgs) == 1:
            axes = [axes]
        for ax, img in zip(axes, selected_imgs):
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        plt.suptitle(f'Random CLAHE images for label: {label}')
        plt.show()


