# Debug function: Show 2 random CLAHE images from every label
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from torchvision import transforms

def preprocess_image_cv2(img, resize_shape=(256, 256)):
    """
    Apply the full preprocessing pipeline to a single image using cv2:
    1. Resize
    2. Convert to grayscale
    3. Gaussian blur
    4. CLAHE (adaptive histogram equalization)
    5. Sobel and Canny edge detection
    6. Merge (CLAHE, Sobel, Canny)
    Returns: merged image (3 channels), sobel image, canny image, clahe image
    """
    # Resize
    img_resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
    # Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    img_clahe = clahe.apply(img_gray)
    return img_clahe



def get_augmentation_transform():
    """
    Returns a torchvision transform for data augmentation on images (numpy arrays).
    """
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
    """
    Apply preprocess_image_cv2 to a list of images.
    Returns: merged_images_all, sobel_images_all, canny_images_all, clahe_images_all
    """
    processed_results = [preprocess_image_cv2(img) for img in images_cv2]
    clahe_images = processed_results
    return clahe_images

def show_random_clahe_images_per_label(clahe_images, labels, n_per_label=2):
    """
    Show n random CLAHE images for each label.
    Args:
        clahe_images: list of CLAHE images (grayscale, shape HxW)
        labels: list of labels (same order as images)
        n_per_label: number of images to show per label
    """
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




# def remove_black_background_and_grayscale(img, debug=False):
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img.copy()
#     # Otsu's thresholding
#     _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # Erode mask to separate blade from border
#     kernel = np.ones((5, 5), np.uint8)
#     mask_eroded = cv2.erode(mask, kernel, iterations=1)
#     # Find contours and filter by area
#     contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     h_img, w_img = mask.shape
#     min_area = 0.05 * h_img * w_img
#     blade_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
#     if blade_contours:
#         largest = max(blade_contours, key=cv2.contourArea)
#         mask_final = np.zeros_like(mask)
#         cv2.drawContours(mask_final, [largest], -1, 255, thickness=cv2.FILLED)
#         result = cv2.bitwise_and(gray, mask_final)
#         # Remove all-black rows/columns
#         coords = np.argwhere(result)
#         if coords.size > 0:
#             y0, x0 = coords.min(axis=0)
#             y1, x1 = coords.max(axis=0) + 1
#             result_cropped = result[y0:y1, x0:x1]
#         else:
#             result_cropped = result
#     else:
#         result_cropped = gray
#     if True:
#         import matplotlib.pyplot as plt
#         plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title('Gray')
#         plt.subplot(1,3,2); plt.imshow(mask_eroded, cmap='gray'); plt.title('Eroded Mask')
#         plt.subplot(1,3,3); plt.imshow(result_cropped, cmap='gray'); plt.title('Cropped Blade')
#         plt.show()
#     return result_cropped