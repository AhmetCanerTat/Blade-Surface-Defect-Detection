import os
import cv2
import matplotlib.pyplot as plt


def load_images(data_dir="Data_GKN"):
    images_cv2 = []
    labels_cv2 = []
    # Walk through all subdirectories in the Data_GKN folder
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        images_cv2.append(img)
                        labels_cv2.append(label)
                    else:
                        print(f"Warning: {img_path} could not be loaded by cv2.")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    print_image_count_per_label(labels_cv2, images_cv2)
    return images_cv2, labels_cv2

def print_image_count_per_label(labels_cv2, images_cv2):
    from collections import Counter, defaultdict
    import random
    label_counts = Counter(labels_cv2)
    print(f"Loaded {len(images_cv2)} images with {len(set(labels_cv2))} unique labels using cv2")
    print("Image count per label:")
    # Build a mapping from label to indices
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels_cv2):
        label_to_indices[label].append(idx)
    # Prepare to show one random image per label side by side
    n_labels = len(label_counts)
    if n_labels == 0:
        print("No labels found. Cannot display images.")
        return
    fig, axes = plt.subplots(1, n_labels, figsize=(4*n_labels, 4))
    if n_labels == 1:
        axes = [axes]
    for ax, (label, count) in zip(axes, label_counts.items()):
        print(f"  {label}: {count}")
        rand_idx = random.choice(label_to_indices[label])
        img = images_cv2[rand_idx]
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def split_data(images, labels):
    from sklearn.model_selection import train_test_split
    # First split: train vs temp (val+test)
    X_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    # Second split: validation vs test (half of temp each)
    x_val, X_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {len(X_train)}, Validation: {len(x_val)}, Test: {len(X_test)}")
    return X_train, x_val, X_test, y_train, y_val, y_test