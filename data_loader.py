import os
import cv2


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
    print(f"Loaded {len(images_cv2)} images with {len(set(labels_cv2))} unique labels using cv2")
    return images_cv2, labels_cv2


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