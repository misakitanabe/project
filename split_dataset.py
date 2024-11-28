# pip install scikit-learn

from sklearn.model_selection import train_test_split
import os
import shutil

# Paths
dataset_dir = 'dataset/fruit_images'
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'

# Create directories if not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split dataset
classes = os.listdir(dataset_dir)
for class_name in classes:
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)
    num_images = len(os.listdir(class_path))
    print(f"Class '{class_name}' has {num_images} images.")

    # Create full paths for all images
    image_paths = [os.path.join(class_path, img) for img in images]

    # Create class labels for stratification
    labels = [class_name] * len(images)

    # Perform train-validation-test split
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels
    )

    # Move files to their respective directories
    for img in train_imgs:
        dest = os.path.join(train_dir, class_name)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img, dest)

    for img in val_imgs:
        dest = os.path.join(val_dir, class_name)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img, dest)

    for img in test_imgs:
        dest = os.path.join(test_dir, class_name)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img, dest)

print("Dataset split completed successfully!")