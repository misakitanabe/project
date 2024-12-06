from sklearn.model_selection import train_test_split
from utils import plot_combined_class_distribution, clear_directory, count_images
import os
import shutil

def split_dataset():
    # Paths
    dataset_dir = 'dataset/fruit_images'
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    # Clear directories before splitting
    clear_directory(train_dir)
    clear_directory(test_dir)

    # Create directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Split dataset
    parent_classes = os.listdir(dataset_dir)  # ['bad_fruit', 'good_fruit']
    for parent_class in parent_classes:
        parent_class_path = os.path.join(dataset_dir, parent_class)
        
        # Check if it's a directory
        if not os.path.isdir(parent_class_path):
            continue

        # Get all subdirectories (e.g., 'apple_bad', 'apple_good') and collect their images
        subdirs = os.listdir(parent_class_path)
        image_paths = []
        
        for subdir in subdirs:
            subdir_path = os.path.join(parent_class_path, subdir)
            
            # Check if it's a directory
            if not os.path.isdir(subdir_path):
                continue

            # Collect all image paths from this subdirectory
            images = os.listdir(subdir_path)
            image_paths.extend([os.path.join(subdir_path, img) for img in images])

        # Create labels for stratification
        labels = [parent_class] * len(image_paths)  # Label all images as 'bad_fruit' or 'good_fruit'

        print(f"Parent class '{parent_class}' contains {len(image_paths)} images.")

        # Split into train and test (80/20 split)
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Copy images to train directory
        for img in train_imgs:
            dest = os.path.join(train_dir, parent_class)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img, dest)

        # Copy images to test directory
        for img in test_imgs:
            dest = os.path.join(test_dir, parent_class)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img, dest)

    print("Dataset split into train and test successfully!")

    # Count images in each class for train and test sets
    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    # Plot combined class distribution
    plot_combined_class_distribution(train_counts, test_counts, "Class Distribution (Train vs Test)", save_path="plots/class_distribution.jpg")
