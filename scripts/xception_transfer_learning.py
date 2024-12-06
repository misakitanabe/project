# python3.9 /Users/misakitanabe/Documents/Cal\ Poly/year4/CSC\ 466/project/scripts/main.py 

from preprocess import load_data
from train import build_model, train_model_with_kfold
from evaluate import evaluate_model
from utils import plot_training_history, extract_data_and_labels
from split_dataset import split_dataset

# Hyperparameters
img_size = (299, 299)
batch_size = 16
epochs = 10
num_classes = 2 # Number of classes in the dataset
k = 3

# Paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
model_path = f'models/xception_{epochs}_epochs.h5'
results_path = f"plots/xception_{epochs}_epochs/cross_validation"

# Step 1: Load Data
split_dataset()
train_data, test_data = load_data(train_dir, test_dir, img_size, batch_size)

# Step 2: Build Model
model = build_model(num_classes, 'xception')

# Step 3: Train Model
train_images, train_labels = extract_data_and_labels(train_data)
transfer_history = (
    train_model_with_kfold
    (
        data=train_images,
        labels=train_labels,
        num_classes=num_classes,
        k=k,
        epochs=epochs,
        batch_size=batch_size,
        output_path=model_path
        base_model='xception'
    )
)

# Step 4: Evaluate Model
evaluate_model(model_path, test_data, batch_size, epochs)

# Plot Training History
plot_training_history(transfer_history, results_path, 'xception')
