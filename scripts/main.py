# python3.9 /Users/misakitanabe/Documents/Cal\ Poly/year4/CSC\ 466/project/scripts/main.py 

from preprocess import load_data
from train import build_model, train_model_with_kfold
from evaluate import evaluate_model
from utils import plot_training_history, extract_data_and_labels

# Hyperparameters
img_size = (299, 299)
batch_size = 16
# epochs = 20
epochs = 3
# num_classes = 12  # Number of classes in the dataset
num_classes = 2

# Paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
model_path = f'models/xception_{epochs}_epochs.h5'
results_path = f"results/transfer_learning/{epochs}_epochs"

# Step 1: Load Data
train_data, test_data = load_data(train_dir, test_dir, img_size, batch_size)

# Step 2: Build Model
model = build_model(num_classes)

# Step 3: Train Model
train_images, train_labels = extract_data_and_labels(test_data) # CHANGE BACK TO TRAIN DATA LATER
transfer_history = (
    train_model_with_kfold
    (
        data=train_images,
        labels=train_labels,
        num_classes=num_classes,
        k=3,
        epochs=epochs,
        batch_size=batch_size,
        output_path=model_path
    )
)

# Step 4: Evaluate Model
evaluate_model(model_path, test_data)

# Plot Training History
plot_training_history(transfer_history, results_path)
