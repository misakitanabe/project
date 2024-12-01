from preprocess import load_data
from train import build_model, train_model
from evaluate import evaluate_model
from utils import plot_training_history

# Hyperparameters
img_size = (299, 299)
batch_size = 16
# epochs = 20
epochs = 10
num_classes = 12  # Number of classes in the dataset

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'
model_path = f'models/xception_{epochs}_epochs.h5'
results_path = f"results/transfer_learning/{epochs}_epochs"

# Step 1: Load Data
train_data, val_data, test_data = load_data(train_dir, val_dir, test_dir, img_size, batch_size)

# Step 2: Build Model
model = build_model(num_classes)

# Step 3: Train Model
transfer_history = train_model(model, train_data, val_data, model_path, epochs=epochs)

# Step 4: Evaluate Model
evaluate_model(model_path, test_data)

# Plot Training History
plot_training_history(transfer_history, results_path)
