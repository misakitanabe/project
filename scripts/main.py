from preprocess import load_data
from train import build_model, train_model
from evaluate import evaluate_model
from utils import plot_training_history

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'
model_path = 'models/xception_finetuned.h5'

# Hyperparameters
img_size = (299, 299)
batch_size = 16
# epochs = 20
epochs = 3
num_classes = 12  # Number of classes in the dataset

# Step 1: Load Data
train_data, val_data, test_data = load_data(train_dir, val_dir, test_dir, img_size, batch_size)

# Step 2: Build Model
model = build_model(num_classes)

# Step 3: Train Model
train_model(model, train_data, val_data, model_path, epochs=epochs)

# Step 4: Evaluate Model
evaluate_model(model_path, test_data)

# Optional: Plot Training History (if available)
# plot_training_history(history)
