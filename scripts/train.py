from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf

def build_model(num_classes):
    """Builds and compiles the Xception-based fine-tuning model."""
    # Load the base model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model initially

    # Add custom layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # dense layer
        Dense(num_classes, activation='softmax')  # Adjust for the number of classes
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model_with_kfold(data, labels, num_classes, k=10, epochs=10, batch_size=16, output_path="final_model.h5"):
    """Performs KFold cross-validation and saves the final model."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    training_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Starting fold {fold + 1}/{k}...")

        # Split data into training and validation sets
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Preprocess data using tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Build and train the model
        model = build_model(num_classes)
        model.fit(
            train_dataset,
            # validation_data=val_dataset,
            epochs=epochs
        )

        # Evaluate on validation set
        val_loss, val_accuracy = model.evaluate(val_dataset)
        training_loss, training_accuracy = model.evaluate(train_dataset)
        training_accuracies.append(training_accuracy)
        print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")
        fold_accuracies.append(val_accuracy)

    # Compute the average validation accuracy
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Validation Accuracy across {k} folds: {avg_accuracy:.4f}")
    avg_training_accuracy = np.mean(training_accuracies)
    print(f"Average Training Accuracy across {k} folds: {avg_training_accuracy:.4f}")

    # Train final model on the entire dataset
    print("Training final model on the entire training dataset...")
    full_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    full_dataset = full_dataset.shuffle(len(data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    final_model = build_model(num_classes)
    history = final_model.fit(full_dataset, epochs=epochs)

    # Save the final model
    final_model.save(output_path)
    print(f"Final model saved to {output_path}")

    return history
