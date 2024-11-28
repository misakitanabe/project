from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

def build_model(num_classes):
    """Builds and compiles the Xception-based fine-tuning model."""
    # Load the base model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model initially

    # Add custom layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Optional dense layer
        Dense(num_classes, activation='softmax')  # Adjust for the number of classes
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_data, val_data, output_path, epochs=10):
    """Trains the model and saves it."""
    # Train the model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_steps=val_data.samples // val_data.batch_size
    )

    # Save the model
    model.save(output_path)
    print(f"Model saved to {output_path}")
