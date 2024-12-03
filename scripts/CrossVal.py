import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(dataset_dir):
    data = []
    labels = []
    
    # Iterate through each class folder
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Process images in the class folder
        for img_name in os.listdir(class_path):
            # Check if it's an image file
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            
            # Read and process the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            
            # Resize the image to 224x224 for MobileNet
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            data.append(img)
            labels.append(class_name)
    
    return np.array(data), np.array(labels)

def create_transfer_model(num_classes):
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def cross_validate_with_existing_split(train_dir, n_folds=5):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use 20% of training data for validation in each fold
    )

    # Prepare cross-validation results
    cv_results = []

    # Get unique classes
    classes = sorted(os.listdir(train_dir))
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Perform cross-validation
    for fold in range(n_folds):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Create model for this fold
        model = create_transfer_model(len(classes))
        
        # Create train and validation generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            subset='training',
            shuffle=True,
            seed=fold * 42  # Different seed for each fold
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            seed=fold * 42
        )
        
        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy = model.evaluate(val_generator)
        cv_results.append(val_accuracy)
        
        print(f"Validation Accuracy for Fold {fold + 1}: {val_accuracy * 100:.2f}%")
    
    # Print overall results
    print("\nCross-Validation Results:")
    print(f"Average Accuracy: {np.mean(cv_results) * 100:.2f}%")
    print(f"Standard Deviation: {np.std(cv_results) * 100:.2f}%")
    
    return cv_results

# Run cross-validation
train_dir = 'dataset/train'
cross_validate_with_existing_split(train_dir)