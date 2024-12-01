from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir, test_dir, img_size=(299, 299), batch_size=16):
    """Loads training, validation, and test datasets."""
    # Define ImageDataGenerators
    # Normalize pixels for training, validation, and testing
    
    # Just for training:
    # data augmentation to increase diversity of dataset by applying random transformations during epochs
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
        rotation_range=30,  
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    # Load datasets
    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',    # this will one-hot encode the labels
        shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print(train_data.class_indices)

    return train_data, val_data, test_data
