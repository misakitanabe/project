from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model_path = 'models/xception_3_epochs.h5'
model = load_model(model_path)

# Load the image
image_path = "IMG_3401.jpg"
img_size = (299, 299) 
image = load_img(image_path, target_size=img_size)

# Convert the image to a NumPy array
image_array = img_to_array(image)

# Normalize pixel values to [0, 1] (if done during training)
image_array = image_array / 255.0

# Add a batch dimension
image_batch = np.expand_dims(image_array, axis=0)  # Shape: (1, 299, 299, 3)

# Predict using the trained model
class_labels = ({
    0: "Bad Fruit", 1: "Good Fruit", 
})

# predict
predictions = model.predict(image_batch)
print("\nThis is the prediction: ", class_labels[np.argmax(predictions[0])])

