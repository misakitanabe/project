# python3.9 /Users/misakitanabe/Documents/Cal\ Poly/year4/CSC\ 466/project/models/transfer_learning/xception_predict.py

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model_path = 'models/transfer_learning/xception_final_model.h5'
# model_path = 'models/transfer_learning/mobilenet_final_model.h5'
model = load_model(model_path)

# Load the image
image_path = "apple.jpg"
img_size = (299, 299) 
# img_size = (224, 224) 
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

