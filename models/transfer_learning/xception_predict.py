# pip install tensorflow== 2.18.0
# pip install numpy
# pip install opencv-python

# python version: 3.11.5

# run using "python3.9 <full path>", not pressing run on IDE it doesn't use correct python environment

from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model_path = 'models/xception_finetuned.h5'

model = load_model(model_path)
# print(model.summary())

# can replace path in this function with any image to test
img = cv2.imread('dataset/fruit_images/Lime_Bad/IMG_6950.JPG')
print("original shape", img.shape)

img = cv2.resize(img, (299, 299))
print("resized shape", img.shape)

# create numpy array for the model
data = np.empty((1, 299, 299, 3))

# store our image inside the "batch" of images
data[0] = img
print("data shape for the model", data.shape)

# normalize the data
data = preprocess_input(data)


class_labels = ({
    0: "Apple_Bad", 1: "Apple_Good", 
    2: "Banana_Bad", 3: "Banana_Good", 
    4: "Guava_Bad", 5: "Guava_Good", 
    6: "Lime_Bad", 7: "Lime_Good", 
    8: "Orange_Bad", 9: "Orange_Good", 
    10: "Pomegranate_Bad", 11: "Pomegranate_Good"
})

# predict
predictions = model.predict(data)
print("\nThis is the prediction: ", class_labels[np.argmax(predictions[0])])


# {'Apple_Bad': 0, 'Apple_Good': 1, 
# 'Banana_Bad': 2, 'Banana_Good': 3, 
# 'Guava_Bad': 4, 'Guava_Good': 5, 
# 'Lime_Bad': 6, 'Lime_Good': 7, 
# 'Orange_Bad': 8, 'Orange_Good': 9, 
# 'Pomegranate_Bad': 10, 'Pomegranate_Good': 11}