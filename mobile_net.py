# pip install tensorflow== 2.18.0
# pip install numpy
# pip install opencv-python

# python version: 3.11.5

# run using "python3.9 <full path>", not pressing run it doesn't use correct python environment

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2

model = MobileNetV2(weights='imagenet')
# print(model.summary())

# Directory containing the images
folder_path = 'fruit_images/Good_Quality_Fruits/Banana_Good'

# Initialize counters
banana_count = 0
total_images = 0

img = cv2.imread('fruit_images/Good_Quality_Fruits/Banana_Good/IMG_8486.JPG')
print("original shape", img.shape)

img = cv2.resize(img, (224, 224))
print("resized shape", img.shape)

# create numpy array for the model
data = np.empty((1, 224, 224, 3))

# store our image inside the "batch" of images
data[0] = img
print("data shape for the model", data.shape)

# normalize the data
data = preprocess_input(data)

# predict
predictions = model.predict(data)
print(predictions)

# get the highest value
highest_value = np.argmax(predictions, axis=1)
print(highest_value)

print("The predicted score value is :", predictions[0][highest_value])

print("These are the top 5 predictions")
for name, desc, score in decode_predictions(predictions, top=5)[0]:
    print(name, desc, score)