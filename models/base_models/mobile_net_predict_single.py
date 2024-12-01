# pip install tensorflow== 2.18.0
# pip install numpy
# pip install opencv-python

# python version: 3.11.5

# run using "python3.9 <full path>", not pressing run on IDE it doesn't use correct python environment

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2

# model pretrained on imagenet data so can only be used as of now for these 1000 classes:
# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
model = MobileNetV2(weights='imagenet')
# print(model.summary())

# can replace path in this function with any image to test
img = cv2.imread('dataset/fruit_images/Banana_Good/IMG_0121.JPG')
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
print("\nThese are the top 5 predictions")
for name, desc, score in decode_predictions(predictions, top=5)[0]:
    print(desc, score)