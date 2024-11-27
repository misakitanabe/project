from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2
import os

# Load the pretrained Xception model with ImageNet weights
model = MobileNetV2(weights='imagenet')

# Directory containing the images
folder_path = 'fruit_images/Good_Quality_Fruits/Banana_Good'

# Initialize counters
banana_count = 0
total_images = 0

# Loop through each image in the folder
for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    
    # Check if it's an image file
    if not (img_name.endswith('.jpg') or img_name.endswith('.JPG') or img_name.endswith('.png') or img_name.endswith('.jpeg')):
        print(f"Skipping non-image file: {img_name}")
        continue

    # Read and process the image
    img = cv2.imread(img_path)

    # Ensure the image was read successfully
    if img is None:
        print(f"Error reading image: {img_name}")
        continue

    print(f"Processing image: {img_name}")
    total_images += 1

    # Resize the image to 224x224 for MobileNet
    img = cv2.resize(img, (224, 224))

    # Create a batch of images for the model
    data = np.empty((1, 224, 224, 3))
    data[0] = img

    # Normalize the image data
    data = preprocess_input(data)

    # Make predictions
    predictions = model.predict(data)

    # Decode the top-1 prediction
    top_prediction = decode_predictions(predictions, top=1)[0][0]  # Get the top prediction tuple
    class_name, description, score = top_prediction

    print(f"Top prediction for {img_name}: {description} ({score:.4f})")

    # Check if the predicted class is "banana"
    if description.lower() == 'banana':
        banana_count += 1

# Calculate the percentage of "banana" predictions
if total_images > 0:
    banana_percentage = (banana_count / total_images) * 100
else:
    banana_percentage = 0

# Print the results
print("\nFinal Results:")
print(f"Total images processed: {total_images}")
print(f"Total images predicted as 'banana': {banana_count}")
print(f"Percentage of 'banana' predictions: {banana_percentage:.2f}%")

# Percentage of correct 'banana' predictions with just pretrained model (no transfer learning): 10.51%
