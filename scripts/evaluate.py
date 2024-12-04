import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from utils import extract_data_and_labels
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data, batch_size=16):
    """Evaluates the model based on accuracy and recall."""
    model = load_model(model_path)
    test_images, test_labels = extract_data_and_labels(test_data)
    test_labels = np.argmax(test_labels, axis=1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Get predictions
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices
    y_true = np.array(test_labels)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')  # 'macro' for multi-class recall

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, recall, conf_matrix
