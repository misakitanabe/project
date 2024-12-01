from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model_path, test_data):
    """Evaluates the model on the test dataset and computes precision, recall, and accuracy."""
    # Load the trained model
    model = load_model(model_path)

    # Get predictions and true labels
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)  # Convert probabilities to class indices
    y_true = test_data.classes  # True class indices

    # Class labels
    class_labels = list(test_data.class_indices.keys())

    # Compute and display metrics
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    # Compute loss and accuracy directly
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
