from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data):
    """Evaluates the model on the test dataset."""
    # Load the trained model
    model = load_model(model_path)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
