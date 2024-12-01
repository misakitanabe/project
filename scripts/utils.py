import matplotlib.pyplot as plt
import os

def plot_training_history(history, output_path=None):
    # Create directories if not exist
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    """Plots training and validation accuracy/loss over epochs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    if output_path:
        plt.savefig(f"{output_path}/accuracy.jpg")

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if output_path:
        plt.savefig(f"{output_path}/loss.jpg")

    # plt.show()
