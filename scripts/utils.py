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


# Plotting the combined distribution
def plot_combined_class_distribution(train_counts, test_counts, title, save_path=None):
    """Plots the combined distribution of images per class for train and test sets."""
    classes = list(train_counts.keys())
    train_values = list(train_counts.values())
    test_values = [test_counts.get(cls, 0) for cls in classes]

    x = range(len(classes))  # Indices for the classes
    width = 0.35  # Width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(x, train_values, width, label='Train', color='blue', alpha=0.7)
    plt.bar([i + width for i in x], test_values, width, label='Test', color='orange', alpha=0.7)

    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.xticks([i + width / 2 for i in x], classes)  # Center labels
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
    # plt.show()
