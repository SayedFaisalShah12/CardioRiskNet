import numpy as np
import tensorflow as tf
from model import create_model
import matplotlib.pyplot as plt
import os

def train_network():
    print("Loading preprocessed data...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    
    # Create model
    input_shape = X_train.shape[1]
    model = create_model(input_shape)
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/cardiorisknet_model.h5')
    print("Model saved to 'models/cardiorisknet_model.h5'")
    
    # Save training history plot
    os.makedirs('reports', exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reports/training_curves.png')
    print("Training curves saved to 'reports/training_curves.png'")

if __name__ == "__main__":
    train_network()
