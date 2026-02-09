import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    """
    Builds the CardioRiskNet architecture:
    Input Layer
     ↓
    Dense Layer + ReLU
     ↓
    Dense Layer + ReLU
     ↓
    Dropout
     ↓
    Dense Layer
     ↓
    Sigmoid Output (Risk Probability)
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        
        # Dense Layer 1
        layers.Dense(64, activation='relu'),
        
        # Dense Layer 2
        layers.Dense(32, activation='relu'),
        
        # Dropout for regularization
        layers.Dropout(0.2),
        
        # Dense Layer 3
        layers.Dense(16, activation='relu'),
        
        # Output Layer (Binary Classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model summary
    test_model = create_model(13)
    test_model.summary()
