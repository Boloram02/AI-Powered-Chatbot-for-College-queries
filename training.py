"""
Training script for the AI College Chatbot
This script trains an intent classification model using TF-IDF + Keras
"""

import json
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os

# Flush output for immediate display
sys.stdout.flush()

def load_intents(file_path='intents.json'):
    """Load intents data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(intents):
    """Prepare training data from intents"""
    patterns = []
    tags = []
    
    # Extract patterns and their corresponding tags
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern.lower())
            tags.append(intent['tag'])
    
    return patterns, tags

def create_model(input_dim, output_dim):
    """Create a neural network model for classification"""
    model = keras.Sequential([
        # Input layer (TF-IDF vectors)
        layers.Dense(128, input_shape=(input_dim,), activation='relu'),
        layers.Dropout(0.5),
        
        # Hidden layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(output_dim, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main training function"""
    print("Loading intents data...", flush=True)
    intents_data = load_intents()
    
    print("Preprocessing data...", flush=True)
    patterns, tags = preprocess_data(intents_data)
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF vectorizer...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Fit and transform patterns
    X = vectorizer.fit_transform(patterns).toarray()
    
    # Create tag mapping
    unique_tags = sorted(set(tags))
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    # Convert tags to categorical format
    y = np.array([tag_to_idx[tag] for tag in tags])
    y_categorical = keras.utils.to_categorical(y, num_classes=len(unique_tags))
    
    # Split data into training and testing sets
    print("Splitting data...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    # Create and compile model
    print("Creating model...", flush=True)
    model = create_model(X.shape[1], len(unique_tags))
    
    # Display model summary
    model.summary()
    
    # Train the model
    print("Training model...", flush=True)
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=4,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...", flush=True)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}", flush=True)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/chat_model.h5')
    print("Model saved as models/chat_model.h5", flush=True)
    
    # Save vectorizer
    with open('models/vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    print("Vectorizer saved as models/vectorizer.pkl", flush=True)
    
    # Save tag mappings
    with open('models/tag_mappings.pkl', 'wb') as file:
        pickle.dump({
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag,
            'unique_tags': unique_tags
        }, file)
    print("Tag mappings saved as models/tag_mappings.pkl", flush=True)
    
    # Save intents data for responses
    with open('models/intents_data.pkl', 'wb') as file:
        pickle.dump(intents_data, file)
    print("Intents data saved as models/intents_data.pkl", flush=True)
    
    print("\nTraining completed successfully!", flush=True)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}", flush=True)
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}", flush=True)

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

