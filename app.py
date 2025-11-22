"""
AI-Powered College Query Chatbot - Flask Backend
This Flask application handles user queries and returns intelligent responses
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Global variables for model components
model = None
vectorizer = None
tag_mappings = None
intents_data = None

def load_model_components():
    """Load all required model components"""
    global model, vectorizer, tag_mappings, intents_data
    
    try:
        # Load the trained model
        model = load_model('models/chat_model.h5')
        print("Model loaded successfully")
        
        # Load vectorizer
        with open('models/vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        print("Vectorizer loaded successfully")
        
        # Load tag mappings
        with open('models/tag_mappings.pkl', 'rb') as file:
            tag_mappings = pickle.load(file)
        print("Tag mappings loaded successfully")
        
        # Load intents data
        with open('models/intents_data.pkl', 'rb') as file:
            intents_data = pickle.load(file)
        print("Intents data loaded successfully")
        
        return True
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run training.py first to generate model files.")
        return False
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

def predict_intent(user_input):
    """Predict the intent of user input"""
    try:
        # Preprocess user input
        user_input_lower = user_input.lower()
        
        # Convert to TF-IDF vector
        input_vector = vectorizer.transform([user_input_lower]).toarray()
        
        # Predict intent
        prediction = model.predict(input_vector, verbose=0)
        
        # Get the predicted class index
        predicted_idx = np.argmax(prediction[0])
        
        # Get the predicted tag
        predicted_tag = tag_mappings['idx_to_tag'][predicted_idx]
        
        # Get confidence score
        confidence = float(prediction[0][predicted_idx])
        
        return predicted_tag, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

def get_response(tag, user_input):
    """Get response based on predicted intent"""
    try:
        # Find the intent with matching tag
        for intent in intents_data['intents']:
            if intent['tag'] == tag:
                # Return a random response from available responses
                import random
                return random.choice(intent['responses'])
        
        return "I'm sorry, I didn't understand that. Could you please rephrase your question?"
    except Exception as e:
        print(f"Error getting response: {e}")
        return "Sorry, I encountered an error. Please try again."

def init_database():
    """Initialize SQLite database for chat logs"""
    try:
        conn = sqlite3.connect('chat_logs.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                bot_response TEXT,
                intent_tag TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

def save_chat_log(user_message, bot_response, intent_tag, confidence):
    """Save chat log to database"""
    try:
        conn = sqlite3.connect('chat_logs.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_logs (user_message, bot_response, intent_tag, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_message, bot_response, intent_tag, confidence))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat log: {e}")

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        # Get user input
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({
                'response': 'Please enter a valid message.',
                'intent': None,
                'confidence': 0.0
            })
        
        # Predict intent and get response
        intent_tag, confidence = predict_intent(user_input)
        response = get_response(intent_tag, user_input)
        
        # Save chat log
        save_chat_log(user_input, response, intent_tag, confidence)
        
        return jsonify({
            'response': response,
            'intent': intent_tag,
            'confidence': round(confidence, 3)
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'intent': None,
            'confidence': 0.0
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Chatbot is running'
    })

if __name__ == '__main__':
    print("=" * 50)
    print("AI College Chatbot - Starting Application")
    print("=" * 50)
    
    # Initialize database
    init_database()
    
    # Load model components
    if load_model_components():
        print("\nApplication started successfully!")
        print("Open http://localhost:5001 in your browser")
        print("=" * 50)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("\nFailed to load model. Please run training.py first.")
        print("=" * 50)

