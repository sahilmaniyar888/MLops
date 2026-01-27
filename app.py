from flask import Flask, render_template, request, jsonify
import numpy as np
from joblib import load
import os

app = Flask(__name__)

# Global variable to store the model
model = None
species_names = ['Setosa', 'Versicolor', 'Virginica']

def load_model():
    """Load the trained model from disk"""
    global model
    model_path = 'model.joblib'

    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found!")
        print("Please run 'python combined_ml_code.py' first to train the model.")
        return False

    try:
        model = load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model

    # Check if model is loaded
    if model is None:
        if not load_model():
            return jsonify({
                'error': 'Model not found. Please train the model first by running: python combined_ml_code.py'
            }), 503

    try:
        # Get data from request
        data = request.get_json()

        # Extract features in the correct order
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get prediction probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])

        # Convert prediction to species name
        species = species_names[prediction]

        return jsonify({
            'prediction': species,
            'prediction_class': int(prediction),
            'confidence': confidence
        })

    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Try to load the model on startup
    load_model()

    # Run the Flask app
    print("\n" + "="*50)
    print("Starting Iris Flower Classification Web App")
    print("="*50)
    print("\nOpen your browser and go to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
