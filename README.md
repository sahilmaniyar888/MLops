# ML Iris Classification Project

A modular machine learning project for classifying Iris flower species using Random Forest algorithm, complete with a Flask web interface for predictions.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, model building, training, and evaluation
- **Configuration-Driven**: YAML-based configuration for easy parameter tuning without code changes
- **Web Interface**: Beautiful, responsive Flask-based UI for interactive predictions
- **REST API**: RESTful endpoints for programmatic access
- **Model Serialization**: Trained model saved using joblib for production deployment
- **Docker Support**: Containerized deployment with health checks
- **Multiple Entry Points**: Both modular and standalone training scripts

## Project Structure

```
ML_Modular_Code/
├── app.py                      # Flask web application
├── main.py                     # Main training pipeline
├── combined_ml_code.py         # Standalone training script
├── model.joblib                # Trained model (generated after training)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── configs/
│   └── config.yaml             # Training configuration
├── src/
│   ├── data_loader.py          # Data loading module
│   ├── model.py                # Model builder
│   ├── train.py                # Training module
│   └── evaluate.py             # Evaluation module
└── templates/
    └── index.html              # Web UI template
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ML_Modular_Code
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

#### Option 1: Modular Training (Recommended)
```bash
python main.py
```

This uses the configuration from [configs/config.yaml](configs/config.yaml) and executes the modular training pipeline.

#### Option 2: Standalone Training
```bash
python combined_ml_code.py
```

This runs a self-contained training script with inline configuration.

### Running the Web Application

1. Ensure the model is trained (model.joblib exists)
2. Start the Flask server:
```bash
python app.py
```

3. Access the web interface:
```
http://localhost:5000
```

## API Endpoints

### Web Interface
- **GET /** - Serves the interactive web UI

### Prediction API
- **POST /predict**
  - Accepts JSON with iris measurements
  - Returns predicted species and confidence score

  **Request Body:**
  ```json
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
  ```

  **Response:**
  ```json
  {
    "prediction": "setosa",
    "confidence": 0.95
  }
  ```

### Health Check
- **GET /health** - Returns service health status

## Configuration

Modify [configs/config.yaml](configs/config.yaml) to adjust training parameters:

```yaml
data:
  dataset: "iris"

model:
  type: "RandomForestClassifier"
  params:
    n_estimators: 100    # Number of trees
    max_depth: 4         # Maximum depth of trees

train:
  test_size: 0.2         # Train/test split ratio
  random_state: 42       # Random seed

output:
  model_path: "model.joblib"
```

## Docker Deployment

### Build the Docker Image
```bash
docker build -t iris-classifier .
```

### Run the Container
```bash
docker run -p 5000:5000 iris-classifier
```

Access the application at http://localhost:5000

## Model Specifications

- **Algorithm**: Random Forest Classifier
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes**:
  - Setosa
  - Versicolor
  - Virginica
- **Train/Test Split**: 80/20
- **Default Parameters**: 100 estimators, max depth 4

## Dependencies

- numpy - Numerical computing
- pandas - Data manipulation
- scikit-learn - ML algorithms and metrics
- joblib - Model serialization
- pyyaml - Configuration parsing
- flask - Web framework

See [requirements.txt](requirements.txt) for specific versions.

## Project Workflow

1. **Data Loading**: Load Iris dataset from scikit-learn
2. **Model Building**: Initialize Random Forest classifier with configured parameters
3. **Training**: Fit model on training data (80% of dataset)
4. **Evaluation**: Calculate accuracy on test data (20% of dataset)
5. **Serialization**: Save trained model to model.joblib
6. **Deployment**: Load model in Flask app for predictions

## Development

### Running Tests
```bash
# Train and verify model performance
python main.py
```

### Modifying the Model
Edit [configs/config.yaml](configs/config.yaml) to change model parameters, then retrain.

### Adding New Features
- Data processing: Modify [src/data_loader.py](src/data_loader.py)
- Model architecture: Update [src/model.py](src/model.py)
- Training logic: Edit [src/train.py](src/train.py)
- Evaluation metrics: Enhance [src/evaluate.py](src/evaluate.py)

