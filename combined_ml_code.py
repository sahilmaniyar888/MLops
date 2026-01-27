"""
Combined ML Code for Iris Dataset Classification
This file combines all modular components into a single script for easy execution.
"""

import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump


def load_data():
    """Load Iris dataset from scikit-learn"""
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y


def build_model(config):
    """Build Random Forest classifier with specified parameters"""
    params = config["model"]["params"]
    model = RandomForestClassifier(**params)
    return model


def train(X, y, model, config):
    """Train the model on the dataset and save it to disk"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["train"]["test_size"], 
        random_state=config["train"]["random_state"]
    )
    
    model.fit(X_train, y_train)
    dump(model, config["output"]["model_path"])
    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    """Evaluate model performance on test dataset"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    # Load configuration from YAML file
    config = {
        "data": {
            "dataset": "iris"
        },
        "model": {
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 100,
                "max_depth": 4
            }
        },
        "train": {
            "test_size": 0.2,
            "random_state": 42
        },
        "output": {
            "model_path": "model.joblib"
        }
    }

    print("Starting Iris dataset classification...")
    
    # Load data
    X, y = load_data()
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Build model
    clf = build_model(config)
    print(f"Created {config['model']['type']} with parameters: {config['model']['params']}")
    
    # Train model
    clf, X_test, y_test = train(X, y, clf, config)
    print(f"Training completed. Model saved to: {config['output']['model_path']}")
    
    # Evaluate model
    accuracy = evaluate(clf, X_test, y_test)
    
    print("\nTraining and evaluation completed successfully!")
