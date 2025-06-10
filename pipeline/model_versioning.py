import os
import json
import joblib
from datetime import datetime

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model, metrics, version_name=None):
    """
    Saves the trained model and its evaluation metrics.
    Generates a version name based on timestamp if not provided.
    """
    if version_name is None:
        version_name = datetime.now().strftime("model_%Y%m%d_%H%M%S")
    
    model_path = os.path.join(MODELS_DIR, f"{version_name}.pkl")
    metrics_path = os.path.join(MODELS_DIR, f"{version_name}_metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nModel '{version_name}' saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    return version_name, model_path

def load_model(version_name):
    """Loads a specific version of the model."""
    model_path = os.path.join(MODELS_DIR, f"{version_name}.pkl")
    metrics_path = os.path.join(MODELS_DIR, f"{version_name}_metrics.json")
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return None, None

    model = joblib.load(model_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Model '{version_name}' loaded successfully.")
    return model, metrics

def get_latest_model_version():
    """Identifies the latest model based on timestamp in the filename."""
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    if not model_files:
        return None

    model_files.sort(reverse=True)
    latest_model_file = model_files[0]
    version_name = latest_model_file.replace('.pkl', '')
    return version_name

# if __name__ == "__main__":
#     from sklearn.linear_model import LogisticRegression
#     dummy_model = LogisticRegression()
#     dummy_metrics = {'accuracy': 0.85, 'f1_score': 0.78}

#     version, path = save_model(dummy_model, dummy_metrics)

#     latest_version = get_latest_model_version()
#     if latest_version:
#         loaded_model, loaded_metrics = load_model(latest_version)
#         print(f"Loaded model version: {latest_version}")
#         print(f"Loaded metrics: {loaded_metrics}")