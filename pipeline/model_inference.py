import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from model_classes.models import DataInput

app = FastAPI(title="ML Model Prediction Service")

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Ensure it's in the root directory.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error decoding config.json: {e}")
    exit()

ML_TASK_CONFIG = config['ml_task_config']
DATA_CONFIG = config['data_config']

MODELS_DIR = "models"
PREPROCESSOR_PATH = "artefacts/preprocessor.pkl"
ACTIVE_MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "active_model_version.txt")

model = None
preprocessor = None
current_active_model_version = None

def get_active_model_version_from_file():
    """Reads the active model version from a file."""
    if os.path.exists(ACTIVE_MODEL_VERSION_FILE):
        with open(ACTIVE_MODEL_VERSION_FILE, 'r') as f:
            return f.read().strip()
    return None

def load_active_model_and_preprocessor():
    """Loads the actively deployed model and preprocessor."""
    global model, preprocessor, current_active_model_version
    
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("Preprocessor loaded successfully.")
        except Exception as e:
            print(f"Error loading preprocessor from {PREPROCESSOR_PATH}: {e}. Ensure it's valid.")
            preprocessor = None
    else:
        print(f"Error: Preprocessor not found at {PREPROCESSOR_PATH}. Ensure data preparation ran.")
        preprocessor = None

    active_version = get_active_model_version_from_file()
    if active_version:
        model_path = os.path.join(MODELS_DIR, f"{active_version}.pkl")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                current_active_model_version = active_version
                print(f"Active model '{active_version}' loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading active model from {model_path}: {e}. Cannot deploy.")
                model = None
        else:
            print(f"Error: Active model file {model_path} not found. Cannot deploy.")
            model = None
    else:
        print("No active model version set. Please activate a model first.")
        model = None
    return False

@app.on_event("startup")
async def startup_event():
    """Loads model and preprocessor when the API starts up."""
    print("Starting FastAPI app...")
    load_active_model_and_preprocessor()

@app.get("/")
async def root():
    """Root endpoint for basic check."""
    return {"message": "ML Model Prediction Service is running. Use /predict to get predictions."}

@app.post("/predict")
async def predict(data: DataInput):
    """
    Receives input data and returns a prediction based on the configured ML task.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded. Please check server logs.")

    input_dict = data.data

    all_expected_features = (
        DATA_CONFIG['numerical_features'] +
        DATA_CONFIG['categorical_features'] +
        DATA_CONFIG['boolean_features']
    )
    
    input_df = pd.DataFrame([input_dict])

    input_df = input_df.reindex(columns=all_expected_features, fill_value=None)

    for col in DATA_CONFIG['boolean_features']:
        if col in input_df.columns and input_df[col].dtype == 'bool':
            input_df[col] = input_df[col].astype(int)
        elif col in input_df.columns and input_df[col].dtype == 'object':
             input_df[col] = input_df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)


    try:
        processed_input = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}. Check input format and ensure all required features are provided according to config.json.")

    task_type = ML_TASK_CONFIG['task_type']
    prediction_output = {}

    if task_type == 'classification':
        prediction = model.predict(processed_input)[0]
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(processed_input)[0, 1]
            prediction_output["prediction_probability"] = round(float(prediction_proba), 4)
        
        prediction_output["prediction"] = bool(prediction)
    
    elif task_type == 'regression':
        prediction = model.predict(processed_input)[0]
        prediction_output["prediction"] = round(float(prediction), 4)
    
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported task type configured: {task_type}")

    prediction_output["active_model_version"] = current_active_model_version
    return prediction_output

@app.post("/activate_model/{version_name}")
async def activate_model_endpoint(version_name: str):
    """
    Activates a specific model version for deployment.
    This simulates switching the production model.
    """
    model_path = os.path.join(MODELS_DIR, f"{version_name}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model version '{version_name}' not found.")

    try:
        with open(ACTIVE_MODEL_VERSION_FILE, 'w') as f:
            f.write(version_name)

        success = load_active_model_and_preprocessor()
        if success:
            return {"message": f"Model version '{version_name}' activated successfully and loaded."}
        else:
            raise HTTPException(status_code=500, detail="Failed to load activated model. Check server logs.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {e}")