import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from model_classes.models import ADASData

app = FastAPI(title="ADAS Model Prediction Service")

MODELS_DIR = "models"
PREPROCESSOR_PATH = "preprocessor.pkl"
ACTIVE_MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "active_model_version.txt")

model = None
preprocessor = None

def get_active_model_version():
    """Reads the active model version from a file."""
    if os.path.exists(ACTIVE_MODEL_VERSION_FILE):
        with open(ACTIVE_MODEL_VERSION_FILE, 'r') as f:
            return f.read().strip()
    return None

def load_active_model_and_preprocessor():
    """Loads the actively deployed model and preprocessor."""
    global model, preprocessor
    
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
    else:
        print(f"Error: Preprocessor not found at {PREPROCESSOR_PATH}. Ensure data preparation ran.")
        preprocessor = None

    active_version = get_active_model_version()
    if active_version:
        model_path = os.path.join(MODELS_DIR, f"{active_version}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Active model '{active_version}' loaded successfully.")
            return True
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
    return {"message": "ADAS Prediction Service is running. Use /predict to get predictions."}

@app.post("/predict")
async def predict_intervention(data: ADASData):
    """
    Receives ADAS sensor data and returns a prediction for 'intervention_needed'.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded. Please check logs.")

    input_df = pd.DataFrame([data.dict()])

    expected_columns = [
        'vehicle_speed_kmh', 'distance_to_obstacle_m', 'lane_departure_warning',
        'front_collision_warning', 'driver_attention_level', 'weather_conditions',
        'road_type', 'light_conditions', 'steering_angle_degrees',
        'accelerator_pedal_position', 'brake_pedal_position'
    ]
    
    input_df = input_df[expected_columns]

    try:
        processed_input = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}. Check input format.")

    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0, 1]

    return {
        "intervention_needed_prediction": bool(prediction),
        "prediction_probability": round(prediction_proba, 4),
        "active_model_version": get_active_model_version()
    }

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
