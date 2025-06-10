---

# MLOps Pipeline for ADAS Sensor Data

This repository contains an MLOps pipeline designed to monitor, train, and deploy machine learning models for **Advanced Driver-Assistance Systems (ADAS)** sensor readings. The pipeline automates data fetching, preprocessing, model training, versioning, and continuous monitoring for data and model drift, with automatic retraining capabilities.

---

## Architecture Overview

The solution comprises several interconnected Python scripts, each responsible for a specific stage of the MLOps lifecycle:

* **`data_simulator.py`**: Simulates ADAS sensor readings and ingests them into a MongoDB database. This acts as our continuous data source.
* **`data_preparation.py`**: Connects to MongoDB, fetches raw data, performs data cleaning (handling duplicates, missing values), feature engineering, and preprocessing (scaling numerical features, one-hot encoding categorical features). It also handles data imbalance using **Random Undersampling**.
* **`model_training.py`**: Defines, trains, and evaluates a machine learning model (e.g., Logistic Regression). It takes preprocessed data and returns a trained model and its performance metrics.
* **`model_versioning.py`**: Manages saving and loading different versions of the trained models and their associated metrics. It ensures that the best-performing model can be tracked and retrieved.
* **`mlops_orchestrator.py`**: The central component that orchestrates the entire pipeline. It periodically checks for new data, monitors for data and model drift using Evidently AI, and triggers retraining and deployment of newer, better models when necessary.

---

## Features

* **Automated Data Ingestion**: Simulates real-time ADAS sensor data flowing into MongoDB.
* **Robust Data Preparation**: Includes steps for handling missing values (median for numerical, mode for categorical) and duplicate records.
* **Feature Engineering**: Automatically handles numerical and categorical feature transformations.
* **Data Balancing**: Utilizes **Random Undersampling** during training to address class imbalance in the target variable (`intervention_needed`).
* **Model Versioning**: Stores trained models and their performance metrics, enabling easy retrieval of the latest or best-performing model.
* **Continuous Monitoring (Evidently AI)**:
    * **Data Drift Detection**: Monitors changes in input data distributions using `DataDriftPreset`.
    * **Model Performance Monitoring**: Tracks key classification metrics (Accuracy, F1-score) using `ClassificationPreset` to identify performance degradation.
* **Automated Retraining**: Triggers model retraining based on predefined intervals or detection of significant data/model drift.
* **Champion/Challenger Deployment**: Automatically deploys a new model if its performance surpasses the currently active production model.

---

## Setup and Installation

### Prerequisites

* Python 3.9+
* MongoDB Atlas Cluster (or a local MongoDB instance) for data storage. Ensure you have the `MONGO_URI` with read/write access.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd mlops-adas-pipeline # Or wherever your cloned directory is
```

### 2. Set up Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` content:**

```
evidently==0.6.0
Faker==37.3.0
fastapi==0.115.12
imbalanced-learn==0.13.0
joblib==1.5.1
pandas==2.3.0
pydantic==2.11.5
pymongo==4.13.0
scikit-learn==1.5.2
uvicorn==0.34.3
```

### 4. Configure MongoDB URI

Open each Python script (`data_simulator.py`, `data_preparation.py`, `mlops_orchestrator.py`) and update the `MONGO_URI`, `DATABASE_NAME`, and `COLLECTION_NAME` variables with your MongoDB Atlas connection string and desired database/collection names.

Example:
```python
MONGO_URI = "MONGO-URL"
DATABASE_NAME = "mlops_pipeline"
COLLECTION_NAME = "sensor_readings"
```

---

## How to Run

Follow these steps to run the MLOps pipeline:

### 1. Start the Data Generator

First, run the data generator to populate your MongoDB collection with synthetic sensor readings. This script will continuously insert data.

```bash
python data_simulator.py
```

You should see messages indicating successful MongoDB connection and data insertion. Let this run for a few minutes to accumulate some initial data.

### 2. Run the MLOps Orchestrator

In a separate terminal, start the MLOps orchestrator. This script will initiate the pipeline, perform initial model training (if no model exists), and then continuously monitor for drift and retrain when necessary.

```bash
python mlops_orchestrator.py
```

Upon its first run, it will:
* Fetch initial data to create a baseline.
* Train an initial model if no model versions are found.
* Save the `preprocessor.pkl` and the first model version.
* Begin its monitoring loop.

### 3. Monitor the Pipeline

Observe the console output of `mlops_orchestrator.py`. You will see messages indicating:
* **Data fetching and preparation progress.**
* **Drift checks:** `Checking for Data and Model Drift...`
* **Model retraining:** `Retraining Model...` if drift is detected or the retraining interval is met.
* **Model deployment decisions.**

A file named `drift_report.html` will be generated in your project directory every time a drift check is performed. Open this file in your web browser to visualize the data and model drift metrics provided by Evidently AI.

---

## Key Files and Components

* **`data_simulator.py`**: Generates synthetic ADAS sensor data, including:
    * `vehicle_speed_kmh`
    * `distance_to_obstacle_m`
    * `lane_departure_warning` (boolean)
    * `front_collision_warning` (boolean)
    * `driver_attention_level`
    * `weather_conditions` (categorical: 'Clear', 'Rainy', 'Foggy', 'Snowy')
    * `road_type` (categorical: 'Highway', 'Urban', 'Rural')
    * `light_conditions` (categorical: 'Daylight', 'Night', 'Dusk/Dawn')
    * `steering_angle_degrees`
    * `accelerator_pedal_position`
    * `brake_pedal_position`
    * `gps_latitude`, `gps_longitude`
    * `intervention_needed` (target variable, boolean, with added noise for realistic accuracy)
* **`data_preparation.py`**:
    * `fetch_data_from_mongodb()`: Connects to MongoDB and retrieves records.
    * `prepare_data(df)`: Handles data cleaning (duplicates, missing values using median/mode imputation), drops `_id`, `record_id`, `timestamp`. Sets up `ColumnTransformer` for numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`). Applies **Random Undersampling** for imbalance. Saves the fitted preprocessor as `preprocessor.pkl`.
* **`model_training.py`**:
    * `train_model(X_train, y_train)`: Trains a `LogisticRegression` model.
    * `evaluate_model(model, X_test, y_test)`: Evaluates the model, returning accuracy, precision, recall, and F1-score.
* **`model_versioning.py`**:
    * `save_model(model, metrics)`: Saves a trained model and its evaluation metrics to the `models/` directory with a timestamped version name.
    * `get_latest_model_version()`: Identifies the latest model version based on filenames.
    * `load_model(version_name)`: Loads a specific model version and its metrics.
* **`mlops_orchestrator.py`**:
    * Main loop orchestrating checks and actions.
    * Configures `RETRAIN_INTERVAL_SECONDS`, `DRIFT_CHECK_INTERVAL_SECONDS`, `MIN_DATA_FOR_TRAINING`.
    * Uses `DataDriftPreset` and `ClassificationPreset` from Evidently AI for monitoring.
    * `baseline_data.pkl`: Stores a snapshot of initial data for drift reference.
    * `last_drift_check_time.txt`: Records the timestamp of the last drift check to control monitoring frequency.

---

## Customization

* **Model Choice**: Easily swap `LogisticRegression` in `model_training.py` with other scikit-learn models (e.g., `RandomForestClassifier`, `GradientBoostingClassifier`).
* **Drift Thresholds**: Adjust `DRIFT_THRESHOLD_P_VALUE` in `mlops_orchestrator.py` to make drift detection more or less sensitive.
* **Intervals**: Modify `RETRAIN_INTERVAL_SECONDS` and `DRIFT_CHECK_INTERVAL_SECONDS` to control how frequently the pipeline performs its tasks.
* **Data Generation**: Modify `data_simulator.py` to change the distribution or introduce different types of patterns/noise in the simulated data.
* **Data Balancing Strategy**: Change `RandomUnderSampler` to `SMOTE` or other `imblearn` techniques in `data_preparation.py` if oversampling is preferred.

---

## Troubleshooting

* **MongoDB Connection Errors**: Double-check your `MONGO_URI`, network connectivity, and MongoDB Atlas IP access list.
* **Evidently Errors**: Ensure your `evidently` library version is compatible with the code. If you encounter `unexpected keyword argument` errors, it's likely a version mismatch. Consider running `pip install --upgrade evidently` and adjusting the `mlops_orchestrator.py` code to match the latest Evidently API (as discussed in previous interactions, this might involve changes around `column_mapping` or `ClassificationPreset` arguments like `task_type`).
* **`No data found in MongoDB`**: Make sure `data_simulator.py` is running and successfully inserting documents before starting the orchestrator.
* **100% Accuracy**: This was intentionally addressed by introducing `FLIP_PROBABILITY` in `data_simulator.py`. If you still see 100%, increase the `FLIP_PROBABILITY`.
* **Empty DataFrames in Evidently**: Ensure enough new data has been collected between drift checks (`DRIFT_CHECK_INTERVAL_SECONDS`).

---

Feel free to explore, modify, and extend this pipeline to suit your specific MLOps needs!