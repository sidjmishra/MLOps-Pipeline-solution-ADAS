import os
import json
import time
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta

from data_preparation import fetch_data_from_mongodb, prepare_data
from model_training import train_model, evaluate_model
from model_versioning import save_model, get_latest_model_version, load_model

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset, RegressionPreset # Import RegressionPreset

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please create one based on the example.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error decoding config.json: {e}")
    exit()

DATA_CONFIG = config['data_config']
ML_TASK_CONFIG = config['ml_task_config']
PIPELINE_CONFIG = config['pipeline_config']
MONGODB_CONFIG = config['mongodb_config']

MONGO_URI = MONGODB_CONFIG['mongo_uri']
DATABASE_NAME = MONGODB_CONFIG['database_name']
COLLECTION_NAME = MONGODB_CONFIG['collection_name']

RETRAIN_INTERVAL_SECONDS = PIPELINE_CONFIG['retrain_interval_seconds']
DRIFT_CHECK_INTERVAL_SECONDS = PIPELINE_CONFIG['drift_check_interval_seconds']
MIN_DATA_FOR_TRAINING = PIPELINE_CONFIG['min_data_for_training']
PERFORMANCE_DEGRADATION_THRESHOLD = PIPELINE_CONFIG['performance_degradation_threshold']

BASELINE_DATA_PATH = "artefacts/baseline_data.pkl"
LAST_CHECK_TIME_FILE = "artefacts/last_drift_check_time.txt"

preprocessor_loaded = None 

def get_db_collection():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    return client, db[COLLECTION_NAME]

def get_last_check_timestamp():
    if os.path.exists(LAST_CHECK_TIME_FILE):
        with open(LAST_CHECK_TIME_FILE, 'r') as f:
            timestamp_str = f.read().strip()
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str)
    return datetime.min

def set_last_check_timestamp(timestamp: datetime):
    with open(LAST_CHECK_TIME_FILE, 'w') as f:
        f.write(timestamp.isoformat())

def get_production_model_metrics(version_name):
    metrics_path = os.path.join("models", f"{version_name}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

def activate_model_in_api(version_name: str):
    active_model_file = os.path.join("models", "active_model_version.txt")
    try:
        with open(active_model_file, 'w') as f:
            f.write(version_name)
        print(f"API activation: Set active model to '{version_name}'.")
    except Exception as e:
        print(f"Error model activation: {e}")

def run_mlops_pipeline():
    global preprocessor_loaded

    last_retrain_time = datetime.now() - timedelta(days=365)
    last_drift_check_time = get_last_check_timestamp()

    client, collection = get_db_collection()
    initial_data_cursor = collection.find({}).limit(1000) 
    initial_data = list(initial_data_cursor)
    client.close()
    
    if not initial_data:
        print("Waiting for initial data to be collected in MongoDB for baseline.")
        print("Please run the data generation script first (if using it).")
        return

    baseline_df = pd.DataFrame(initial_data).drop(columns=DATA_CONFIG['columns_to_drop'], errors='ignore')
    if DATA_CONFIG['target_column_name'] in baseline_df.columns:
        if ML_TASK_CONFIG['task_type'] == 'classification':
            baseline_df[DATA_CONFIG['target_column_name']] = baseline_df[DATA_CONFIG['target_column_name']].astype(int)
    
    joblib.dump(baseline_df, BASELINE_DATA_PATH)
    print(f"Baseline data saved to {BASELINE_DATA_PATH}")


    if not get_latest_model_version():
        print("No model found. Performing initial training and deployment.")
        df = fetch_data_from_mongodb(config)
        if not df.empty and df.shape[0] >= MIN_DATA_FOR_TRAINING:
            X_train, X_test, y_train, y_test, initial_preprocessor = prepare_data(df, config)
            if X_train is not None:
                model = train_model(X_train, y_train, config)
                if model:
                    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, config)
                    version_name, _ = save_model(model, metrics)
                    activate_model_in_api(version_name)
                    preprocessor_loaded = initial_preprocessor
                    print("Initial model trained and deployed.")
                    last_retrain_time = datetime.now()
        else:
            print(f"Not enough initial data to train a model. Need at least {MIN_DATA_FOR_TRAINING} records. Waiting for more data.")
    
    if preprocessor_loaded is None and os.path.exists('preprocessor.pkl'):
        preprocessor_loaded = joblib.load('preprocessor.pkl')
        print("Preprocessor loaded from preprocessor.pkl for ongoing monitoring.")
    elif preprocessor_loaded is None:
        print("Warning: preprocessor.pkl not found and not generated by initial training. Model monitoring might be limited.")


    while True:
        current_time = datetime.now()
        
        drift_detected = False

        if (current_time - last_drift_check_time).total_seconds() >= DRIFT_CHECK_INTERVAL_SECONDS:
            print(f"\n--- Checking for Data and Model Drift at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            client, collection = get_db_collection()
            recent_data_cursor = collection.find({
                "timestamp": {"$gte": last_drift_check_time}
            }).limit(2000)
            current_raw_data_df = pd.DataFrame(list(recent_data_cursor))
            client.close()

            target_col = DATA_CONFIG['target_column_name']

            if current_raw_data_df.empty or target_col not in current_raw_data_df.columns:
                print(f"Not enough recent data for drift detection or target column '{target_col}' missing.")
            else:
                current_raw_data_df = current_raw_data_df.drop(columns=DATA_CONFIG['columns_to_drop'], errors='ignore')
                if ML_TASK_CONFIG['task_type'] == 'classification':
                    current_raw_data_df[target_col] = current_raw_data_df[target_col].astype(int)

                baseline_df = joblib.load(BASELINE_DATA_PATH)
                
                active_version = get_latest_model_version()
                active_model, _ = (load_model(active_version) if active_version else (None, None))
                
                if active_model and preprocessor_loaded:
                    try:
                        features_for_prediction = current_raw_data_df.drop(target_col, axis=1, errors='ignore')
                        
                        processed_features = preprocessor_loaded.transform(features_for_prediction)
                        
                        if hasattr(processed_features, 'toarray'):
                            processed_features = processed_features.toarray()

                        metrics, y_pred, y_proba = evaluate_model(active_model, processed_features, current_raw_data_df[target_col], config)
                        
                        current_data_for_report = current_raw_data_df.copy()
                        current_data_for_report['target'] = current_data_for_report[target_col]
                        current_data_for_report['prediction'] = y_pred
                        if y_proba is not None:
                            current_data_for_report['prediction_proba'] = y_proba

                        baseline_df_for_report = baseline_df.copy()
                        baseline_df_for_report['target'] = baseline_df_for_report[target_col]
                        baseline_features_for_prediction = baseline_df_for_report.drop(target_col, axis=1, errors='ignore')
                        processed_baseline_features = preprocessor_loaded.transform(baseline_features_for_prediction)
                        
                        if hasattr(processed_baseline_features, 'toarray'):
                            processed_baseline_features = processed_baseline_features.toarray()

                        _, y_pred_baseline, y_proba_baseline = evaluate_model(active_model, processed_baseline_features, baseline_df_for_report[target_col], config)
                        
                        baseline_df_for_report['prediction'] = y_pred_baseline
                        if y_proba_baseline is not None:
                            baseline_df_for_report['prediction_proba'] = y_proba_baseline

                        evidently_metrics = [DataDriftPreset()]
                        if ML_TASK_CONFIG['task_type'] == 'classification':
                            evidently_metrics.append(ClassificationPreset())
                        elif ML_TASK_CONFIG['task_type'] == 'regression':
                            evidently_metrics.append(RegressionPreset())
                        
                        drift_report = Report(metrics=evidently_metrics)

                        column_mapping = {
                            'target': target_col,
                            'prediction': 'prediction',
                            'numerical_features': [],
                            'categorical_features': []
                        }
                        if ML_TASK_CONFIG['task_type'] == 'classification' and 'prediction_proba' in current_data_for_report.columns:
                            column_mapping['prediction_probas'] = 'prediction_proba'

                        all_features = DATA_CONFIG['numerical_features'] + DATA_CONFIG['categorical_features'] + DATA_CONFIG['boolean_features']
                        for col in all_features:
                            if col in current_data_for_report.columns and col != target_col:
                                if col in DATA_CONFIG['numerical_features'] or col in DATA_CONFIG['boolean_features']:
                                    column_mapping['numerical_features'].append(col)
                                elif col in DATA_CONFIG['categorical_features']:
                                    column_mapping['categorical_features'].append(col)

                        drift_report.run(
                            reference_data=baseline_df_for_report,
                            current_data=current_data_for_report,
                            # column_mapping=column_mapping
                        )
                        
                        report_html_path = "artefacts/drift_report.html"
                        drift_report.save_html(report_html_path)
                        print(f"Drift report saved to {report_html_path}")

                        data_drift_result = drift_report.as_dict()['metrics'][0]['result']
                        if data_drift_result.get('dataset_drift'):
                            print("  Dataset drift detected!")
                            drift_detected = True

                        main_metric = ML_TASK_CONFIG['main_metric_for_deployment']
                        current_main_metric_value = metrics.get(main_metric, 0)
                        
                        print(f"  Current {main_metric.replace('_', '-').title()}: {current_main_metric_value:.4f}")

                        if active_version:
                            current_active_model_metrics = get_production_model_metrics(active_version)
                            baseline_main_metric_value = current_active_model_metrics.get(main_metric, 0)

                            if ML_TASK_CONFIG['task_type'] == 'classification':
                                if current_main_metric_value < baseline_main_metric_value * PERFORMANCE_DEGRADATION_THRESHOLD:
                                    print(f"  Model {main_metric.replace('_', '-').title()} dropped from {baseline_main_metric_value:.4f} to {current_main_metric_value:.4f}. Performance degradation detected!")
                                    drift_detected = True
                            elif ML_TASK_CONFIG['task_type'] == 'regression':
                                if main_metric in ['mae', 'mse', 'rmse']:
                                    if current_main_metric_value > baseline_main_metric_value / PERFORMANCE_DEGRADATION_THRESHOLD and baseline_main_metric_value != 0:
                                        print(f"  Model {main_metric.replace('_', '-').title()} increased from {baseline_main_metric_value:.4f} to {current_main_metric_value:.4f}. Performance degradation detected!")
                                        drift_detected = True
                                elif main_metric == 'r2_score':
                                    if current_main_metric_value < baseline_main_metric_value * PERFORMANCE_DEGRADATION_THRESHOLD:
                                        print(f"  Model {main_metric.replace('_', '-').title()} dropped from {baseline_main_metric_value:.4f} to {current_main_metric_value:.4f}. Performance degradation detected!")
                                        drift_detected = True
                                else:
                                    print(f"  Warning: Unknown regression metric '{main_metric}' for comparison.")
                                    if current_main_metric_value < baseline_main_metric_value * PERFORMANCE_DEGRADATION_THRESHOLD:
                                        print(f"  Model {main_metric.replace('_', '-').title()} dropped from {baseline_main_metric_value:.4f} to {current_main_metric_value:.4f}. Performance degradation detected!")
                                        drift_detected = True
                        
                        if drift_detected:
                            print("  Drift detected, considering retraining.")
                        else:
                            print("  No significant drift or performance degradation detected.")
                            
                    except Exception as e:
                        print(f"Error during Evidently report generation or analysis: {e}")
                        drift_detected = False

                else:
                    print("Active model or preprocessor not loaded. Cannot run full drift detection.")

            set_last_check_timestamp(current_time)

        if (current_time - last_retrain_time).total_seconds() >= RETRAIN_INTERVAL_SECONDS or drift_detected:
            print(f"\n--- Retraining Model at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            df = fetch_data_from_mongodb(config)
            
            if not df.empty and df.shape[0] >= MIN_DATA_FOR_TRAINING:
                X_train, X_test, y_train, y_test, new_preprocessor = prepare_data(df, config)
                if X_train is not None:
                    model = train_model(X_train, y_train, config)
                    if model:
                        new_metrics, _, _ = evaluate_model(model, X_test, y_test, config)
                        
                        active_version = get_latest_model_version()
                        if active_version:
                            current_metrics = get_production_model_metrics(active_version)
                            
                            main_metric = ML_TASK_CONFIG['main_metric_for_deployment']
                            new_model_score = new_metrics.get(main_metric, 0)
                            current_model_score = current_metrics.get(main_metric, 0)

                            if ML_TASK_CONFIG['task_type'] == 'classification' or main_metric == 'r2_score':
                                if new_model_score > current_model_score:
                                    print(f"New model ({main_metric.title()}: {new_model_score:.4f}) is better than current ({main_metric.title()}: {current_model_score:.4f}).")
                                    version_name, _ = save_model(model, new_metrics)
                                    activate_model_in_api(version_name)
                                    preprocessor_loaded = new_preprocessor
                                    print(f"New model version '{version_name}' deployed!")
                                else:
                                    print(f"New model ({main_metric.title()}: {new_model_score:.4f}) is NOT better than current ({main_metric.title()}: {current_model_score:.4f}). Keeping current model.")
                            elif ML_TASK_CONFIG['task_type'] == 'regression':
                                if new_model_score < current_model_score:
                                    print(f"New model ({main_metric.title()}: {new_model_score:.4f}) is better than current ({main_metric.title()}: {current_model_score:.4f}).")
                                    version_name, _ = save_model(model, new_metrics)
                                    activate_model_in_api(version_name)
                                    preprocessor_loaded = new_preprocessor
                                    print(f"New model version '{version_name}' deployed!")
                                else:
                                    print(f"New model ({main_metric.title()}: {new_model_score:.4f}) is NOT better than current ({main_metric.title()}: {current_model_score:.4f}). Keeping current model.")
                                    
                        else:
                            print("No active model found. Deploying the newly trained model.")
                            version_name, _ = save_model(model, new_metrics)
                            activate_model_in_api(version_name)
                            preprocessor_loaded = new_preprocessor
                            print(f"Initial model version '{version_name}' deployed!")
                
                last_retrain_time = current_time

            else:
                print(f"Not enough data for retraining. Need at least {MIN_DATA_FOR_TRAINING} records.")

        time.sleep(60)

if __name__ == "__main__":
    print("Starting MLOps Orchestrator...")
    try:
        run_mlops_pipeline()
    except KeyboardInterrupt:
        print("\nMLOps Orchestrator stopped by user.")
    except Exception as e:
        print(f"An error occurred in the orchestrator: {e}")