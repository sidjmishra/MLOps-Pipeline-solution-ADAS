import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings

def train_model(X_train, y_train, config):
    model_type = config['ml_task_config']['model_type']
    hyperparameters = config['ml_task_config']['model_hyperparameters']
    task_type = config['ml_task_config']['task_type']

    model = None
    print(f"Training {model_type} model for {task_type} task...")

    if task_type == 'classification':
        if model_type == 'logistic_regression':
            model = LogisticRegression(**hyperparameters)
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier(**hyperparameters)
        elif model_type == 'gradient_boosting_classifier':
            model = GradientBoostingClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unsupported classification model type: {model_type}")
    elif task_type == 'regression':
        if model_type == 'linear_regression':
            model = LinearRegression(**hyperparameters)
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(**hyperparameters)
        elif model_type == 'gradient_boosting_regressor':
            model = GradientBoostingRegressor(**hyperparameters)
        else:
            raise ValueError(f"Unsupported regression model type: {model_type}")
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Must be 'classification' or 'regression'.")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        model.fit(X_train, y_train)
    
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test, config):
    task_type = config['ml_task_config']['task_type']
    metrics_to_calculate = config['ml_task_config']['evaluation_metrics']
    
    y_pred = model.predict(X_test)
    metrics = {}

    if task_type == 'classification':
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]

        print("Calculating classification metrics...")
        if "accuracy" in metrics_to_calculate:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics_to_calculate:
            metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        if "recall" in metrics_to_calculate:
            metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        if "f1_score" in metrics_to_calculate:
            metrics["f1_score"] = f1_score(y_test, y_pred, zero_division=0)
        
        metric_output = ", ".join([f"{k.replace('_', '-').title()}={v:.4f}" for k, v in metrics.items()])
        print(f"Model Metrics: {metric_output}")

        return metrics, y_pred, y_proba

    elif task_type == 'regression':
        print("Calculating regression metrics...")
        if "mae" in metrics_to_calculate:
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
        if "mse" in metrics_to_calculate:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
        if "rmse" in metrics_to_calculate:
            metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        if "r2_score" in metrics_to_calculate:
            metrics["r2_score"] = r2_score(y_test, y_pred)

        metric_output = ", ".join([f"{k.replace('_', '-').title()}={v:.4f}" for k, v in metrics.items()])
        print(f"Model Metrics: {metric_output}")
        
        return metrics, y_pred, None 
    
    else:
        raise ValueError(f"Unsupported task type for evaluation: {task_type}")