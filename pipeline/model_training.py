import os
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(X_train, y_train):
    """Trains a machine learning model."""
    if X_train is None or y_train is None:
        print("Training data is not available.")
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    # model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Alternative

    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data."""
    if model is None or X_test is None or y_test is None:
        print("Model or test data not available for evaluation.")
        return {}

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba, zero_division=0)
    }
    print("\nModel Evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")
    return metrics

# if __name__ == "__main__":
#     try:
#         from data_preparation import fetch_data_from_mongodb, prepare_data
#         raw_data_df = fetch_data_from_mongodb()
#         X_train, X_test, y_train, y_test, preprocessor = prepare_data(raw_data_df)
#     except ImportError:
#         print("Please run data_preparation.py first to get X_train, X_test, y_train, y_test.")
#         from sklearn.datasets import make_classification
#         X_train, y_train = make_classification(n_samples=1000, n_features=10, random_state=42)
#         X_test, y_test = make_classification(n_samples=200, n_features=10, random_state=42)
#         preprocessor = None


#     if X_train is not None:
#         model = train_model(X_train, y_train)
#         if model:
#             metrics = evaluate_model(model, X_test, y_test)