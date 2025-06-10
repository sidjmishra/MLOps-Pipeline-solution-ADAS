import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE


MONGO_URI = "MONGO-URL"
DATABASE_NAME = "mlops_pipeline"
COLLECTION_NAME = "sensor_readings"

def fetch_data_from_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    data = list(collection.find({}))
    client.close()
    print(f"Fetched {len(data)} records from MongoDB.")
    if not data:
        print("Warning: No data found in MongoDB collection.")
        return pd.DataFrame()
    return pd.DataFrame(data)

def prepare_data(df):
    if df.empty:
        print("DataFrame is empty, cannot prepare data.")
        return None, None, None, None, None

    print("\n--- Starting Data Cleaning ---")

    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    df = df.drop(columns=['record_id', 'gps_latitude', 'gps_longitude'])
    
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_duplicates = df.shape[0]
    print(f"Dropped {initial_rows - rows_after_duplicates} duplicate rows.")

    potential_numerical_features = df.select_dtypes(include=['number']).columns
    potential_categorical_features = df.select_dtypes(include=['object', 'bool']).columns

    for col in potential_numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled nulls in numerical column '{col}' with median: {median_val}")

    for col in potential_categorical_features:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled nulls in categorical/boolean column '{col}' with mode: {mode_val}")

    rows_after_nulls = df.shape[0]
    print(f"Handled null values. Current rows: {rows_after_nulls}")
    
    if df.isnull().sum().sum() > 0:
        print(f"Warning: {df.isnull().sum().sum()} null values remaining after imputation. Dropping rows with nulls.")
        df.dropna(inplace=True)
        print(f"Dropped {rows_after_nulls - df.shape[0]} rows due to remaining nulls.")


    print("\n--- Starting Feature Engineering & Preprocessing ---")

    df = df.drop(columns=['record_id', 'timestamp'], errors='ignore')

    if 'intervention_needed' not in df.columns:
        print("Error: 'intervention_needed' column not found after cleaning. Cannot prepare data.")
        return None, None, None, None, None

    X = df.drop('intervention_needed', axis=1)
    y = df['intervention_needed'].astype(int)

    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    if not numerical_features:
        print("Warning: No numerical features found.")
        numerical_transformer = 'drop'
    else:
        numerical_transformer = StandardScaler()

    if not categorical_features:
        print("Warning: No categorical features found.")
        categorical_transformer = 'drop'
    else:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Apply preprocessing BEFORE splitting ---
    # The preprocessor needs to be fitted on the entire dataset (or training set)
    # for consistent transformation of train and test data.
    if not X.empty:
        X_processed = preprocessor.fit_transform(X)
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
    else:
        print("Warning: No features left after cleaning for processing.")
        return None, None, None, None, None
    
    # Save the preprocessor AFTER fitting but BEFORE splitting
    # This preprocessor will be used for both training and later for inference
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Preprocessor saved to preprocessor.pkl")

    # --- Split data into training and testing sets FIRST ---
    # This is CRUCIAL to prevent data leakage from SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data split: Training samples={X_train.shape[0]}, Test samples={X_test.shape[0]}")
    print(f"Original training class distribution: \n{pd.Series(y_train).value_counts()}")

    # --- Data Balancing (SMOTE) ONLY on the Training Data ---
    print("\n--- Starting Data Balancing (SMOTE) on Training Data ---")
    if pd.Series(y_train).value_counts().min() < pd.Series(y_train).value_counts().max():
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Resampled training class distribution: \n{pd.Series(y_train_resampled).value_counts()}")
        X_train = X_train_resampled
        y_train = y_train_resampled
    else:
        print("Training dataset is already balanced or has only one class. Skipping SMOTE.")
    
    return X_train, X_test, y_train, y_test, preprocessor

# if __name__ == "__main__":
#     print("Running data_preparation.py in standalone mode for testing.")
#     raw_data_df = fetch_data_from_mongodb()
#     if not raw_data_df.empty:
#         X_train, X_test, y_train, y_test, preprocessor = prepare_data(raw_data_df)
#         if X_train is not None:
#             print("\nData Preparation Complete.")
#             print(f"Shape of X_train: {X_train.shape}")
#             print(f"Shape of y_train: {y_train.shape}")
#             print(f"Shape of X_test: {X_test.shape}")
#             print(f"Shape of y_test: {y_test.shape}")
#         else:
#             print("Data preparation did not yield valid training/test sets.")
#     else:
#         print("No data fetched from MongoDB to prepare.")