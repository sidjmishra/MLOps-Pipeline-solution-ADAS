import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def fetch_data_from_mongodb(config):
    mongo_uri = config['mongodb_config']['mongo_uri']
    database_name = config['mongodb_config']['database_name']
    collection_name = config['mongodb_config']['collection_name']
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]

        data = list(collection.find({}))
        df = pd.DataFrame(data)
        print(f"Fetched {len(df)} records from MongoDB.")
        return df
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return pd.DataFrame()
    finally:
        if client:
            client.close()

def prepare_data(df, config):
    data_config = config['data_config']
    target_column = data_config['target_column_name']
    drop_columns = data_config['columns_to_drop']
    numerical_features = data_config['numerical_features']
    categorical_features = data_config['categorical_features']
    boolean_features = data_config['boolean_features']
    
    ml_task_type = config['ml_task_config']['task_type']

    if df.empty:
        print("DataFrame is empty, cannot prepare data.")
        return None, None, None, None, None

    df = df.drop(columns=drop_columns, errors='ignore')

    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")

    for col in boolean_features:
        if col in df.columns and df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            if col not in numerical_features:
                numerical_features.append(col)
        elif col in df.columns and df[col].dtype == 'object':
             df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
             if col not in numerical_features:
                 numerical_features.append(col)


    for col in numerical_features:
        if col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Imputed missing values in numerical column '{col}' with median.")
    
    for col in categorical_features:
        if col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"Imputed missing values in categorical column '{col}' with mode.")

    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    if y.dtype == 'bool':
        y = y.astype(int)

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, [f for f in numerical_features if f in X.columns]),
            ('cat', categorical_transformer, [f for f in categorical_features if f in X.columns])
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y if ml_task_type == 'classification' else None
    )

    if ml_task_type == 'classification':
        print("Checking for imbalanced data...")
        class_counts = y_train.value_counts()
        print(f"Original training set class distribution: {class_counts.to_dict()}")
        
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
        print(f"Resampled training set class distribution: {y_train_resampled.value_counts().to_dict()}")
        
        joblib.dump(preprocessor, 'artefacts/preprocessor.pkl')
        print("Preprocessor saved as 'artefacts/preprocessor.pkl'.")
        
        return X_train_resampled, X_test, y_train_resampled, y_test, preprocessor
    else:
        joblib.dump(preprocessor, 'artefacts/preprocessor.pkl')
        print("Preprocessor saved as 'artefacts/preprocessor.pkl'.")
        return X_train, X_test, y_train, y_test, preprocessor