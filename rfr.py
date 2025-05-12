import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_data(x_train_path='X_train.csv', x_test_path='X_test.csv',
              y_train_path='y_train.csv', y_test_path='y_test.csv'):
    """
    Load preprocessed training and test sets from CSV files.
    Converts any season-like string column 'YYYY-YY' to the integer year YYYY.
    Returns:
        X_train, X_test (pd.DataFrame)
        y_train, y_test (np.ndarray)
    """
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # Flatten if y is one-column DataFrame
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Convert any object columns matching 'YYYY-YY' to integer year YYYY
    season_pattern = r'^\d{4}-\d{2}$'
    for col in X_train.select_dtypes(include=['object']).columns:
        if X_train[col].str.match(season_pattern).all():
            X_train[col] = X_train[col].str[:4].astype(int)
            X_test[col] = X_test[col].str[:4].astype(int)

    return X_train, X_test, y_train, y_test

def build_pipeline(rf_params):
    """
    Construct a preprocessing + RandomForest pipeline.  
    - One-hot encodes remaining categorical features.
    - Leaves numerical features as-is.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), [] )
        ],
        remainder='passthrough'
    )

    rf = RandomForestRegressor(**rf_params, n_jobs=-1)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', rf)
    ])
    return pipeline


def train_and_evaluate(n_estimators=100, max_depth=None, random_state=42):
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Identify remaining categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build pipeline
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': random_state
    }
    pipeline = build_pipeline(rf_params)

    # Set categorical transformer
    pipeline.named_steps['preprocessor'].transformers[0] = (
        'cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Random Forest Regressor Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # Feature importances
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_

    # Get feature names
    preproc = pipeline.named_steps['preprocessor']
    feature_names = []
    if cat_cols:
        ohe = preproc.named_transformers_['cat']
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    feature_names.extend(num_cols)

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 20 Feature Importances:")
    print(fi.head(20))

    # Save artifacts
    joblib.dump(pipeline, 'random_forest_model.pkl')
    fi.to_csv('feature_importances.csv')
    print("\nPipeline saved to 'random_forest_model.pkl' and importances to 'feature_importances.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest Regressor on NBA plus-minus data"
    )
    parser.add_argument(
        '--n_estimators', type=int, default=100,
        help='Number of trees in the forest (default: 100)'
    )
    parser.add_argument(
        '--max_depth', type=int, default=None,
        help='Maximum depth of each tree (default: None)'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()
    train_and_evaluate(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
