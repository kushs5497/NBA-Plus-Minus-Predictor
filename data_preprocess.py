import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Read the combined data
    df = pd.read_csv('combined_box_scores_2010_2024.csv')
    
    # Drop unnecessary columns
    columns_to_drop = [
        'gameId', 'personId', 'personName', 'comment', 'jerseyNum',
        'teamId', 'teamCity', 'teamName', 'teamSlug'
    ]
    df = df.drop(columns=columns_to_drop)
    
    # Convert game_date to datetime and extract useful features
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['day_of_week'] = df['game_date'].dt.dayofweek
    df['month'] = df['game_date'].dt.month
    df = df.drop(columns=['game_date'])
    
    # Convert position to categorical
    df['position'] = df['position'].fillna('UNK')
    
    # Convert game_type to binary (0 for regular season, 1 for playoff)
    df['game_type'] = (df['game_type'] == 'playoff').astype(int)
    
    # Extract home/away from matchup
    df['is_home'] = df['matchup'].fillna('').str.contains('vs.').astype(int)
    df = df.drop(columns=['matchup'])
    
    # One-hot encode categorical variables
    categorical_columns = ['position', 'teamTricode']
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Handle missing values in numerical columns
    numerical_columns = [
        'minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
        'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',
        'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
        'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal',
        'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points'
    ]
    
    # Convert minutes to numerical (e.g., "12:34" -> 12.57)
    df['minutes'] = df['minutes'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if isinstance(x, str) else 0)
    
    # Fill missing values with 0 for counting stats and median for percentages
    for col in numerical_columns:
        if 'Percentage' in col:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    # Separate features and target
    X = df.drop(columns=['plusMinusPoints'])
    y = df['plusMinusPoints']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Save preprocessed data
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("Preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print("\nFeature columns:")
    print(X_train.columns.tolist())

if __name__ == "__main__":
    preprocess_data()
