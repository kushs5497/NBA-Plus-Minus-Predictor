import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PlusMinusPredictorV1(nn.Module):
    """Version 1.0: Initial model with basic architecture."""
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(PlusMinusPredictorV1, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ConfigV1:
    """Version 1.0 configuration."""
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_v1():
    """Train Version 1.0 model using pre-existing CSV files."""
    # Load data from CSV files
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    
    # Convert 'season_year' to numerical format (e.g., '2011-12' -> 2011)
    if 'season_year' in X_train.columns:
        X_train['season_year'] = X_train['season_year'].str[:4].astype(int)
        X_test['season_year'] = X_test['season_year'].str[:4].astype(int)
    
    # Handle non-numeric columns by one-hot encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure both dataframes have the same columns
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    missing_cols_train = test_cols - train_cols
    for c in missing_cols_train:
        X_train[c] = 0
    
    missing_cols_test = train_cols - test_cols
    for c in missing_cols_test:
        X_test[c] = 0
    
    # Ensure the order of columns is the same
    X_test = X_test[X_train.columns]
    
    # Normalize features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Normalize target variable
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=ConfigV1.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=ConfigV1.BATCH_SIZE)
    
    # Initialize model
    model = PlusMinusPredictorV1(
        input_size=X_train.shape[1],
        hidden_layers=ConfigV1.HIDDEN_LAYERS,
        dropout_rate=ConfigV1.DROPOUT_RATE
    ).to(ConfigV1.DEVICE)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ConfigV1.LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(ConfigV1.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(ConfigV1.DEVICE), y_batch.to(ConfigV1.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(ConfigV1.DEVICE), y_batch.to(ConfigV1.DEVICE)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{ConfigV1.NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_v1.pth')
        else:
            patience_counter += 1
            if patience_counter >= ConfigV1.EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model_v1.pth'))
    
    # Evaluate on test set
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(ConfigV1.DEVICE)
            outputs = model(X_batch)
            y_pred.extend(target_scaler.inverse_transform(outputs.cpu().numpy()))
            y_true.extend(target_scaler.inverse_transform(y_batch.numpy()))
    
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'\nFinal Test Results for Version 1.0:')
    print(f'MSE: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    
    # Plotting the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss_v1.png')  # Save the plot as an image
    plt.show()  # Display the plot
    
    # Plotting predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    plt.savefig('predicted_vs_actual_v1.png')  # Save the plot as an image
    plt.show()  # Display the plot

if __name__ == "__main__":
    train_model_v1()