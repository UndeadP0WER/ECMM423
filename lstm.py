import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMParams:
    #array representation of the params, 
    #   hidden_size, num_layers, droput, learn_rate, epochs

    #hidden_size, num_layers and epochs are all ints
    # dropout, learn_rate are all floats between 0 & 1
    
    #so when exploring hyperparameters, all should be between 0 and 1, and the ints should be scaled by a large number, INT_SCALE, and rounded
    INT_SCALE = 39
    
    #init from individual values
    #does not need to be scaled from [0,1] 
    def __init__(self, hidden_size=5, num_layers=2, dropout=0, learn_rate=0.001, epochs=20):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learn_rate
        self.epochs = epochs
        return
    
    #init from an array
    #previously mentioned variables should be scaled and rounded
    def from_arr(arr):
        p = LSTMParams(
            hidden_size = round(arr[0]*LSTMParams.INT_SCALE)+1,
            num_layers = round(arr[1]*LSTMParams.INT_SCALE)+1,
            dropout = arr[2],
            learn_rate = arr[3],
            epochs = round(arr[4]*LSTMParams.INT_SCALE)+1
            )
        return p
    
#modelled after https://colab.research.google.com/github/aaubs/ds-master/blob/main/notebooks/M3_LSTM_Tutorial_v3.ipynb#scrollTo=1jyjApvHVkcO

class LSTMModel(nn.Module):
    def __init__(self, output_size=1, params=LSTMParams()):
        super(LSTMModel, self).__init__()
        self.params = params
        
        self.hidden_size = params.hidden_size
        self.num_layers = params.num_layers

        #loads the data, applies some parameters, prepares for training
        self.prep_data()

        self.lstm = nn.LSTM(self.n_features, params.hidden_size, params.num_layers,
                            batch_first=True, dropout=params.dropout)
        self.fc = nn.Linear(params.hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out
    
        
    def create_sequences_multivariate(data, n_timesteps, target_column_index):
        X = []
        y = []
        for i in range(len(data) - n_timesteps):
            seq_x = data[i:i + n_timesteps]
            seq_y = data[i + n_timesteps, target_column_index]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def denormalize(scaled_data, scaler, index):
        data = np.zeros((len(scaled_data), scaler.n_features_in_))
        data[:, index] = scaled_data[:, 0]
        data = scaler.inverse_transform(data)
        return data[:, index]

    def prep_data(self):
        ##Preparing data
        #loads data
        data = pd.read_csv('time_series_data.csv', parse_dates=True)
        series = data['Value'].values
        
        #scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        
        #split into training, testing and validation sets
        
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        train_valid_size = int(len(train_data) * 0.8)
        train_data_final = train_data[:train_valid_size]
        valid_data = train_data[train_valid_size:]
        
       
        n_timesteps = 1
        self.n_features = data_scaled.shape[1]
        self.target_column_index = 0  # 'Close' is the target column
        
        # Create sequences for training, validation, and testing
        X_train, y_train = LSTMModel.create_sequences_multivariate(train_data_final, n_timesteps, self.target_column_index)
        X_valid, y_valid = LSTMModel.create_sequences_multivariate(valid_data, n_timesteps, self.target_column_index)
        X_test , y_test  = LSTMModel.create_sequences_multivariate(test_data, n_timesteps, self.target_column_index)
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)
        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.Xtest = X_test
        self.ytest = y_test
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        valid_dataset = TensorDataset(X_valid, y_valid)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        
    def train_model(self):    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.params.lr)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        train_losses = []
        valid_losses = []
        
                
        for epoch in range(self.params.epochs):
            self.train()
            train_loss = 0
            for inputs, targets in self.train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
        
                optimizer.zero_grad()
                # 2. Forward Pass
                outputs = self(inputs)
                # 3. FeedForward Evaluation
                loss = criterion(outputs.squeeze(), targets)
                # 4. Backward Pass / Gradient Calculation
                loss.backward()
                # 5. Back Propagation / Update Weights
                optimizer.step()
        
                train_loss += loss.item() * inputs.size(0)
        
            train_loss /= len(self.train_loader.dataset)
            train_losses.append(train_loss)
        
            # Validation
            self.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in self.valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    valid_loss += loss.item() * inputs.size(0)
            valid_loss /= len(self.valid_loader.dataset)
            valid_losses.append(valid_loss)
        
            if (epoch + 1) % 2 == 0:
                print(f'Epoch {epoch + 1}/{self.params.epochs}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')
        
        
        self.eval()
        with torch.no_grad(): # this way prevents PyTorch from storing unnecessary information for backpropagation, making inference faster and more memory-efficient.
            test_preds = self(self.Xtest.to(device)).cpu().numpy()
            test_actuals = self.ytest.numpy()

        
        test_preds_denorm = LSTMModel.denormalize(test_preds, self.scaler, self.target_column_index)
        test_actuals_denorm = LSTMModel.denormalize(test_actuals.reshape(-1, 1), self.scaler, self.target_column_index)
        return test_preds_denorm, test_actuals_denorm
    
def generate_dataset(year_count):
    #assume seeding the rng is done before calling

    #generate a date range
    date_range = pd.date_range(start=str(2025-year_count)+'-01-01', end=str(2025)+'-01-01', freq='MS')

    #create synthetic time series data
    trend = np.linspace(0, 10, len(date_range)) ##general trend line across time, long term memory
    seasonal_effect = 10 * np.sin(np.linspace(0, year_count * 2 * np.pi, len(date_range))) # seasonal stuff, short term memory
    noise = np.random.normal(loc=0, scale=1, size=len(date_range)) #random noise
    data = trend + seasonal_effect + noise

    #create a dataframe
    df = pd.DataFrame(data, index=date_range, columns=['Value'])

    #save the data to a file
    df.to_csv('time_series_data.csv')
    
    #returns the dataframe
    return df


#params = LSTMParams(hidden_size=20, num_layers=10, dropout=0.2, learn_rate=0.001, epochs=20)
#model = LSTMModel(output_size=1, params=params)
#tp, ta = model.train_model()
#print(tp)
#print(ta)