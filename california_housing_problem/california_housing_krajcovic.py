import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import csv

class CaliforniaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(NeuralNet, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, 1))  
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, optimizer, train_loader, test_loader, num_epochs, config, csv_writer):
    criterion = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            test_losses = []
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_losses.append(loss.item())
            avg_test_loss = np.mean(test_losses)

        # Write the results to CSV
        csv_writer.writerow({
            'Optimizer': config['optimizer'],
            'Hidden_Sizes': str(config['hidden_sizes']),
            'Epoch': epoch,
            'Train_Loss': avg_train_loss,
            'Test_Loss': avg_test_loss
        })

        print(f"Optimizer: {config['optimizer']}, Hidden Sizes: {config['hidden_sizes']}, Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")


def train_and_evaluate(hidden_sizes, optimizer_type, lr, train_loader, test_loader, num_epochs, csv_writer):
    input_size = next(iter(train_loader))[0].shape[1]  
    model = NeuralNet(input_size, hidden_sizes)
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'SGD_momentum':
        # Reduced the learning rate here to avoid NaNs
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        raise ValueError("Unsupported optimizer type")

    config = {
        'optimizer': optimizer_type,
        'hidden_sizes': hidden_sizes
    }

    print(f"\nTraining with {optimizer_type}, hidden sizes {hidden_sizes}, for {num_epochs} epochs")
    print(f"Optimizer, Hidden_Sizes, Epoch, Train_Loss, Test_Loss")
    train_model(model, optimizer, train_loader, test_loader, num_epochs, config, csv_writer)


if __name__ == "__main__":
    data = pd.read_csv('housing.csv')

    # Handle housing.csv
    if 'total_bedrooms' in data.columns:
        data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())
    
    if 'ocean_proximity' in data.columns:
        data['ocean_proximity'] = data['ocean_proximity'].astype(str)
        data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
    
    bool_columns = data.select_dtypes(include=['bool']).columns.tolist()
    if bool_columns:
        data[bool_columns] = data[bool_columns].astype(int)
    
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42)
    
    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    feature_indices = [X.columns.get_loc(col) for col in numerical_features]
    scaler_X = StandardScaler()
    X_train[:, feature_indices] = scaler_X.fit_transform(X_train[:, feature_indices])
    X_test[:, feature_indices] = scaler_X.transform(X_test[:, feature_indices])

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    batch_size = 32
    train_dataset = CaliforniaDataset(X_train, y_train)
    test_dataset = CaliforniaDataset(X_test, y_test)

    num_workers = 0 #can add cores manually

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    learning_rates = {'SGD': 0.1, 'SGD_momentum': 0.1, 'Adam': 0.0005} 
    
    
    #config
    hidden_layer_configs = [
        [64, 32],
        [256, 128, 128]
    ]

    optimizers = ['SGD', 'SGD_momentum', 'Adam']

    # Open results.csv for writing
    with open('results.csv', mode='w', newline='') as file:
        fieldnames = ['Optimizer', 'Hidden_Sizes', 'Epoch', 'Train_Loss', 'Test_Loss']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for optimizer_type in optimizers:
            for hidden_sizes in hidden_layer_configs:
                lr = learning_rates[optimizer_type]
                train_and_evaluate(hidden_sizes, optimizer_type, lr, train_loader, test_loader, num_epochs=20, csv_writer=writer)






    results = pd.read_csv('results.csv')
    results['Config'] = results['Optimizer'] + '_' + results['Hidden_Sizes']

    large_config_str = str([256, 128, 128])
    small_config_str = str([64, 32])

    # Plotting for large configuration [256, 128, 128]
    plt.figure(figsize=(12, 6))
    large_subset = results[results['Hidden_Sizes'] == large_config_str]
    for opt in large_subset['Optimizer'].unique():
        opt_subset = large_subset[large_subset['Optimizer'] == opt]
        plt.plot(opt_subset['Epoch'], opt_subset['Train_Loss'], label=f'{opt} Train')
        plt.plot(opt_subset['Epoch'], opt_subset['Test_Loss'], label=f'{opt} Test', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Testing Losses per Epoch for {large_config_str} Configuration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_results_large.png')
    plt.show()

    # Plotting for small configuration [64, 32]
    plt.figure(figsize=(12, 6))
    small_subset = results[results['Hidden_Sizes'] == small_config_str]
    for opt in small_subset['Optimizer'].unique():
        opt_subset = small_subset[small_subset['Optimizer'] == opt]
        plt.plot(opt_subset['Epoch'], opt_subset['Train_Loss'], label=f'{opt} Train')
        plt.plot(opt_subset['Epoch'], opt_subset['Test_Loss'], label=f'{opt} Test', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Testing Losses per Epoch for {small_config_str} Configuration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_results_small.png')
    plt.show()

    print("\nTraining complete. Results have been saved to 'results.csv', 'training_results_large.png' and 'training_results_small.png'.")