import numpy as np
import pandas as pd
import torch
import os

# Function to preprocess the data
def prepare_data(df):
    T = df.index.values.astype(float)
    X = df.values
    M = ~np.isnan(X)
    X[np.isnan(X)] = 0  # Replace NaNs with zeros
    return T, X, M

# Load your AirQualityUCI dataset
file_path = 'AirQualityUCI_Irregular.csv'
df = pd.read_csv(file_path)

# Convert the Datetime column to pandas datetime and set it as index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Drop the unnecessary columns
df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

# Prepare the data
T, X, M = prepare_data(df)

# Reshape T to match the shape of X and M
T_expanded = np.tile(T, (X.shape[1], 1)).T

# Combine T, X, M into a single tensor
combined_data = np.stack((T_expanded, X, M), axis=2)

# Create directories if they don't exist
os.makedirs('PrimeNet/data/pretrain', exist_ok=True)
os.makedirs('PrimeNet/data/finetune', exist_ok=True)

# Save the data in the required format for pre-training
torch.save(torch.tensor(combined_data[:int(len(df) * 0.6)], dtype=torch.float32), 'PrimeNet/data/pretrain/X_train.pt')
torch.save(torch.tensor(combined_data[int(len(df) * 0.6):int(len(df) * 0.8)], dtype=torch.float32), 'PrimeNet/data/pretrain/X_val.pt')

# Generate random labels for fine-tuning (assuming binary classification)
labels = np.random.randint(0, 2, size=(len(df),))  # Binary labels for classification, example

# Split into train, validation, and test sets
labels_train, labels_val, labels_test = labels[:int(len(df) * 0.6)], labels[int(len(df) * 0.6):int(len(df) * 0.8)], labels[int(len(df) * 0.8):]

# Save the fine-tuning data
torch.save(torch.tensor(combined_data[:int(len(df) * 0.6)], dtype=torch.float32), 'PrimeNet/data/finetune/X_train.pt')
torch.save(torch.tensor(combined_data[int(len(df) * 0.6):int(len(df) * 0.8)], dtype=torch.float32), 'PrimeNet/data/finetune/X_val.pt')
torch.save(torch.tensor(combined_data[int(len(df) * 0.8):], dtype=torch.float32), 'PrimeNet/data/finetune/X_test.pt')

torch.save(torch.tensor(labels_train, dtype=torch.long), 'PrimeNet/data/finetune/y_train.pt')
torch.save(torch.tensor(labels_val, dtype=torch.long), 'PrimeNet/data/finetune/y_val.pt')
torch.save(torch.tensor(labels_test, dtype=torch.long), 'PrimeNet/data/finetune/y_test.pt')

print("Data generation and saving completed.")
