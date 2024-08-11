import numpy as np
import pandas as pd
import torch
import os

# Example function to preprocess data
def prepare_data(df):
    T = df['time'].values
    X = df.drop(columns=['time']).values
    M = ~np.isnan(X)
    X[np.isnan(X)] = 0  # Replace NaNs with zeros
    return T, X, M

# Generate your custom irregular dataset
np.random.seed(42)
time_points = np.arange(0, 100, 1)
features = np.random.rand(100, 5)  # 5 features

# Convert to DataFrame
df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(1, 6)])
df['time'] = time_points

# Introduce irregularity
for col in df.columns[:-1]:
    df.loc[df.sample(frac=0.3).index, col] = np.nan  # Remove 30% of the data randomly

# Introduce asynchronicity
for col in df.columns[:-1]:
    df.loc[df.sample(frac=0.2).index, col] = np.nan  # Additional 20% data removal to simulate asynchronicity

# Prepare the data
T, X, M = prepare_data(df)

# Reshape T to match the shape of X and M
T_expanded = np.tile(T, (X.shape[1], 1)).T

# Combine T, X, M into a single tensor
combined_data = np.stack((T_expanded, X, M), axis=2)

# Create directories if they don't exist

print(T_expanded.shape)
print(X.shape)
print(M.shape)
os.makedirs('data/pretrain', exist_ok=True)
os.makedirs('data/finetune', exist_ok=True)

# Save the data in the required format for pre-training
torch.save(torch.tensor(combined_data[:60], dtype=torch.float32), 'data/pretrain/X_train.pt')
torch.save(torch.tensor(combined_data[60:80], dtype=torch.float32), 'data/pretrain/X_val.pt')

# Generate random labels for fine-tuning (assuming binary classification)
labels = np.random.randint(0, 2, size=(100,))  # Binary labels for classification, example

# Split into train, validation, and test sets
labels_train, labels_val, labels_test = labels[:60], labels[60:80], labels[80:]

# Save the fine-tuning data
torch.save(torch.tensor(combined_data[:60], dtype=torch.float32), 'data/finetune/X_train.pt')
torch.save(torch.tensor(combined_data[60:80], dtype=torch.float32), 'data/finetune/X_val.pt')
torch.save(torch.tensor(combined_data[80:], dtype=torch.float32), 'data/finetune/X_test.pt')

torch.save(torch.tensor(labels_train, dtype=torch.long), 'data/finetune/y_train.pt')
torch.save(torch.tensor(labels_val, dtype=torch.long), 'data/finetune/y_val.pt')
torch.save(torch.tensor(labels_test, dtype=torch.long), 'data/finetune/y_test.pt')

print("Data generation and saving completed.")
