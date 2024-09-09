import torch
import numpy as np
import os
import argparse
import pickle
from collections import Counter

def pad_array(arr, max_len):
    # Pad the array to the max_len with zeros
    if len(arr.shape) == 2:  # For T array
        padded_arr = np.zeros((arr.shape[0], max_len))
    else:  # For X and M arrays
        padded_arr = np.zeros((arr.shape[0], max_len, arr.shape[2]))
    padded_arr[:, :arr.shape[1]] = arr
    return padded_arr

def pad_sequences(data, mask, time, max_len, num_features):
    padded_data = np.zeros((max_len, num_features))
    padded_mask = np.zeros((max_len, num_features))
    padded_time = np.zeros((max_len, 1))

    padded_data[:data.shape[0], :] = data
    padded_mask[:mask.shape[0], :] = mask
    padded_time[:time.shape[0], 0] = time

    return np.concatenate((padded_data, padded_mask, padded_time), axis=1)

def prepare_data_for_saving(data, max_len):
    X_padded = []
    labels = []

    for record_id, tt, vals, mask, label in data:
        if label is not None:
            print(f"Before padding: vals shape = {vals.shape}, time shape = {tt.shape}, mask shape = {mask.shape}, label shape = {label.shape}")
            combined_data = np.concatenate((vals, mask, tt[:, None]), axis=1)  # Concatenate vals, mask, and time as is
            X_padded.append(combined_data)
            print(label[4])
            labels.append(label[4])

    # Convert lists to numpy arrays
    X_padded = np.array(X_padded)
    labels = np.array(labels)

    return X_padded, labels

def save_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    torch.save(torch.tensor(X_train, dtype=torch.float32), os.path.join(output_dir, 'X_train.pt'))
    torch.save(torch.tensor(y_train, dtype=torch.long), os.path.join(output_dir, 'y_train.pt'))

    torch.save(torch.tensor(X_val, dtype=torch.float32), os.path.join(output_dir, 'X_val.pt'))
    torch.save(torch.tensor(y_val, dtype=torch.long), os.path.join(output_dir, 'y_val.pt'))

    torch.save(torch.tensor(X_test, dtype=torch.float32), os.path.join(output_dir, 'X_test.pt'))
    torch.save(torch.tensor(y_test, dtype=torch.long), os.path.join(output_dir, 'y_test.pt'))

    print("Data saved successfully.")

if __name__ == '__main__':
    # Load the .pkl files instead of .npy files
    with open("mTAN/src/train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("mTAN/src/test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)
    with open("mTAN/src/val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)
    record_id, tt, vals, mask, labels = train_data[0]
    # Find the maximum length in the time dimension across all datasets
    max_len_train = max([tt.shape[0] for _, tt, _, _, _ in train_data])
    max_len_val = max([tt.shape[0] for _, tt, _, _, _ in val_data])
    max_len_test = max([tt.shape[0] for _, tt, _, _, _ in test_data])
    # Process the train, validation, and test data
    X_train, y_train = prepare_data_for_saving(train_data, max_len_train)
    X_test, y_test = prepare_data_for_saving(test_data, max_len_test)
    X_val, y_val = prepare_data_for_saving(val_data, max_len_val)

    # Print sizes of the datasets
    print(f"Train Data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Validation Data: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")
    print(f"Test Data: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    # Save the data
    save_data(X_train, y_train, X_val, y_val, X_test, y_test, 'PrimeNet/data/finetune/')