import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class AirQualityDataset(Dataset):
    def __init__(self, csv_file, target_column="CO(GT)", device=torch.device("cpu")):
        data = pd.read_csv(csv_file, parse_dates=['Datetime'])
        data = data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
        self.device = device
        self.target_column = target_column

        # Drop rows with NaN target values
        data = data.dropna(subset=[target_column])

        # Sort data by datetime
        data = data.sort_values('Datetime')

        # Extract features and target
        self.features = [col for col in data.columns if col not in ['Datetime', target_column]]
        self.data = data[self.features].fillna(0).values
        self.target = data[target_column].values

        # Time deltas in hours
        self.timestamps = (data['Datetime'] - data['Datetime'].min()).dt.total_seconds() / 3600

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time = torch.tensor([self.timestamps[idx]], dtype=torch.float32).to(self.device)
        vals = torch.tensor(self.data[idx], dtype=torch.float32).to(self.device)
        mask = torch.tensor(~pd.isnull(self.data[idx]), dtype=torch.float32).to(self.device)
        label = torch.tensor(self.target[idx], dtype=torch.float32).to(self.device)
        return (idx, time, vals, mask, label)

def variable_time_collate_fn3(batch, args, device=torch.device("cpu"), data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
    - record_id is a unique identifier for each time series
    - tt is a 1-dimensional tensor containing time values of observations.
    - vals is a (T, D) tensor containing observed values for D variables.
    - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    - labels is the target variable value for each time series.
    
    Returns:
    - combined_tt: The union of all time observations in the batch.
    - combined_vals: (B, T, D) tensor containing the observed values.
    - combined_mask: (B, T, D) tensor containing 1 where values were observed and 0 otherwise.
    - combined_labels: (B, 1) tensor containing labels for each time series.
    """
    D = batch[0][2].shape[0]  # Number of features
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_labels = torch.zeros(len(batch), 1).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        combined_labels[b] = labels

    # Normalization if required
    if data_min is not None and data_max is not None:
        combined_vals = (combined_vals - data_min) / (data_max - data_min)
        combined_vals[combined_mask == 0] = 0

    return {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels
    }

def get_data_min_max(records, device):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for _, tt, vals, mask, _ in records:
        n_features = vals.size(-1) if vals.dim() > 1 else 1

        batch_min = []
        batch_max = []
        for i in range(n_features):
            if n_features == 1:
                non_missing_vals = vals[mask == 1]
            else:
                non_missing_vals = vals[:, i][mask[:, i] == 1]
                
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if data_min is None and data_max is None:
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual data file
    dataset = AirQualityDataset(csv_file='AirQualityUCI_Irregular.csv', device=torch.device("cpu"))
    
    # Define batch size and other DataLoader parameters
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=variable_time_collate_fn3)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)