import os
import utils
import tarfile
import pickle
import torch
import random
from torchvision.datasets.utils import download_url
import numpy as np
from sklearn import model_selection
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Process PhysioNet dataset")
parser.add_argument('--n', type=int, default=8000, help='Number of samples to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
parser.add_argument('--classif', action='store_true', help='Whether to perform classification')
args = parser.parse_args()


class PhysioNet(object):

    urls = [
    'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
    'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
    'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
    labels_dict = {k: i for i, k in enumerate(labels)}



    def __init__(self, root, train=True, download=False,
        quantization = 0.1, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.train = train
        self.device = device
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(data_file)
        if self.device == 'cpu':
            self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]



    def download(self):
        if self._check_exists():
            return

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    labels,
                    os.path.join(self.processed_folder, filename.split('.')[0] + '.pt')
                )

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params)).to(self.device)]
                    mask = [torch.zeros(len(self.params)).to(self.device)]
                    nobs = [torch.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)).to(self.device))
                            mask.append(torch.zeros(len(self.params)).to(self.device))
                            nobs.append(torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            #vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
            patients,
            os.path.join(self.processed_folder, 
                filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )
            
        print('Done!')



    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
            os.path.join(self.processed_folder, 
                filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True



    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')



    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')



    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)



    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)



    @property
    def label_file(self):
        return 'Outcomes-a.pt'



    def __getitem__(self, index):
        return self.data[index]



    def __len__(self):
        return len(self.data)



    def get_label(self, record_id):
        return self.labels[record_id]



    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str


    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

        #for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:,param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o') 
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


q = 0.016
device = 'cpu'

train_dataset_obj = PhysioNet('physionet', train=True,
                                quantization=q,
                                download=True, n_samples=min(10000, args.n),
                                device=device)
# Use custom collate_fn to combine samples with arbitrary time observations.
# Returns the dataset along with mask and time steps
test_dataset_obj = PhysioNet('physionet', train=False,
                                quantization=q,
                                download=True, n_samples=min(10000, args.n),
                                device=device)

# Combine and shuffle samples from physionet Train and physionet Test
total_dataset = train_dataset_obj[:len(train_dataset_obj)]

if not args.classif:
    # Concatenate samples from original Train and Test sets
    # Only 'training' physionet samples are have labels.
    # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
    total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]
print(len(total_dataset))

# Shuffle and split
train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)

record_id, tt, vals, mask, labels = train_data[0]
# record_id, tt, vals, mask, labels = total_dataset[0]

# n_samples = len(total_dataset)
input_dim = vals.size(-1)
data_min, data_max = utils.get_data_min_max(total_dataset, device)
batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)

# batch_size = min(min(len(total_dataset), args.batch_size), args.n)
# total_data_combined = utils.variable_time_collate_fn(total_dataset, device, classify=False, data_min=data_min, data_max=data_max)
flag = 0

if flag:
    test_data_combined = utils.variable_time_collate_fn(test_data, device, classify=args.classif,
                                                    data_min=data_min, data_max=data_max)

    if args.classif:
        train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                random_state=11, shuffle=True)

        train_record_id = [record_id for record_id, tt, vals, mask, labels in train_data]
        train_data_combined = utils.variable_time_collate_fn(
            train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
        val_data_combined = utils.variable_time_collate_fn(
            val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
        print(train_data_combined[1].sum(
        ), val_data_combined[1].sum(), test_data_combined[1].sum())
        print(train_data_combined[0].size(), train_data_combined[1].size(),
                val_data_combined[0].size(), val_data_combined[1].size(),
                test_data_combined[0].size(), test_data_combined[1].size())

    else:
        train_data_combined = utils.variable_time_collate_fn(
            train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
        print(train_data_combined.size(), test_data_combined.size())


# save the data and labels here
# code removed to avoid accidental overwriting of the current data split

os.makedirs('data/pretrain', exist_ok=True)
os.makedirs('data/finetune', exist_ok=True)

def mean_standardize(data, mask):
    features = [[] for i in range(data[0].shape[1])]
    for i in range(len(data)):
        for j in range(data[i].shape[1]):
            values = data[i][:, j][mask[i][:, j].astype(bool)]
            features[j].extend(values)
    
    mean, std = [], []
    for feat in features:
        feat = np.array(feat)
        mean.append(np.mean(feat))
        std.append(np.std(feat))
    mean, std = np.array(mean), np.array(std)

    for i in range(len(data)): 
        data[i] = (data[i] - mean) / std
        data[i] = np.where(mask[i] == 1, data[i], 0.0)

    return data

def pad_sequences(data, mask, time, max_len, num_features):
    padded_data = np.zeros((max_len, num_features))
    padded_mask = np.zeros((max_len, num_features))
    padded_time = np.zeros((max_len, 1))

    padded_data[:data.shape[0], :] = data
    padded_mask[:mask.shape[0], :] = mask
    padded_time[:time.shape[0], 0] = time

    return np.concatenate((padded_data, padded_mask, padded_time), axis=1)

def prepare_data_for_saving(data_tuple, max_len):
    vals, tt, mask = data_tuple
    num_features = vals.shape[1]
    
    tt_padded = np.zeros((max_len, 1))
    tt_padded[:len(tt), 0] = tt

    vals_padded = np.zeros((max_len, num_features))
    vals_padded[:vals.shape[0], :] = vals

    mask_padded = np.zeros((max_len, num_features))
    mask_padded[:mask.shape[0], :] = mask

    combined_data = np.concatenate([vals_padded, tt_padded, mask_padded], axis=1)
    return combined_data

def process_and_save_data(train_data, val_data, test_data, max_len, output_dir):
    def save_data(data, split_name, output_dir):
        X = []
        y = []
        for record_id, tt, vals, mask, label in data:
            if label is not None:
                combined_data = prepare_data_for_saving((vals, tt, mask), max_len)
                X.append(combined_data)
                y.append(label)
        
        print(f"Before saving {split_name}: len(X) = {len(X)}, len(y) = {len(y)}")

        # Ensure X and y have the same length
        assert len(X) == len(y), f"Mismatch: len(X) = {len(X)}, len(y) = {len(y)}"

        X = np.array(X)
        y = np.array(y)
    
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(output_dir, f'X_{split_name}.pt'))
        if len(y) > 0:  # Save y only if labels exist
            torch.save(torch.tensor(y, dtype=torch.long), os.path.join(output_dir, f'y_{split_name}.pt'))

    save_data(train_data, 'train', output_dir)
    save_data(val_data, 'val', output_dir)
    save_data(test_data, 'test', output_dir)

# Shuffle and split the original dataset into training and testing sets
train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)

# Further split the training data into training and validation sets
train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8, random_state=42, shuffle=True)

# Determine the maximum length across all data points for consistent padding
max_len_train = max([tt.size(0) for _, tt, _, _, _ in train_data])
max_len_val = max([tt.size(0) for _, tt, _, _, _ in val_data])
max_len_test = max([tt.size(0) for _, tt, _, _, _ in test_data])

# The overall maximum length
max_len = max(max_len_train, max_len_val, max_len_test)

output_dir = 'data/finetune'  # Set this to your desired output directory

# Process and save the data
process_and_save_data(train_data, val_data, test_data, max_len, output_dir)