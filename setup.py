# setup.py

import os

# Install required packages
os.system('pip install dvc dvc[gdrive] torch numpy pandas')
os.system('pip install git+https://github.com/synsense/rockpool@develop')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rockpool.nn.networks import SynNet
from scipy import sparse
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to the models directory on Google Drive
model_dir = '/content/drive/MyDrive/CITS5553/models/'

# Function to preprocess spikes
def raster(spikes: np.ndarray):
    spikes = np.asarray(spikes)
    dt = 1 / 110e3
    num_clk_per_period = 0.01 / dt
    spike_sum = np.cumsum(spikes, axis=0)[:: int(num_clk_per_period), :]
    spike_sum[1:, :] -= spike_sum[:-1, :]

    spike_sum[spike_sum > 15] = 15

    return spike_sum

# Function to preload samples
def preload_samples(data, base_path, target_length):
    preloaded_samples = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        path = row['path']
        data_path = base_path + '/' + path
        sparse_spikes = sparse.load_npz(data_path)
        dense_spikes = sparse_spikes.toarray()
        raster_spikes = torch.Tensor(raster(dense_spikes))

        sample_length = raster_spikes.size(0)
        if sample_length < target_length:
            padding = target_length - sample_length
            raster_spikes = F.pad(raster_spikes, (0, 0, 0, padding))
        elif sample_length > target_length:
            start_idx = (sample_length - target_length) // 2
            end_idx = start_idx + target_length
            raster_spikes = raster_spikes[start_idx:end_idx]

        preloaded_samples.append(raster_spikes)

    data['tensor'] = preloaded_samples

    return data[['class', 'tensor']]

# Dataset class
class SpikeDataset(Dataset):
    def __init__(self, annotations_file, split="train", target_length=200, alpha=1):
        self._label_map = {"car": 0, "cv": 1, "background": 2}
        self.spike_dir = os.path.dirname(annotations_file)
        self.split = split
        self.target_length = target_length
        self.alpha = alpha

        data = pd.read_csv(annotations_file)
        if split == "train":
            train_spiked = data[data["split"] == "train"]
            self.loaded_data = preload_samples(train_spiked, self.spike_dir, target_length)
        elif split == "val":
            val_spiked = data[data["split"] == "val"]
            self.loaded_data = preload_samples(val_spiked, self.spike_dir, target_length)
        elif split == "test":
            test_spiked = data[data["split"] == "test"]
            self.loaded_data = preload_samples(test_spiked, self.spike_dir, target_length)

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        label = self.loaded_data.iloc[idx]['class']
        label_idx = self._label_map[label]
        label_tensor = torch.full((self.target_length, 3), -self.alpha)
        label_tensor[:, label_idx] = self.alpha

        raster_spikes = self.loaded_data.iloc[idx]['tensor']

        return raster_spikes, label_tensor.float()

# Create model
net = SynNet(
    output="vmem",
    n_channels=16,  # Number of input channels
    n_classes=3,  # Number of output classes (output channels)
    size_hidden_layers=[24, 24, 24],  # Number of neurons in each hidden layer
    time_constants_per_layer=[2, 4, 8],  # Number of time constants in each hidden layer
).to(device)

# Define optimizer and loss function
optimizer = Adam(net.parameters().astorch(), lr=1e-3)
loss_fun = MSELoss()

# Load the models and datasets
initial_model = torch.load(model_dir + 'initial.pt')
train_dataset = torch.load(model_dir + 'train.pt')
val_dataset = torch.load(model_dir + 'val.pt')
test_dataset = torch.load(model_dir + 'test.pt')

print("Setup complete: Packages installed, and models loaded successfully.")
