import h5py
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset


class H5PklDataset(Dataset):
    """
    Custom Dataset to load data from an h5 file and corresponding labels from a pkl file.
    Handles sequence segmentation, padding, and mask generation.
    """
    def __init__(self, h5_file_path, pkl_file_path, max_length):
        """
        Initialize the dataset.

        Args:
            h5_file_path (str): Path to the h5 file containing feature data.
            pkl_file_path (str): Path to the pkl file containing label data.
            max_length (int): Maximum length of each sequence segment.
        """
        # Load h5 file
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.datasets = list(self.h5_file.keys())  # Get all dataset names
        self.labels = self._load_labels(pkl_file_path)  # Load labels from pkl file
        self.max_length = max_length

    def _load_labels(self, pkl_file_path):
        """
        Load labels from the pkl file.

        Args:
            pkl_file_path (str): Path to the pkl file.

        Returns:
            dict: A dictionary mapping index to label.
        """
        with open(pkl_file_path, 'rb') as pkl_file:
            df = pickle.load(pkl_file)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("The pkl file does not contain a DataFrame.")
            # Create a dictionary mapping index to label
            return dict(zip(df.iloc[:, 0], df['human']))

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.datasets)

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (data, label, mask), where:
                - data: Tensor of shape (num_segments, max_length, feature_dim)
                - label: Tensor of shape (1,)
                - mask: Tensor of shape (num_segments, max_length), indicating original vs. padded.
        """
        # Retrieve dataset name and feature from h5 file
        dataset_name = self.datasets[idx]
        feature = self.h5_file[dataset_name][()]  # Load feature as a numpy array
        if feature.ndim == 1:
            # Per-protein mean embedding; keep a sequence dimension for downstream code.
            feature = feature.reshape(1, -1)
        elif feature.ndim != 2:
            raise ValueError(f"Expected feature to be 1D or 2D, but got shape {feature.shape}")

        # Extract index from dataset name (e.g., "index-123")
        index = int(dataset_name.split('-')[-1])

        # Retrieve label from the preloaded labels dictionary
        if index in self.labels:
            label = self.labels[index]
        else:
            raise KeyError(f"Index {index} not found in the pkl file.")

        # Segment the sequence and pad
        seq_length, feature_dim = feature.shape
        if feature_dim != 1024:
            raise ValueError(f"Expected feature_dim to be 1024, but got {feature_dim}")

        # Calculate the number of segments
        num_segments = (seq_length + self.max_length - 1) // self.max_length

        # Initialize padded data and mask
        padded_data = torch.zeros((num_segments, self.max_length, feature_dim), dtype=torch.float32)
        mask = torch.zeros((num_segments, self.max_length), dtype=torch.bool)

        for i in range(num_segments):
            start_idx = i * self.max_length
            end_idx = min(start_idx + self.max_length, seq_length)
            segment_length = end_idx - start_idx

            # Copy original data into padded_data
            padded_data[i, :segment_length, :] = torch.tensor(feature[start_idx:end_idx], dtype=torch.float32)

            # Mark the valid positions in the mask
            mask[i, :segment_length] = True
        # print(f"Sample {idx}: data shape {padded_data.shape}, mask shape {mask.shape}")
        return padded_data, torch.tensor(label, dtype=torch.float32), mask

def collate_fn(batch):
    """
    Collate function to pad each sample's segments to the maximum number of segments in the batch.
    Each sample is a tuple (data, label, mask) where:
      - data shape: (num_segments, max_length, feature_dim)
      - mask shape: (num_segments, max_length)
    """
    datas, labels, masks = zip(*batch)
    
    # Determine the maximum number of segments in this batch
    max_segments = max(d.shape[0] for d in datas)
    
    padded_datas = []
    padded_masks = []
    
    for i, (d, m) in enumerate(zip(datas, masks)):
        num_segments = d.shape[0]
        pad_segments = max_segments - num_segments
        
        # Pad data: pad along the segment dimension so that shape becomes (max_segments, max_length, feature_dim)
        if pad_segments > 0:
            # Create a tensor of zeros for padding with shape (pad_segments, max_length, feature_dim)
            pad_tensor = torch.zeros((pad_segments, *d.shape[1:]), dtype=d.dtype)
            # Concatenate along the first dimension (segments)
            padded_d = torch.cat([d, pad_tensor], dim=0)
        else:
            padded_d = d
        # print(f"Sample {i} padding后 shape: {padded_d.shape}")
        padded_datas.append(padded_d)
        
        # Pad mask: similarly pad mask along the segment dimension to have (max_segments, max_length)
        if pad_segments > 0:
            pad_mask = torch.zeros((pad_segments, *m.shape[1:]), dtype=m.dtype)
            padded_m = torch.cat([m, pad_mask], dim=0)
        else:
            padded_m = m
        padded_masks.append(padded_m)
    
    # Stack all samples into a batch
    padded_data = torch.stack(padded_datas)  # shape: (batch_size, max_segments, max_length, feature_dim)
    padded_masks = torch.stack(padded_masks)   # shape: (batch_size, max_segments, max_length)
    labels = torch.stack(labels)               # shape: (batch_size,)
    
    return padded_data, labels, padded_masks
