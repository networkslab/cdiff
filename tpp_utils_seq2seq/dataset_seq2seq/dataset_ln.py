from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch.utils.data as data_utils
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
import scipy
import argparse
import tpp_utils_seq2seq.dataset_seq2seq.Constants as Constants
from sklearn.preprocessing import PowerTransformer
# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
import math

def load_dataset_ln(dataset_dir, mode, target_length=20, batch_size=64, shuffle=False, device=None,
                    scale=-1, train_mean=-1, train_std=-1, train_min=-1, train_ln_mean=-1, train_ln_std=-1,
                    train_ln_min=-1,
                    **kwargs):
    """

    :param dataset_dir:
    :param mode:
    :param target_length: default 20
    :param batch_size: default 64
    :param shuffle: defualt False
    :param device:
    :param kwargs:
    :return:
    """

    print('loading {} datasets...'.format(mode))

    dataset = SeqDatasetLn(
        dataset_dir=dataset_dir, mode=mode, target_length=target_length, device=device,
        train_mean=train_mean, train_std=train_std, train_min=train_min, train_ln_mean=train_ln_mean,
        train_ln_std=train_ln_std, train_ln_min=train_ln_min,

    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collateln)

    return data_loader, dataset

def one_hot_embedding(labels: torch.Tensor, num_types: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_types: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_types).to(device)
    return y[labels.long()][..., :-1]

class SeqDatasetLn(Dataset):
    def __init__(self, dataset_dir, mode, target_length=20, device=None, data_name='taxi',
                 train_mean=-1, train_std=-1, train_min=-1, train_ln_mean=-1, train_ln_std=-1, train_ln_min=-1):
        """
        """
        assert mode in {'train', 'dev', 'test'}, 'the mode should be train or dev or test'
        print('data is {} mode is {} for ln'.format(data_name, mode))
        # Get dataset directory
        dataset_dir = os.path.join(dataset_dir, '{}.pkl'.format(mode))

        # Get data
        self.data, self.num_types = self.load_dataset_hypro_format(dataset_dir, mode)

        # Get the target length (forecasting length)
        self.target_length = target_length

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # time step
        time_seq = [[x["time_since_start"] for x in seq] for seq in self.data]
        self.time_seq = [torch.tensor(seq[1:]) for seq in time_seq]
        # # Need to know the length of the sequences for dealing with the varied padding
        # self.history_times = [seq[:-target_length] for seq in self.time_seq]
        # self.seq_lengths = [seq.size(0) for seq in self.history_times]
        # self.target_times = [seq[-target_length:] for seq in self.time_seq]

        # event type
        event_seq = [[x["type_event"] for x in seq] for seq in self.data]
        # The num_types is the padding
        # i.e., if there are 22 event types, then 0-21 is the type indicator, 22 is the padding
        self.event_seq = [torch.tensor(seq[1:]) for seq in event_seq]
        # self.history_types = [seq[:-target_length] for seq in self.event_seq]
        # self.target_types = [seq[-target_length:] for seq in self.event_seq]

        # inter-arrival time
        time_delta_seq = [[x["time_since_last_event"] for x in seq] for seq in self.data]

        # Un-normalized inter-arrival time
        self.unnormed_time_delta_seq = [torch.tensor(seq[1:])+Constants.EPS for seq in time_delta_seq]

        if mode == 'train':
            self.mean_inter_time, self.std_inter_time = self.get_mean_std(self.unnormed_time_delta_seq)
            self.min_inter_time = self.get_min(self.unnormed_time_delta_seq)
        else:
            self.mean_inter_time = train_mean
            self.std_inter_time = train_std
            self.min_inter_time = train_min
        print('mean: {:.3f} | std: {:.3f} | min: {:.3f}'.format(self.mean_inter_time,
                                                                self.std_inter_time,
                                                                self.min_inter_time))
        if data_name == 'retweet':
            self.normed_time_delta_seq = [
                torch.log((torch.tensor(seq).float() + Constants.EPS) * Constants.SCALE_RETWEET)[1:] for seq in
                time_delta_seq]
        else:
            self.normed_time_delta_seq = [torch.log((torch.tensor(seq).float() + Constants.EPS)*Constants.SCALE_UNIFORM)[1:] for seq in time_delta_seq]

        if mode == 'train':
            self.ln_mean, self.ln_std = self.get_mean_std(self.normed_time_delta_seq)
            self.ln_min_inter_time = self.get_min(self.normed_time_delta_seq)
        else:
            self.ln_mean = train_ln_mean
            self.ln_std = train_ln_std
            self.ln_min_inter_time = train_ln_min
        for i in range(len(self.normed_time_delta_seq)):
            self.normed_time_delta_seq[i] = (self.normed_time_delta_seq[i] - self.ln_mean) / self.ln_std


        self.history_times = []
        self.target_times = []
        self.history_types = []
        self.target_types = []
        self.unnormed_history_dt = []
        self.unnormed_target_dt = []
        self.history_dt = []
        self.target_dt = []
        if mode == 'train':
            for seq_time, seq_type, seq_dt_unormed, seq_dt_normed in zip(self.time_seq,
                                                                         self.event_seq,
                                                                         self.unnormed_time_delta_seq,
                                                                         self.normed_time_delta_seq):
                for i in range(1, 5):
                    self.history_times.append(seq_time[:-target_length - i])
                    self.target_times.append(seq_time[-target_length - i:-i])
                    self.history_types.append(seq_type[:-target_length - i])
                    self.target_types.append(seq_type[-target_length - i:-i])
                    self.unnormed_history_dt.append(seq_dt_unormed[:-target_length - i])
                    self.unnormed_target_dt.append(seq_dt_unormed[-target_length - i:-i])
                    self.history_dt.append(seq_dt_normed[:-target_length - i])
                    self.target_dt.append(seq_dt_normed[-target_length - i:-i])
                self.history_times.append(seq_time[:-target_length])
                self.target_times.append(seq_time[-target_length:])
                self.history_types.append(seq_type[:-target_length])
                self.target_types.append(seq_type[-target_length:])
                self.unnormed_history_dt.append(seq_dt_unormed[:-target_length])
                self.unnormed_target_dt.append(seq_dt_unormed[-target_length:])
                self.history_dt.append(seq_dt_normed[:-target_length])
                self.target_dt.append(seq_dt_normed[-target_length:])
        else:
            for seq_time, seq_type, seq_dt_unormed, seq_dt_normed in zip(self.time_seq,
                                                                         self.event_seq,
                                                                         self.unnormed_time_delta_seq,
                                                                         self.normed_time_delta_seq):
                self.history_times.append(seq_time[:-target_length])
                self.target_times.append(seq_time[-target_length:])
                self.history_types.append(seq_type[:-target_length])
                self.target_types.append(seq_type[-target_length:])
                self.unnormed_history_dt.append(seq_dt_unormed[:-target_length])
                self.unnormed_target_dt.append(seq_dt_unormed[-target_length:])
                self.history_dt.append(seq_dt_normed[:-target_length])
                self.target_dt.append(seq_dt_normed[-target_length:])

        self.seq_lengths = [seq.size(0) for seq in self.history_times]
        self.length = len(self.history_times)

        self.seq_lengths = [seq.size(0) for seq in self.history_times]
        self.length = len(self.target_dt)

    def get_median(self, unnormed_time_delta_seq):
        flat_out_times = torch.cat(unnormed_time_delta_seq)
        return torch.median(flat_out_times)

    def get_mean_std(self, unnormed_time_delta_seq):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(unnormed_time_delta_seq)
        return flat_out_times.mean(), flat_out_times.std()

    def get_min(self, unnormed_time_delta_seq):
        """Get the min inter-arrival time"""
        flat_out_times = torch.cat(unnormed_time_delta_seq)
        return flat_out_times.min()

    def __getitem__(self, key):
        """

        :param key:
        :return: history_times, history_types, history_dt,
                 target_times, target_types, target_dt,
                 num_types, device, unnormed_history_dt, unnormed_target_dt
        """
        return self.history_times[key], self.history_types[key], self.history_dt[key], \
               self.target_times[key], self.target_types[key], self.target_dt[key], \
               self.num_types, self.device, self.unnormed_history_dt[key], self.unnormed_target_dt[key], \
               self.seq_lengths[key]

    def __len__(self):
        return self.length

    def load_dataset_hypro_format(self, dataset_dir, dict_name):
        with open(dataset_dir, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

def collateln(batch):
    """
    :param batch: history_times, history_types, history_dt,
                  target_times, target_types, target_dt,
                  num_types, device, unnormed_history_dt, unnormed_target_dt
    :return: Batch instance, batch:
                  history_times, history_types, history_dt,
                  target_times, target_types, target_dt, target_onehots,
                  unnormed_history_dt, unnormed_target_dt
    """
    num_types = batch[0][6]
    device = batch[0][7]

    history_times = [item[0] for item in batch]
    history_types = [item[1] for item in batch]
    history_dt = [item[2] for item in batch]

    target_times = [item[3] for item in batch]
    target_types = [item[4] for item in batch]
    target_dt = [item[5] for item in batch]

    unnormed_history_dt = [item[8] for item in batch]
    unnormed_target_dt = [item[9] for item in batch]
    seq_lengths = torch.tensor([item[10] for item in batch])
    history_times = torch.nn.utils.rnn.pad_sequence(history_times, batch_first=True, padding_value=0.0)
    history_dt = torch.nn.utils.rnn.pad_sequence(history_dt, batch_first=True, padding_value=0.0)
    history_types = torch.nn.utils.rnn.pad_sequence(history_types, batch_first=True, padding_value=num_types)

    target_times = torch.nn.utils.rnn.pad_sequence(target_times, batch_first=True, padding_value=0.0)
    target_dt = torch.nn.utils.rnn.pad_sequence(target_dt, batch_first=True, padding_value=0.0)
    target_types = torch.nn.utils.rnn.pad_sequence(target_types, batch_first=True, padding_value=num_types)
    target_onehots = one_hot_embedding(target_types, num_types + 1)

    unnormed_history_dt = torch.nn.utils.rnn.pad_sequence(unnormed_history_dt, batch_first=True, padding_value=0.0)
    unnormed_target_dt = torch.nn.utils.rnn.pad_sequence(unnormed_target_dt, batch_first=True, padding_value=0.0)

    return Batch(
        history_times.to(device),
        history_types.to(device),
        history_dt.to(device),
        target_times.to(device),
        target_types.to(device),
        target_dt.to(device),
        target_onehots.to(device),
        unnormed_history_dt.to(device),
        unnormed_target_dt.to(device),
        seq_lengths.to(device)
    )


class Batch:
    def __init__(self, history_times, history_types, history_dt,
                 target_times, target_types, target_dt, target_onehots,
                 unnormed_history_dt, unnormed_target_dt, seq_lengths):
        self.history_times = history_times
        self.history_types = history_types.long()
        self.history_dt = history_dt
        self.target_times = target_times.long()
        self.target_types = target_types
        self.target_dt = target_dt
        self.target_onehots = target_onehots.long()
        self.unnormed_history_dt = unnormed_history_dt
        self.unnormed_target_dt = unnormed_target_dt
        self.seq_lengths = seq_lengths


