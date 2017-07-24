import torch
from torch.utils.data import Dataset, DataLoader
from fuel.datasets import H5PYDataset
from fuel.schemes import *
from fuel.streams import DataStream
import numpy as np

from attend_to_detect.evaluation import all_freqs

alarm_classes = [
    'Train horn',
    'Air horn',
    'Car alarm',
    'Reversing beeps',
    'Ambulance (siren)',
    'Police car (siren)',
    'fire truck (siren)',
    'Civil defense siren',
    'Screaming'
]

vehicle_classes = [
    'Bicycle',
    'Skateboard',
    'Car',
    'Car passing by',
    'Bus',
    'Truck',
    'Motorcycle',
    'Train'
]


def compute_weights(dataset):
    weights = []
    loader = DataLoader(dataset)

    for sample in loader:
        x, y1, y2 = sample
        _, class1 = torch.max(y1, 2)
        _, class2 = torch.max(y2, 2)
        class1 = class1.squeeze().tolist()
        class2 = class2.squeeze().tolist()
        class1 = list(filter(lambda x: x > 0, class1))
        class2 = filter(lambda x: x > 0, class2)
        class2 = [x + len(alarm_classes) for x in class2]
        freqs = [all_freqs[i-1] for i in class1 + class2]
        weights.append(1./min(freqs))
    return weights


class ChallengeDataset(Dataset):
    def __init__(self, dataset_name, that_set, scaler,
                 shuffle_targets=False):
        super(ChallengeDataset, self).__init__()
        raw_data = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
        scheme = SequentialExampleScheme(raw_data.num_examples)
        self.data = DataStream(dataset=raw_data, iteration_scheme=scheme)
        self.scaler = scaler
        self.shuffle_targets = shuffle_targets

    def __getitem__(self, idx):
        eos = torch.LongTensor(1).zero_()
        x, _, _, y1, y2 = self.data.get_data(idx)
        x_norm = self.scaler.transform(x.squeeze())
        # Get classes from HDF5 file and convert to categorical
        y_categorical = []
        if y1.shape[1] > 1:
            y1 = torch.from_numpy(y1[:, :-1, 1:])
            y1_categorical = torch.max(y1, -1)[1] + 1
            y_categorical += y1_categorical.squeeze().tolist()
        if y2.shape[1] > 1:
            y2 = torch.from_numpy(y2[:, :-1, 1:])
            y2_categorical = torch.max(y2, -1)[1] + 1 + len(alarm_classes)
            y_categorical += y2_categorical.squeeze().tolist()
        y = torch.cat([torch.LongTensor(y_categorical), eos])
        if self.shuffle_targets and y.size(0) > 2:
            y[:-1] = y[torch.randperm(y.size(0) - 1)]
        return torch.FloatTensor(x_norm[np.newaxis]), y.view(1, -1)

    def one_hot(self, y, all_classes):
        y_1_hot = torch.zeros(y.size(0), 1 + len(all_classes))
        y_1_hot.scatter_(1, y, 1.)
        y_1_hot = y_1_hot[:, 1:].contiguous()
        return y_1_hot

    def __len__(self):
        return self.data.dataset.num_examples

    @staticmethod
    def collate(batch):
        feats, targets = zip(*batch)
        # Pad features
        max_len_x = max([sample.size(1) for sample in feats])
        X = torch.FloatTensor(len(feats), 1, max_len_x, feats[0].size(2)).zero_()
        max_len_y = max([sample.size(1) for sample in targets])
        Y = torch.LongTensor(len(targets), max_len_y).zero_()
        for k, (x, y) in enumerate(zip(feats, targets)):
            X[k, 0, :x.size(1), :] = x
            Y[k, :y.size(1)] = y
        return X, Y


if __name__ == '__main__':
    import pickle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    dataset = ChallengeDataset(dataset_name='attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5',
            that_set='train', scaler=scaler, shuffle_targets=True)
    x = dataset[0]

    # Test oversampling
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)

    from torch.utils.data.sampler import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(dataset))
    loader = DataLoader(dataset,
            batch_size=4,
            sampler=sampler,
            collate_fn=ChallengeDataset.collate)
    iterator = loader.__iter__()
    for k in range(10):
        x, y = next(iterator)


