import numpy as np

import torch

from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

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

all_classes = alarm_classes + vehicle_classes


def padder(data):
    data = list(data)
    for index in [0, -2, -1]:
        max_ts = np.max([datum.shape[-2] for datum in data[index]])

        for i in range(len(data[index])):
            len_dif = max_ts - data[index][i].shape[-2]
            if len_dif > 0:
                data[index][i] = np.concatenate((
                    data[index][i],
                    np.zeros((1, len_dif, data[index][i].shape[-1]))),
                    axis=-2
                )
                if index != 0:
                    data[index][i][:, -len_dif:, 0] = 1
    data = tuple(data)

    return data


def padder_single(data):
    data = list(data)
    for index in [0]:
        max_ts = np.max([datum.shape[-2] for datum in data[index]])

        for i in range(len(data[index])):
            len_dif = max_ts - data[index][i].shape[-2]
            if len_dif > 0:
                data[index][i] = np.concatenate((
                    data[index][i],
                    np.zeros((1, len_dif, data[index][i].shape[-1]))),
                    axis=-2
                )
                if index != 0:
                    data[index][i][:, -len_dif:, 0] = 1
    data = tuple(data)

    return data


def get_data_stream(batch_size, dataset_name='dcase_2017_task_4_test.hdf5',
                    that_set='train', calculate_scaling_metrics=True, old_dataset=True,
                    examples=None):
    dataset = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
    if examples is None:
        examples = dataset.num_examples
    scheme = ShuffledScheme(examples=examples, batch_size=batch_size)

    scaler = StandardScaler()

    stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    stream = Mapping(stream, mapping=padder)

    if calculate_scaling_metrics:
        for data in stream.get_epoch_iterator():
            for example in data[0]:
                if old_dataset:
                    scaler.partial_fit(example.reshape(example.shape[1:]))

        return stream, scaler
    return stream, None


def get_data_stream_single(batch_size, dataset_name='dcase_2017_task_4_test.hdf5',
                    that_set='train', calculate_scaling_metrics=True, old_dataset=True,
                    examples=None):
    dataset = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
    if examples is None:
        examples = dataset.num_examples
    scheme = ShuffledScheme(examples=examples, batch_size=batch_size)

    scaler = StandardScaler()

    stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    stream = Mapping(stream, mapping=padder_single)

    if calculate_scaling_metrics:
        for data in stream.get_epoch_iterator():
            for example in data[0]:
                if old_dataset:
                    scaler.partial_fit(example.reshape(example.shape[1:]))

        return stream, scaler
    return stream, None


def get_input(data, scaler, old_dataset=True, volatile=False):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])
    x = np.zeros((data.shape[0], ) + data[0].shape)
    for i, datum in enumerate(data):
        x[i, :, :] = scaler.transform(datum)
    x = Variable(torch.from_numpy(x.reshape((x.shape[0], 1, ) + x.shape[1:])).float(),
                 volatile=volatile)
    if torch.has_cudnn:
        x = x.cuda()
    return x


def get_output(data, old_dataset=True):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])

    y_one_hot = np.zeros((data.shape[0], ) + data[0].shape)
    y_categorical = np.zeros((data.shape[0], ) + data[0].shape[0:1])

    for i, datum in enumerate(data):
        y_one_hot[i, :, :] = datum
        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        y_categorical[i, :] = non_zeros

    y_one_hot = Variable(torch.from_numpy(y_one_hot).float(), requires_grad=False)
    y_categorical = Variable(torch.from_numpy(y_categorical).long(), requires_grad=False)
    if torch.has_cudnn:
        y_one_hot = y_one_hot.cuda()
        y_categorical = y_categorical.cuda()
    return y_one_hot, y_categorical


def get_output_binary(data, old_dataset=True):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])
    y_one_hot = np.zeros((data.shape[0], data[0].shape[-1] - 1, 1))

    for i, datum in enumerate(data):
        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        for n in non_zeros:
            if n > 0:
                y_one_hot[i, n-1, 0] = 1

    y_one_hot = Variable(torch.from_numpy(y_one_hot).float(), requires_grad=False)
    if torch.has_cudnn:
        y_one_hot = y_one_hot.cuda()
    return y_one_hot, None


def get_output_binary_single(data_a, data_v, old_dataset=True):
    """

    """
    if old_dataset:
        for i in range(len(data_a)):
            data_a[i] = data_a[i].reshape(data_a[i].shape[1:])
        for i in range(len(data_v)):
            data_v[i] = data_v[i].reshape(data_v[i].shape[1:])

    total_classes = data_a[0].shape[-1] - 1
    total_classes += data_v[0].shape[-1] - 1
    binarized_classes = np.zeros((data_a.shape[0], total_classes, 1))

    for i, datum in enumerate(data_a):
        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        for n in non_zeros:
            if n > 0:
                binarized_classes[i, n - 1, 0] = 1

    for i, datum in enumerate(data_v):
        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        for n in non_zeros:
            if n > 0:
                binarized_classes[i, n + data_a[0].shape[-1] - 2, 0] = 1

    binarized_classes = Variable(torch.from_numpy(binarized_classes).float(), requires_grad=False)
    if torch.has_cudnn:
        binarized_classes = binarized_classes.cuda()
    return binarized_classes


def get_output_binary_one_hot(data_a, data_v):
    max_i = 0
    for i in range(len(data_a)):
        data_a[i] = data_a[i].reshape(data_a[i].shape[1:])
        data_v[i] = data_v[i].reshape(data_v[i].shape[1:])
        current_i = data_a[i].shape[0] + data_v[i].shape[0] - 1
        if max_i <= current_i:
            max_i = current_i

    total_classes = len(alarm_classes) + len(vehicle_classes) + 1

    y_one_hot = np.zeros((data_a.shape[0], max_i, total_classes))
    y_categorical = np.zeros((data_a.shape[0], max_i))

    for i in range(len(data_a)):
        datum = np.zeros((max_i, total_classes))
        l_l = 0
        u_l = l_l + data_a[i].shape[0] - 1
        datum[l_l:u_l, 1:len(alarm_classes) + 1] = data_a[i][:-1, 1:]
        l_l += u_l
        u_l = l_l + data_v[i].shape[0] - 1
        datum[l_l:u_l, -len(vehicle_classes):] = data_v[i][:-1, 1:]
        datum[u_l:, 0] = 1

        non_zeros = [np.nonzero(dd) for dd in datum]
        non_zeros = [n[0][0] for n in non_zeros]
        y_categorical[i, :] = non_zeros
        y_one_hot[i, :, :] = datum

    return y_one_hot, y_categorical
