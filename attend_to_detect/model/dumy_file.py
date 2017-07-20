#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from __future__ import absolute_import
import sys

from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.transformers import Mapping

import torch
from torch.autograd import Variable
from torch.nn import functional

from sklearn.preprocessing import StandardScaler
import numpy as np

if sys.version_info < (3, 0):
    from .category_branch_2 import CNNRNNEncoder
    from .common_feature_extractor import CommonFeatureExtractor
else:
    from attend_to_detect.model.category_branch_2 import CNNRNNEncoder
    from attend_to_detect.model.common_feature_extractor import CommonFeatureExtractor

__docformat__ = 'reStructuredText'


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


# General variables
batch_size = 64
epochs = 300
dataset_full_path = '../create_dataset/dcase_2017_task_4_test_old.hdf5'

# Variables needed for the common feature extraction layer
common_out_channels = 40
common_kernel_size = (3, 3)
common_stride = (1, 1)
common_padding = (1, 1)
common_dropout = 0.5
common_dilation = (1, 1)
common_activation = functional.leaky_relu


# Variables needed for the alarm branch
branch_alarm_channels_out = [40, 40, 40]
branch_alarm_cnn_kernel_sizes = [(1, 3), (1, 3), (1, 3)]
branch_alarm_cnn_strides = [(1, 2), (1, 2), (1, 2)]
branch_alarm_cnn_paddings = [(0, 0), (0, 0), (0, 0)]
branch_alarm_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_alarm_pool_kernels = [(3, 2), (3, 2), (3, 2)]
branch_alarm_pool_strides = [(3, 2), (3, 2), (3, 2)]
branch_alarm_pool_paddings = [(0, 0), (0, 0), (0, 0)]

branch_alarm_rnn_input_size = 80
branch_alarm_rnn_output_dims = [64, 64]
branch_alarm_rnn_activations = [functional.tanh, functional.tanh]

branch_alarm_dropout_cnn = 0.2
branch_alarm_dropout_rnn_input = 0.2
branch_alarm_dropout_rnn_recurrent = 0.2

branch_alarm_rnn_subsamplings = [3]


# Variables needed for the vehicle branch
branch_vehicle_channels_out = [40, 40, 40]
branch_vehicle_cnn_kernel_sizes = [(1, 3), (1, 3), (1, 3)]
branch_vehicle_cnn_strides = [(1, 2), (1, 2), (1, 2)]
branch_vehicle_cnn_paddings = [(0, 0), (0, 0), (0, 0)]
branch_vehicle_cnn_activations = [functional.leaky_relu, functional.leaky_relu, functional.leaky_relu]

branch_vehicle_pool_kernels = [(3, 2), (3, 2), (3, 2)]
branch_vehicle_pool_strides = [(3, 2), (3, 2), (3, 2)]
branch_vehicle_pool_paddings = [(0, 0), (0, 0), (0, 0)]

branch_vehicle_rnn_input_size = 80
branch_vehicle_rnn_output_dims = [64, 64]
branch_vehicle_rnn_activations = [functional.tanh, functional.tanh]

branch_vehicle_dropout_cnn = 0.2
branch_vehicle_dropout_rnn_input = 0.2
branch_vehicle_dropout_rnn_recurrent = 0.2

branch_vehicle_rnn_subsamplings = [3]


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


def get_data_stream(batch_size, dataset_name='dcase_2017_task_4_test.hdf5',
                    that_set='train', calculate_scaling_metrics=True, old_dataset=True):
    dataset = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size)

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


def get_input(data, scaler, old_dataset=True):
    if old_dataset:
        for i in range(len(data)):
            data[i] = data[i].reshape(data[i].shape[1:])
    x = np.zeros((data.shape[0], ) + data[0].shape)
    for i, datum in enumerate(data):
        x[i, :, :] = scaler.transform(datum)
    return Variable(torch.from_numpy(x.reshape((x.shape[0], 1, ) + x.shape[1:])).float())


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

    return Variable(torch.from_numpy(y_one_hot).float()), \
           Variable(torch.from_numpy(y_categorical).float())


def main():

    # The common feature extraction layer
    common_feature_extractor = CommonFeatureExtractor(
        out_channels=common_out_channels,
        kernel_size=common_kernel_size,
        stride=common_stride,
        padding=common_padding,
        dropout=common_dropout,
        dilation=common_dilation,
        activation=common_activation
    )

    # The alarm branch layers
    branch_alarm = CNNRNNEncoder(
        cnn_channels_in=common_out_channels,
        cnn_channels_out=branch_alarm_channels_out,
        cnn_kernel_sizes=branch_alarm_cnn_kernel_sizes,
        cnn_strides=branch_alarm_cnn_strides,
        cnn_paddings=branch_alarm_cnn_paddings,
        cnn_activations=branch_alarm_cnn_activations,
        max_pool_kernels=branch_alarm_pool_kernels,
        max_pool_strides=branch_alarm_pool_strides,
        max_pool_paddings=branch_alarm_pool_paddings,
        rnn_input_size=branch_alarm_rnn_input_size,
        rnn_out_dims=branch_alarm_rnn_output_dims,
        rnn_activations=branch_alarm_rnn_activations,
        dropout_cnn=branch_alarm_dropout_cnn,
        dropout_rnn_input=branch_alarm_dropout_rnn_input,
        dropout_rnn_recurrent=branch_alarm_dropout_rnn_recurrent,
        rnn_subsamplings=branch_alarm_rnn_subsamplings
    )

    # The vehicle branch layers
    branch_vehicle = CNNRNNEncoder(
        cnn_channels_in=common_out_channels,
        cnn_channels_out=branch_vehicle_channels_out,
        cnn_kernel_sizes=branch_vehicle_cnn_kernel_sizes,
        cnn_strides=branch_vehicle_cnn_strides,
        cnn_paddings=branch_vehicle_cnn_paddings,
        cnn_activations=branch_vehicle_cnn_activations,
        max_pool_kernels=branch_vehicle_pool_kernels,
        max_pool_strides=branch_vehicle_pool_strides,
        max_pool_paddings=branch_vehicle_pool_paddings,
        rnn_input_size=branch_vehicle_rnn_input_size,
        rnn_out_dims=branch_vehicle_rnn_output_dims,
        rnn_activations=branch_vehicle_rnn_activations,
        dropout_cnn=branch_vehicle_dropout_cnn,
        dropout_rnn_input=branch_vehicle_dropout_rnn_input,
        dropout_rnn_recurrent=branch_vehicle_dropout_rnn_recurrent,
        rnn_subsamplings=branch_vehicle_rnn_subsamplings
    )

    # Check if we have GPU, and if we do then GPU them all
    if torch.has_cudnn:
        common_feature_extractor = common_feature_extractor.cuda()
        branch_alarm = branch_alarm.cuda()
        branch_vehicle = branch_vehicle.cuda()

    # Create optimizers for all layers
    optim_common_feature_extractor = torch.optim.Adam(common_feature_extractor.parameters())
    optim_branch_alarm = torch.optim.Adam(branch_alarm.parameters())
    optim_branch_vehicle = torch.optim.Adam(branch_vehicle.parameters())

    # ***********************
    # Add the loss functions

    # Get the training data stream
    train_data, scaler = get_data_stream(
        dataset_name=dataset_full_path,
        batch_size=batch_size
    )

    # Get the validation data stream
    valid_data, _ = get_data_stream(
        dataset_name=dataset_full_path,
        batch_size=batch_size,
        that_set='test',
        calculate_scaling_metrics=False,
    )

    for epoch in range(epochs):

        for batch in train_data.get_epoch_iterator():
            # Get input
            x = get_input(batch[0], scaler)

            # Get target values for alarm classes
            y_alarm_1_hot, y_alarm_logits = get_output(batch[-2])

            # Get target values for vehicle classes
            y_vehicle_1_hot, y_vehicle_logits = get_output(batch[-1])

            # Go through the common feature extractor
            common_features = common_feature_extractor(x)

            # Go through the alarm branch
            alarm_output = branch_alarm(common_features)

            # Go through the vehicle branch
            vehicle_output = branch_vehicle(common_features)

            # Zero out the grads in optimizers
            common_feature_extractor.zero_grad()
            branch_alarm.zero_grad()
            branch_vehicle.zero_grad()

            # Calculate losses, do backward passing, and do updates

        for batch in valid_data.get_epoch_iterator():
            # Get input
            x = get_input(batch[0], scaler)

            # Get target values for alarm classes
            y_alarm_1_hot, y_alarm_logits = get_output(batch[-2])

            # Get target values for vehicle classes
            y_vehicle_1_hot, y_vehicle_logits = get_output(batch[-1])

            # Go through the common feature extractor
            common_features = common_feature_extractor(x)

            # Go through the alarm branch
            alarm_output = branch_alarm(common_features)

            # Go through the vehicle branch
            vehicle_output = branch_vehicle(common_features)

            # Calculate validation losses

if __name__ == '__main__':
    main()

# EOF
