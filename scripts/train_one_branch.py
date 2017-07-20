#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import importlib
import numpy as np
import timeit
import shutil
from argparse import ArgumentParser
from contextlib import closing
from mimir import Logger
from tqdm import tqdm

import torch
from torch.nn.utils import clip_grad_norm

from attend_to_detect.dataset import (
    vehicle_classes, alarm_classes, get_input, get_output_binary_one_hot, get_data_stream_single,
    get_output_binary_single)
from attend_to_detect.model import CNNRNNEncoder
from attend_to_detect.evaluation import validate_single_branch, \
    binary_category_cost_single, binary_accuracy_single, manual_b_entropy, multi_label_loss

__docformat__ = 'reStructuredText'


def main():
    # Getting configuration file from the command line argument
    parser = ArgumentParser()
    parser.add_argument('--train-examples', type=int, default=-1)
    parser.add_argument('config_file')
    parser.add_argument('checkpoint_path')
    parser.add_argument('--print-grads', action='store_true')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--visdom-port', type=int, default=5004)
    parser.add_argument('--visdom-server', default='http://localhost')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--job-id', default='')
    args = parser.parse_args()

    if args.debug:
        torch.has_cudnn = False

    config = importlib.import_module(args.config_file)

    # The alarm branch layers
    network = CNNRNNEncoder(
        cnn_channels_in=1,
        cnn_channels_out=config.network_channels_out,
        cnn_kernel_sizes=config.network_cnn_kernel_sizes,
        cnn_strides=config.network_cnn_strides,
        cnn_paddings=config.network_cnn_paddings,
        cnn_activations=config.network_cnn_activations,
        max_pool_kernels=config.network_pool_kernels,
        max_pool_strides=config.network_pool_strides,
        max_pool_paddings=config.network_pool_paddings,
        rnn_input_size=config.network_rnn_input_size,
        rnn_out_dims=config.network_rnn_output_dims,
        rnn_activations=config.network_rnn_activations,
        dropout_cnn=config.network_dropout_cnn,
        dropout_rnn_input=config.network_dropout_rnn_input,
        dropout_rnn_recurrent=config.network_dropout_rnn_recurrent,
        rnn_subsamplings=config.network_rnn_subsamplings,
        decoder_dim=config.network_decoder_dim,
        output_classes=1,
        attention_bias=config.network_attention_bias,
        init=config.network_init
    )

    print('Total parameters: {}'.format(network.nb_trainable_parameters()))

    # Check if we have GPU, and if we do then GPU them all
    if torch.has_cudnn:
        network = network.cuda()

    # Create optimizer for all parameters
    optim = config.optimizer(network.parameters(), lr=config.optimizer_lr)

    # Do we have a checkpoint?
    if os.path.isdir(args.checkpoint_path):
        print('Checkpoint directory exists!')
        if os.path.isfile(os.path.join(args.checkpoint_path, 'latest.pt')):
            print('Loading checkpoint...')
            ckpt = torch.load(os.path.join(args.checkpoint_path, 'latest.pt'))
            network.load_state_dict(ckpt['network'])
            optim.load_state_dict(ckpt['optim'])
    else:
        print('Checkpoint directory does not exist, creating...')
        os.makedirs(args.checkpoint_path)

    if args.train_examples == -1:
        examples = None
    else:
        examples = args.train_examples

    if os.path.isfile('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        train_data, _ = get_data_stream_single(
            dataset_name=config.dataset_full_path,
            batch_size=config.batch_size,
            calculate_scaling_metrics=False,
            examples=examples)
    else:
        train_data, scaler = get_data_stream_single(
            dataset_name=config.dataset_full_path,
            batch_size=config.batch_size,
            calculate_scaling_metrics=True,
            examples=examples)
        # Serialize scaler so we don't need to do this again
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    # Get the validation data stream
    valid_data, _ = get_data_stream_single(
        dataset_name=config.dataset_full_path,
        batch_size=config.batch_size,
        that_set='test',
        calculate_scaling_metrics=False,
    )

    logger = Logger("{}_log.jsonl.gz".format(args.checkpoint_path),
                    formatter=None)
    if args.visdom:
        from attend_to_detect.utils.visdom_handler import VisdomHandler
        title_losses = 'Train/valid losses'
        title_accu = 'Train/valid accuracies'
        if args.job_id != '':
            title_losses += ' - Job ID: {}'.format(args.job_id)
            title_accu += ' - Job ID: {}'.format(args.job_id)
        loss_handler = VisdomHandler(
            ['training', 'validation'],
            'loss',
            dict(title=title_losses,
                 xlabel='iteration',
                 ylabel='cross-entropy'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(loss_handler)
        accuracy_handler = VisdomHandler(
            ['training', 'validation'],
            'acc',
            dict(title=title_accu,
                 xlabel='iteration',
                 ylabel='accuracy, %'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(accuracy_handler)

    with closing(logger):
        train_loop(
            config, network,
            train_data, valid_data, scaler, optim, args.print_grads, logger,
            args.checkpoint_path, args.no_tqdm)


def iterate_params(pytorch_module):
    has_children = False
    for child in pytorch_module.children():
        for pair in iterate_params(child):
            yield pair
        has_children = True
    if not has_children:
        for name, parameter in pytorch_module.named_parameters():
            yield (parameter, name, pytorch_module)


def train_loop(config, network, train_data, valid_data, scaler,
               optim, print_grads, logger, checkpoint_path, no_tqdm):
    total_iterations = 0
    # loss_module = torch.nn.MSELoss()
    for epoch in range(config.epochs):
        network.train()
        losses = []
        accuracies = []
        epoch_start_time = timeit.timeit()
        epoch_iterator = enumerate(train_data.get_epoch_iterator())
        if not no_tqdm:
            epoch_iterator = tqdm(epoch_iterator,
                                  total=50000 // config.batch_size)
        for iteration, batch in epoch_iterator:
            # Get input
            x = get_input(batch[0], scaler)

            # Get target values for alarm classes
            y_1_hot = get_output_binary_single(batch[-2], batch[-1])

            # Go through the alarm branch
            network_output, attention_weights = network(x, y_1_hot.shape[1])

            target_values = torch.autograd.Variable(torch.from_numpy(y_1_hot.sum(axis=1)).float())
            if torch.has_cudnn:
                target_values = target_values.cuda()

            # Calculate losses, do backward passing, and do updates
            # loss = multi_label_loss(torch.nn.functional.softmax(network_output), target_values)
            loss = manual_b_entropy(network_output, y_1_hot)
            # loss = loss_module(network_output, y_1_hot)

            reg_loss_l1 = 0
            reg_loss_l2 = 0

            for param in network.parameters():
                if config.l1_factor > 0.0:
                    reg_loss_l1 += torch.sum(config.l1_factor * torch.abs(param))
                if config.l2_factor > 0.0:
                    reg_loss_l2 += torch.sum(config.l2_factor * torch.pow(param, 2))

            loss += reg_loss_l1
            loss += reg_loss_l2

            optim.zero_grad()
            loss.backward()

            if config.grad_clip_norm > 0:
                clip_grad_norm(network, config.grad_clip_norm)

            optim.step()

            if print_grads:
                for param, name, module in iterate_params(network):
                    print("{}\t\t {}\t\t: grad norm {}\t\t weight norm {}".format(
                        name, str(module), param.grad.norm(2).data[0],
                        param.norm(2).data[0]))

            losses.append(loss.data[0])

            accuracies.append(binary_accuracy_single(network_output, target_values))

            if total_iterations % 10 == 0:
                logger.log({
                    'iteration': total_iterations,
                    'epoch': epoch,
                    'records': {
                        'training': dict(
                            loss=np.mean(losses[-10:]),
                            acc=np.mean(accuracies[-10:])),
                    }
                })

            total_iterations += 1

        print('Epoch {:3d} | Elapsed training time {:10.3f} | Loss: {:10.6f}'.format(
                epoch, epoch_start_time - timeit.timeit(),
                np.mean(losses)))

        # Validation
        network.eval()
        validate_single_branch(valid_data, network, scaler, logger, total_iterations, epoch)

        # Checkpoint
        ckpt = {'network': network.state_dict(),
                'optim': optim.state_dict()}
        torch.save(ckpt, os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)))
        shutil.copyfile(
            os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)),
            os.path.join(checkpoint_path, 'latest.pt'))


if __name__ == '__main__':
    main()
