#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
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
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from attend_to_detect.dataset import all_classes
from attend_to_detect.model import CNNRNNFastEncoder, CTCModel
from attend_to_detect.evaluation import validate, binary_accuracy
from attend_to_detect.pytorch_dataset import ChallengeDataset

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
    encoder = CNNRNNFastEncoder(**config.encoder_config)

    decoder = torch.nn.Linear(config.network_decoder_dim, len(all_classes) * 3)
    torch.nn.init.constant(decoder.bias.data, -0.)
    torch.nn.init.xavier_uniform(decoder.weight.data)

    model = CTCModel(encoder, decoder)

    # Check if we have GPU, and if we do then GPU them all
    if torch.has_cudnn:
        model.cuda()

    # Create optimizer for all parameters
    optim = config.optimizer(model.parameters(), lr=config.optimizer_lr)

    # Do we have a checkpoint?
    if os.path.isdir(args.checkpoint_path):
        print('Checkpoint directory exists!')
        if os.path.isfile(os.path.join(args.checkpoint_path, 'latest.pt')):
            print('Loading checkpoint...')
            ckpt = torch.load(os.path.join(args.checkpoint_path, 'latest.pt'))
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
    else:
        print('Checkpoint directory does not exist, creating...')
        os.makedirs(args.checkpoint_path)

    if args.train_examples == -1:
        examples = None
    else:
        examples = args.train_examples

    # Load datasets
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    weights = np.load('weights.npy')

    train_dataset = ChallengeDataset(
        dataset_name=config.dataset_full_path,
        that_set='train', scaler=scaler, shuffle_targets=True)
    valid_dataset = ChallengeDataset(
        dataset_name=config.dataset_full_path,
        that_set='test', scaler=scaler, shuffle_targets=False)

    train_sampler = WeightedRandomSampler(weights, len(train_dataset))
    train_data = DataLoader(train_dataset,
                            batch_size=config.batch_size,
                            sampler=train_sampler,
                            collate_fn=ChallengeDataset.collate)
    train_data.get_epoch_iterator = train_data.__iter__
    valid_data = DataLoader(valid_dataset,
                            batch_size=config.batch_size,
                            collate_fn=ChallengeDataset.collate)
    valid_data.get_epoch_iterator = valid_data.__iter__

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
            ['train', 'valid'],
            'loss',
            dict(title=title_losses,
                 xlabel='iteration',
                 ylabel='cross-entropy'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(loss_handler)
        accuracy_handler = VisdomHandler(
            ['train_alarm', 'train_vehicle', 'valid_alarm', 'valid_vehicle'],
            'acc',
            dict(title=title_accu,
                 xlabel='iteration',
                 ylabel='accuracy, %'),
            server=args.visdom_server, port=args.visdom_port)
        logger.handlers.append(accuracy_handler)

    with closing(logger):
        train_loop(
            config, model,
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


def train_loop(config, model, train_data, valid_data, scaler, optim, print_grads, logger,
               checkpoint_path, no_tqdm):
    total_iterations = 0
    validate(valid_data, model, scaler, logger, total_iterations, -1)
    for epoch in range(config.epochs):
        model.train()
        losses = []
        accuracies = []
        epoch_start_time = timeit.timeit()
        epoch_iterator = enumerate(train_data.get_epoch_iterator())
        if not no_tqdm:
            epoch_iterator = tqdm(epoch_iterator,
                                  total=50000 // config.batch_size)
        for iteration, batch in epoch_iterator:
            x, y = batch
            x = Variable(x.cuda(), requires_grad=False)
            y_1_hot = train_data.dataset.one_hot(y, all_classes)
            if torch.has_cudnn:
                y_1_hot = y_1_hot.cuda()
            y_1_hot = Variable(y_1_hot, requires_grad=False)

            # Get input
            outputs = model(x, len(all_classes))
            outputs.register_hook(lambda grad: setattr(outputs, 'gradient', grad))
            print(outputs[0])

            # Calculate losses, do backward passing, and do updates
            loss = model.cost(outputs, y_1_hot)

            optim.zero_grad()
            loss.backward()

            optim.step()

            if print_grads:
                for param, name, module in iterate_params(model):
                    print("{}\t\t {}\t\t: grad norm {}\t\t weight norm {}".format(
                        name, str(module), param.grad.norm(2).data[0],
                        param.norm(2).data[0]))

            losses.append(loss.data[0])

            if False:
                # TODO
                accuracies_alarm.append(binary_accuracy(alarm_output, y_alarm_1_hot))
                accuracies_vehicle.append(binary_accuracy(vehicle_output, y_vehicle_1_hot))

            if total_iterations % 10 == 0:
                logger.log({
                    'iteration': total_iterations,
                    'epoch': epoch,
                    'records': {
                        'train': dict(
                            loss=np.mean(losses[-10:]))}})

            total_iterations += 1

        print('Epoch {:4d} elapsed training time {:10.5f}'
              '\tLosses: alarm: {:10.6f} | vehicle: {:10.6f}'.format(
                epoch, epoch_start_time - timeit.timeit(),
                np.mean(losses), np.mean(losses)))

        # Validation
        validate(valid_data, model, scaler, logger, total_iterations, epoch)

        # Checkpoint
        ckpt = dict(model=model.state_dict(),
                    optim=optim.state_dict())
        torch.save(ckpt, os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)))
        shutil.copyfile(
            os.path.join(checkpoint_path, 'ckpt_{}.pt'.format(epoch)),
            os.path.join(checkpoint_path, 'latest.pt'))


if __name__ == '__main__':
    main()
